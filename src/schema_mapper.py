"""
schema_mapper.py — Phase 2: map any messy raw export to the fixed internal schema.

Step 2.1 (this file, first piece): robust file loading for CSV/Excel exports.
Later steps add column profiling, the LLM mapping call, validation, and rename.

The loader reads every value as a string (dtype=str) so raw fidelity is preserved
for the LLM mapper — type coercion happens later in the cleaning phase (Phase 3).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from config import (
    ALL_FIELDS,
    LLM_SAMPLE_VALUES,
    MAPPING_CONFIDENCE_THRESHOLD,
    MIN_REQUIRED_FOR_ECOMMERCE,
    REQUIRED_FIELDS,
)

def load_raw_file(path: str | Path) -> pd.DataFrame:
    """
    Load a raw CSV or Excel export into a string-typed DataFrame.

    Detects file type by extension (.csv/.tsv/.txt vs .xlsx/.xls), handles
    common encodings and delimiters, strips whitespace from column names, and
    drops fully empty rows/columns.

    Raises FileNotFoundError if the path is missing and ValueError on an
    unsupported extension or unreadable file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv", ".txt"):
        df = _load_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df = _load_excel(path)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Use .csv, .tsv, .txt, .xlsx, or .xls."
        )

    # Normalize: strip column-name whitespace, drop all-empty rows/cols.
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df = df.reset_index(drop=True)

    if df.empty or df.shape[1] == 0:
        raise ValueError(f"File '{path.name}' contains no usable data.")

    return df


# Encodings tried in order for CSV files (covers most real-world exports).
_CSV_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1")


def _sniff_delimiter(sample: str) -> str:
    """Guess a CSV delimiter from a text sample; default to comma."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV, trying several encodings and sniffing the delimiter."""
    last_err: Exception | None = None
    for enc in _CSV_ENCODINGS:
        try:
            with open(path, "r", encoding=enc, newline="") as fh:
                sample = fh.read(4096)
            delimiter = _sniff_delimiter(sample)
            return pd.read_csv(
                path,
                dtype=str,
                sep=delimiter,
                encoding=enc,
                keep_default_na=True,
                skipinitialspace=True,
            )
        except (UnicodeDecodeError, pd.errors.ParserError) as err:
            last_err = err
            continue
    raise ValueError(f"Could not read CSV '{path.name}'")


def _load_excel(path: Path) -> pd.DataFrame:
    """Load the first sheet of an Excel workbook as strings."""
    return pd.read_excel(path, dtype=str, engine="openpyxl")



def profile_columns(
    df: pd.DataFrame, n_samples: int = LLM_SAMPLE_VALUES
) -> list[dict]:
    """
    Summarize each raw column for the LLM schema mapper.

    Returns one dict per column with:
    3.
    .
      - name        : the raw column name
      - samples     : up to n_samples distinct non-null example values (as strings)
      - non_null    : count of populated cells
      - null        : count of empty/NaN cells
    Distinct samples are preferred so the model sees variety (e.g. several
    category values) rather than the same value repeated.
    """
    profile: list[dict] = []
    total = len(df)
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        # distinct values, order-preserving, capped at n_samples
        seen: list[str] = []
        for val in non_null.tolist():
            text = str(val).strip()
            if text and text not in seen:
                seen.append(text)
            if len(seen) >= n_samples:
                break
        profile.append(
            {
                "name": col,
                "samples": seen,
                "non_null": int(non_null.shape[0]),
                "null": int(total - non_null.shape[0]),
            }
        )
    return profile


# ── LLM mapping (Step 2.3) ─────────────────────────────────────────────────

# Every valid internal field plus "none" (unmapped). The model is told to use
# only these; anything else it returns is coerced to "none" in map_columns().
_FIELD_CHOICES = list(ALL_FIELDS.keys()) + ["none"]


class ColumnMapping(BaseModel):
    """One raw-column -> internal-field decision from the LLM."""

    raw_column: str
    internal_field: str  # one of _FIELD_CHOICES (validated post-hoc)
    confidence: float  # 0.0–1.0, the model's own confidence


class ColumnMappingList(BaseModel):
    """Object wrapper so the response is a JSON object (JSON-mode friendly)."""

    mappings: list[ColumnMapping]


def _build_mapping_prompt(profile: list[dict]) -> str:
    """Construct the instruction + column profile sent to the LLM."""
    required = ", ".join(REQUIRED_FIELDS)
    optional = ", ".join(k for k in ALL_FIELDS if k not in REQUIRED_FIELDS)
    lines = [
        "You map messy e-commerce export columns to a FIXED internal schema.",
        "",
        f"REQUIRED internal fields: {required}",
        f"OPTIONAL internal fields: {optional}",
        "",
        "Rules:",
        "- internal_field MUST be exactly one of: " + ", ".join(_FIELD_CHOICES) + ".",
        "- Map each raw column to exactly one internal field, or 'none' if it "
        "matches no field.",
        "- Use the column NAME and the SAMPLE VALUES to decide.",
        "- Never map two raw columns to the same internal field; if several "
        "could fit, pick the best and map the rest to 'none'.",
        "- order_value = the monetary total of an order (may contain $ or commas).",
        "- order_date = when the order was placed; signup_date = account creation.",
        "- Set confidence in [0,1] reflecting how sure you are.",
        "",
        "Raw columns:",
        json.dumps(profile, indent=2),
    ]
    return "\n".join(lines)


def map_columns(profile: list[dict]) -> list[dict]:
    """
    Ask the LLM to map raw columns to internal fields.

    Returns a list of dicts: {raw_column, internal_field, confidence}, where
    internal_field is a plain string (one of the schema fields or "none"). Any
    field the model invents that is not a valid choice is coerced to "none".
    """
    from src.llm import generate_structured

    prompt = _build_mapping_prompt(profile)
    result = generate_structured(prompt, response_schema=ColumnMappingList)

    allowed = set(_FIELD_CHOICES)
    mappings: list[dict] = []
    for item in result.mappings:
        field = item.internal_field if item.internal_field in allowed else "none"
        conf = max(0.0, min(1.0, float(item.confidence)))
        mappings.append(
            {
                "raw_column": item.raw_column,
                "internal_field": field,
                "confidence": round(conf, 3),
            }
        )
    return mappings


# ── Validation + user guidance (Step 2.4) ──────────────────────────────────

def validate_mapping(mappings: list[dict]) -> dict:
    """
    Validate the LLM's column mapping and produce a structured report with
    plain-English guidance for the user.

    Returns a dict:
      status            : "ok" | "needs_confirmation" | "rejected"
      mapping           : {raw_column -> internal_field} for confident, mapped cols
      uncertain         : [{raw_column, internal_field, confidence}] below threshold
      unmapped_columns  : [raw_column, ...] mapped to "none"
      missing_required  : [internal_field, ...] required fields not confidently found
      duplicate_targets : {internal_field: [raw_column, ...]} mapped more than once
      messages          : [str] human-readable explanations
      suggestions       : [str] actionable next steps for the user
    """
    confident: dict[str, str] = {}          # raw -> field (>= threshold, not none)
    uncertain: list[dict] = []
    unmapped: list[str] = []
    target_to_raws: dict[str, list[str]] = {}

    for m in mappings:
        raw, field, conf = m["raw_column"], m["internal_field"], m["confidence"]
        if field == "none":
            unmapped.append(raw)
            continue
        if conf < MAPPING_CONFIDENCE_THRESHOLD:
            uncertain.append(m)
            continue
        confident[raw] = field
        target_to_raws.setdefault(field, []).append(raw)

    # Duplicate targets: same internal field claimed by >1 confident column.
    duplicate_targets = {f: raws for f, raws in target_to_raws.items() if len(raws) > 1}

    mapped_required = [f for f in REQUIRED_FIELDS if f in target_to_raws]
    missing_required = [f for f in REQUIRED_FIELDS if f not in target_to_raws]

    messages: list[str] = []
    suggestions: list[str] = []

    # --- Decide status ---
    if len(mapped_required) < MIN_REQUIRED_FOR_ECOMMERCE:
        status = "rejected"
        messages.append(
            "This file does not look like e-commerce order data. This tool needs "
            "customer orders with, at minimum, a customer ID, an order ID, an "
            "order date, and an order value."
        )
        if mapped_required:
            messages.append(
                "Only recognized: " + ", ".join(mapped_required) + "."
            )
        messages.append("Missing required fields: " + ", ".join(missing_required) + ".")
        suggestions.append(
            "Upload an e-commerce export (Shopify, WooCommerce, or similar) that "
            "includes one row per order."
        )
        suggestions.append(
            "Or rename/add columns so the file contains: "
            + ", ".join(REQUIRED_FIELDS) + "."
        )
    elif missing_required or uncertain or duplicate_targets:
        status = "needs_confirmation"
        if missing_required:
            messages.append(
                "Could not confidently find these required fields: "
                + ", ".join(missing_required) + "."
            )
            suggestions.append(
                "Tell me which uploaded column holds "
                + " / ".join(missing_required)
                + ", or add it to the file."
            )
        if uncertain:
            for u in uncertain:
                messages.append(
                    f"Low confidence ({u['confidence']}): '{u['raw_column']}' might "
                    f"be '{u['internal_field']}'. Please confirm or correct."
                )
            suggestions.append("Confirm or fix the low-confidence mappings above.")
        if duplicate_targets:
            for field, raws in duplicate_targets.items():
                messages.append(
                    f"Multiple columns mapped to '{field}': "
                    + ", ".join(raws) + ". Pick one."
                )
            suggestions.append("Choose a single column for each duplicated field.")
    else:
        status = "ok"
        messages.append("All required fields mapped with high confidence.")

    return {
        "status": status,
        "mapping": confident,
        "uncertain": uncertain,
        "unmapped_columns": unmapped,
        "missing_required": missing_required,
        "duplicate_targets": duplicate_targets,
        "messages": messages,
        "suggestions": suggestions,
    }


# ── Edge-case helpers (Step 2.5) ───────────────────────────────────────────

def auto_resolve_duplicates(mappings: list[dict]) -> list[dict]:
    """
    When two columns map to the same internal field, keep the highest-confidence
    one and demote the rest to "none". Returns a NEW mappings list (input
    untouched). Useful as a default before showing the user the confirm UI.
    """
    best_for_field: dict[str, dict] = {}
    for m in mappings:
        field = m["internal_field"]
        if field == "none":
            continue
        cur = best_for_field.get(field)
        if cur is None or m["confidence"] > cur["confidence"]:
            best_for_field[field] = m

    winners = {id(m) for m in best_for_field.values()}
    resolved: list[dict] = []
    for m in mappings:
        if m["internal_field"] != "none" and id(m) not in winners:
            resolved.append({**m, "internal_field": "none"})  # demoted loser
        else:
            resolved.append({**m})
    return resolved


def apply_overrides(mappings: list[dict], overrides: dict[str, str]) -> list[dict]:
    """
    Apply user corrections from the confirm UI.

    `overrides` maps raw_column -> internal_field (or "none"). Overridden entries
    get confidence 1.0 (the human is authoritative). Returns a new list.
    """
    valid_targets = set(ALL_FIELDS) | {"none"}
    result: list[dict] = []
    for m in mappings:
        raw = m["raw_column"]
        if raw in overrides:
            target = overrides[raw]
            if target not in valid_targets:
                raise ValueError(
                    f"Invalid override '{target}' for column '{raw}'. "
                    f"Must be one of {sorted(valid_targets)}."
                )
            result.append({"raw_column": raw, "internal_field": target, "confidence": 1.0})
        else:
            result.append({**m})
    return result


def detect_date_format(values: list[str]) -> dict:
    """
    Probe a column's date values to learn how to parse them in Phase 3.

    Returns:
      parseable_fraction : 0–1 share of non-empty values pandas can parse
      dayfirst           : True if day-first parsing fits clearly better
      strategy           : suggested pandas approach ("mixed" by default)
      unparseable_examples : up to 3 values that failed to parse
    """
    cleaned = [str(v).strip() for v in values if str(v).strip()]
    if not cleaned:
        return {
            "parseable_fraction": 0.0,
            "dayfirst": False,
            "strategy": "mixed",
            "unparseable_examples": [],
        }

    series = pd.Series(cleaned)
    default = pd.to_datetime(series, format="mixed", errors="coerce", dayfirst=False)
    dayfirst = pd.to_datetime(series, format="mixed", errors="coerce", dayfirst=True)

    default_ok = default.notna().mean()
    dayfirst_ok = dayfirst.notna().mean()
    use_dayfirst = dayfirst_ok > default_ok

    best = dayfirst if use_dayfirst else default
    parseable = float(best.notna().mean())
    unparseable = series[best.isna()].head(3).tolist()

    return {
        "parseable_fraction": round(parseable, 3),
        "dayfirst": bool(use_dayfirst),
        "strategy": "mixed",
        "unparseable_examples": unparseable,
    }


# ── Apply rename (Step 2.6) ────────────────────────────────────────────────

def apply_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """
    Rename raw columns to internal field names and drop everything unmapped.

    `mapping` is {raw_column -> internal_field} (the confident mapping from the
    validation report, post-overrides). Returns a new DataFrame whose columns
    are exactly the mapped internal fields, ordered required-first then optional.

    Raises ValueError if `mapping` points two raw columns at the same internal
    field, or references a raw column not present in `df`.
    """
    # Guard: no duplicate internal targets.
    targets = [f for f in mapping.values() if f != "none"]
    dupes = {t for t in targets if targets.count(t) > 1}
    if dupes:
        raise ValueError(f"Mapping has duplicate internal targets: {sorted(dupes)}")

    rename: dict[str, str] = {}
    for raw, field in mapping.items():
        if field == "none":
            continue
        if raw not in df.columns:
            raise ValueError(f"Mapped column '{raw}' not found in the data.")
        rename[raw] = field

    normalized = df[list(rename.keys())].rename(columns=rename)

    # Order columns: required fields first, then optional, both in schema order.
    ordered = [f for f in ALL_FIELDS if f in normalized.columns]
    return normalized[ordered].reset_index(drop=True)


# ── Orchestration / human-confirm hook (Step 2.7) ──────────────────────────

def run_schema_mapping(path: str | Path) -> dict:
    """
    Run the full mapping flow for a raw file and return everything the confirm
    UI needs.

    Returns a dict:
      raw_df          : the loaded string DataFrame
      profile         : per-column profile (names + samples + null counts)
      mappings        : the LLM mappings after auto duplicate-resolution
      report          : validate_mapping() report (status, messages, suggestions)
      editable_mapping: {raw_column -> internal_field} proposed defaults for the
                        UI dropdowns (every raw column included, "none" if unmapped)
      field_choices   : valid dropdown options (all internal fields + "none")
    """
    raw_df = load_raw_file(path)
    profile = profile_columns(raw_df)
    mappings = auto_resolve_duplicates(map_columns(profile))
    report = validate_mapping(mappings)

    editable = {m["raw_column"]: m["internal_field"] for m in mappings}
    return {
        "raw_df": raw_df,
        "profile": profile,
        "mappings": mappings,
        "report": report,
        "editable_mapping": editable,
        "field_choices": list(ALL_FIELDS.keys()) + ["none"],
    }


def finalize_mapping(
    raw_df: pd.DataFrame,
    mappings: list[dict],
    overrides: dict[str, str] | None = None,
) -> dict:
    """
    Apply the user's confirmed mapping (with optional overrides), re-validate,
    and produce the normalized DataFrame when the result is acceptable.

    Returns a dict:
      status        : final validate_mapping status
      report        : final validation report
      normalized_df : schema-normalized DataFrame, or None if status=="rejected"
    """
    final = apply_overrides(mappings, overrides) if overrides else mappings
    final = auto_resolve_duplicates(final)
    report = validate_mapping(final)

    normalized_df = None
    if report["status"] != "rejected":
        normalized_df = apply_mapping(raw_df, report["mapping"])

    return {
        "status": report["status"],
        "report": report,
        "normalized_df": normalized_df,
    }


if __name__ == "__main__":
    # Quick manual check against the sample fixture.
    from config import RAW_DIR

    sample = load_raw_file(RAW_DIR / "sample_messy.csv")
    print(f"Loaded shape: {sample.shape}")
    print("Columns:", list(sample.columns))
    print(sample.head(3).to_string(index=False))
    print("\n--- column profile ---")
    for entry in profile_columns(sample):
        print(entry)