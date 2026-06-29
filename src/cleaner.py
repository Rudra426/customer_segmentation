"""
cleaner.py — Phase 3: clean and validate the schema-normalized DataFrame.

Input is the output of schema_mapper.apply_mapping(): correct internal column
names but raw string values (e.g. "$57.17", "Jan 01 2024"). This module coerces
types, removes bad rows, standardizes dates, and emits a validation report.

Built step by step: 3.1 type coercion (this piece) -> ... -> 3.7 report.
"""

from __future__ import annotations

import re

import pandas as pd

from config import ALL_FIELDS, OPTIONAL_FIELDS, REQUIRED_FIELDS

# Currency symbols / thousands separators stripped before float parsing.
_MONEY_RE = re.compile(r"[^0-9.\-]")


def parse_money(value) -> float:
    """
    Parse a messy money string to a float.

    Handles "$1,250.00", "1250.0", "-$45.00", "99.5". Returns NaN for blanks or
    anything with no numeric content. A leading '-' anywhere is treated as
    negative (refunds), but only one minus sign is honored.
    """
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    negative = "-" in text
    cleaned = _MONEY_RE.sub("", text)
    if cleaned in ("", ".", "-"):
        return float("nan")
    cleaned = cleaned.replace("-", "")  # strip stray minus; sign applied below
    try:
        num = float(cleaned)
    except ValueError:
        return float("nan")
    return -num if negative else num


# ID normalization (smart numeric-core matching) ──────────────────────────────
_ID_DIGITS_RE = re.compile(r"\d+")
_ID_NONALNUM_RE = re.compile(r"[^a-z0-9]")


def canonical_customer_id(value) -> str | None:
    """
    Reduce a messy customer id to a canonical identity key (smart matching).

    Different formats of the same id collapse to one key: text prefixes, casing,
    separators, and zero-padding are ignored, and the TRAILING run of digits is
    treated as the identity. So "cus00001", "0001", and "client001" all map to
    "1"; "AB-12" and "xy12" both map to "12". Ids with no digits fall back to
    their lowercased alphanumerics ("John Doe" -> "johndoe").

    Returns None for blanks/NA (those rows are dropped later as missing required).
    """
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text or text in ("nan", "<na>", "none", "null"):
        return None
    digit_groups = _ID_DIGITS_RE.findall(text)
    if digit_groups:
        return str(int(digit_groups[-1]))  # last numeric block, zero-padding stripped
    alnum = _ID_NONALNUM_RE.sub("", text)
    return alnum or None


def normalize_customer_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Unify customer_id values that refer to the same customer in different formats.

    Rows whose customer_id shares a canonical key (see canonical_customer_id) are
    rewritten to a single representative id — the first original value seen for
    that key — so downstream grouping (RFM, clustering) treats them as one
    customer while the displayed id stays human-readable.

    Returns the rewritten frame and a report:
      {"applied", "ids_before", "ids_after", "merged_groups", "examples"}.
    examples lists up to 5 merged groups as {"merged_to", "variants"}.
    """
    empty = {"applied": False, "ids_before": 0, "ids_after": 0,
             "merged_groups": 0, "examples": []}
    if "customer_id" not in df.columns:
        return df, empty

    out = df.copy()
    originals = out["customer_id"].tolist()
    keys = [canonical_customer_id(v) for v in originals]

    representative: dict[str, object] = {}
    variants: dict[str, list] = {}
    for orig, key in zip(originals, keys):
        if key is None:
            continue
        representative.setdefault(key, orig)
        bucket = variants.setdefault(key, [])
        if orig not in bucket:
            bucket.append(orig)

    out["customer_id"] = pd.Series(
        [representative.get(k, o) if k is not None else o
         for o, k in zip(originals, keys)],
        index=out.index,
        dtype="string",
    )

    merged = {k: v for k, v in variants.items() if len(v) > 1}
    examples = [
        {"merged_to": representative[k], "variants": v[:6]}
        for k, v in list(merged.items())[:5]
    ]
    distinct_originals = len({o for o, k in zip(originals, keys) if k is not None})
    report = {
        "applied": True,
        "ids_before": distinct_originals,  # distinct original id strings
        "ids_after": len(variants),        # distinct canonical identities
        "merged_groups": len(merged),      # identities that had >1 original format
        "examples": examples,
    }
    return out, report


# Common currency symbols and a few ISO codes used for the v1 heuristic check.
_CURRENCY_SYMBOLS = {
    "$": "USD/Dollar", "€": "EUR", "£": "GBP", "¥": "JPY/CNY",
    "₹": "INR", "₩": "KRW", "₽": "RUB", "R$": "BRL",
}
_CURRENCY_CODES = re.compile(
    r"\b(USD|EUR|GBP|JPY|CNY|INR|AUD|CAD|KRW|RUB|BRL|CHF)\b"
)


def detect_multicurrency(raw_df: pd.DataFrame) -> dict:
    """
    Detect whether the raw order_value column mixes currencies (v1: flag only).

    Scans the RAW string values (run this BEFORE coerce_types, which strips
    symbols). Returns a dict:
      {"currencies_found": [labels], "mixed": bool, "warning": str | None}.
    A single currency (or none detected) yields warning=None.
    """
    found: set[str] = set()
    if "order_value" in raw_df.columns:
        values = raw_df["order_value"].dropna().astype(str)
        for sym, label in _CURRENCY_SYMBOLS.items():
            if values.str.contains(re.escape(sym), regex=True).any():
                found.add(label)
        for code in values.str.findall(_CURRENCY_CODES).explode().dropna().unique():
            found.add(str(code))

    currencies = sorted(found)
    mixed = len(currencies) > 1
    warning = None
    if mixed:
        warning = (
            "Multiple currencies detected (" + ", ".join(currencies) + "). v1 does "
            "NOT convert currencies — monetary features will mix them and may be "
            "misleading. Convert to a single currency before uploading."
        )
    return {"currencies_found": currencies, "mixed": mixed, "warning": warning}


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce each present internal field to its schema dtype.

    - datetime fields  -> pandas datetime (format="mixed"), unparseable -> NaT
    - order_value      -> float via parse_money
    - int fields       -> nullable Int64 (unparseable -> <NA>)
    - string fields    -> stripped string, empty -> NA

    Unparseable values become NaN/NaT/<NA> and are dealt with in later steps.
    Returns a new DataFrame; the input is not modified.
    """
    out = df.copy()

    for col in out.columns:
        dtype = ALL_FIELDS.get(col)
        if dtype == "datetime":
            out[col] = pd.to_datetime(
                out[col], format="mixed", errors="coerce", dayfirst=False
            )
        elif dtype == "float":
            out[col] = out[col].map(parse_money).astype("float64")
        elif dtype == "int":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
        elif dtype == "string":
            stripped = out[col].astype("string").str.strip()
            out[col] = stripped.replace({"": pd.NA})

    return out


def drop_null_required(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Drop rows missing any required field; report per-field null counts.

    A row is removed if ANY required field present in `df` is null (this is also
    where dates/values that failed coercion in 3.1 get pruned). Returns the
    cleaned frame and a dict: {"per_field": {field: n_null}, "rows_dropped": n}.
    """
    present_required = [f for f in REQUIRED_FIELDS if f in df.columns]
    per_field = {f: int(df[f].isna().sum()) for f in present_required}

    before = len(df)
    cleaned = df.dropna(subset=present_required).reset_index(drop=True)
    rows_dropped = before - len(cleaned)

    return cleaned, {"per_field": per_field, "rows_dropped": rows_dropped}


def drop_duplicate_orders(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Remove duplicate order_id rows, keeping the first occurrence.

    Returns the deduped frame and a dict:
      {"duplicates_removed": n, "duplicate_ids": [up to 10 example ids]}.
    No-op (with order_id absent) is handled gracefully.
    """
    if "order_id" not in df.columns:
        return df.reset_index(drop=True), {"duplicates_removed": 0, "duplicate_ids": []}

    dup_mask = df["order_id"].duplicated(keep="first")
    duplicate_ids = df.loc[dup_mask, "order_id"].unique().tolist()[:10]

    cleaned = df[~dup_mask].reset_index(drop=True)
    return cleaned, {
        "duplicates_removed": int(dup_mask.sum()),
        "duplicate_ids": duplicate_ids,
    }


def drop_nonpositive_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Remove orders whose order_value is <= 0 (refunds, data errors).

    Returns the cleaned frame and a dict:
      {"negative_removed": n, "zero_removed": n, "rows_dropped": n}.
    Negative and zero are counted separately for the report.
    """
    if "order_value" not in df.columns:
        return df.reset_index(drop=True), {
            "negative_removed": 0,
            "zero_removed": 0,
            "rows_dropped": 0,
        }

    negative = int((df["order_value"] < 0).sum())
    zero = int((df["order_value"] == 0).sum())

    cleaned = df[df["order_value"] > 0].reset_index(drop=True)
    return cleaned, {
        "negative_removed": negative,
        "zero_removed": zero,
        "rows_dropped": negative + zero,
    }


def standardize_dates(
    df: pd.DataFrame, reference: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Normalize datetime fields and remove impossible future-dated orders.

    - Strips any timezone so all datetimes are tz-naive and comparable.
    - Drops rows whose order_date is AFTER `reference` (default: today), since a
      future purchase date is invalid and would break recency.
    - Counts (but keeps) future signup_date values as a warning.

    Returns the cleaned frame and a dict:
      {"reference_date": iso, "future_orders_removed": n, "future_signups": n}.
    """
    out = df.copy()
    if reference is None:
        reference = pd.Timestamp.now().normalize()

    date_cols = [c for c in ("order_date", "signup_date") if c in out.columns]
    for col in date_cols:
        # Drop timezone if the column happens to be tz-aware.
        if isinstance(out[col].dtype, pd.DatetimeTZDtype):
            out[col] = out[col].dt.tz_localize(None)

    future_orders_removed = 0
    if "order_date" in out.columns:
        future_mask = out["order_date"] > reference
        future_orders_removed = int(future_mask.sum())
        out = out[~future_mask].reset_index(drop=True)

    future_signups = 0
    if "signup_date" in out.columns:
        future_signups = int((out["signup_date"] > reference).sum())

    return out, {
        "reference_date": reference.date().isoformat(),
        "future_orders_removed": future_orders_removed,
        "future_signups": future_signups,
    }


def clean_data(
    raw_normalized: pd.DataFrame, reference: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Run the full Phase 3 cleaning pipeline and build a validation report.

    Order of operations:
      1. detect_multicurrency  (on raw strings, before symbols are stripped)
      2. coerce_types
      3. normalize_customer_ids (unify same-customer ids in mixed formats)
      4. drop_null_required
      5. drop_duplicate_orders
      6. drop_nonpositive_values
      7. standardize_dates

    Returns (clean_df, report). `report` consolidates every count, warning, the
    resulting date range, optional-field null counts, and customer/order totals.
    """
    rows_in = len(raw_normalized)
    warnings: list[str] = []

    currency = detect_multicurrency(raw_normalized)
    if currency["warning"]:
        warnings.append(currency["warning"])

    df = coerce_types(raw_normalized)
    df, id_info = normalize_customer_ids(df)
    if id_info["merged_groups"]:
        warnings.append(
            f"Merged {id_info['merged_groups']} customer-id format variant group(s) "
            f"({id_info['ids_before']} raw ids -> {id_info['ids_after']} customers). "
            "Different formats of the same id (prefix/zero-padding/case) were unified."
        )
    df, null_info = drop_null_required(df)
    df, dup_info = drop_duplicate_orders(df)
    df, pos_info = drop_nonpositive_values(df)
    df, date_info = standardize_dates(df, reference=reference)

    rows_out = len(df)

    # Optional-field null counts (informational, rows kept).
    optional_nulls = {
        f: int(df[f].isna().sum()) for f in OPTIONAL_FIELDS if f in df.columns
    }

    date_range = None
    if "order_date" in df.columns and rows_out > 0:
        date_range = {
            "min": df["order_date"].min().date().isoformat(),
            "max": df["order_date"].max().date().isoformat(),
            "reference": date_info["reference_date"],
        }

    if rows_out == 0:
        warnings.append("No rows survived cleaning — check the source data quality.")

    report = {
        "rows_in": rows_in,
        "rows_out": rows_out,
        "rows_dropped_total": rows_in - rows_out,
        "drops": {
            "null_required": null_info,
            "duplicate_orders": dup_info,
            "nonpositive_values": pos_info,
            "future_orders": date_info["future_orders_removed"],
        },
        "currency": currency,
        "id_normalization": id_info,
        "date_range": date_range,
        "optional_field_nulls": optional_nulls,
        "n_customers": int(df["customer_id"].nunique()) if "customer_id" in df else 0,
        "n_orders": rows_out,
        "warnings": warnings,
    }
    return df, report


if __name__ == "__main__":
    import json
    import sys

    sys.path.insert(0, ".")
    from config import RAW_DIR
    from src.schema_mapper import apply_mapping, load_raw_file

    raw = load_raw_file(RAW_DIR / "sample_messy.csv")
    mapping = {
        "Cust ID": "customer_id",
        "Order #": "order_id",
        "Order Date": "order_date",
        "Total $": "order_value",
        "SKU Category": "product_category",
        "Qty": "quantity",
    }
    normalized = apply_mapping(raw, mapping)
    clean_df, report = clean_data(normalized)
    print(json.dumps(report, indent=2))
    print("clean rows:", len(clean_df))
