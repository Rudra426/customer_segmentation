"""
validation.py — Phase 6.5 (QA gate): independent validation of segmentation
outputs, run AFTER feature-engineering / clustering / persona labeling and
BEFORE the results are trusted by the dashboard, scoring API, or CSV export.

Philosophy: FAIL LOUDLY. Each check recomputes ground truth from the source
data instead of trusting upstream numbers, so a fabricated, trivial, or
inconsistent result is caught rather than shipped as a plausible-but-wrong
segmentation. Nothing here auto-fixes — checks only surface problems. The gate
HALTS the pipeline (non-zero exit) on any failure so a human signs off.

Each check is an independent function returning ``(passed: bool, detail: dict)``
where ``detail`` always carries ``threshold`` and ``actual`` for the report.
``run_validation()`` executes the applicable checks, writes
``validation_report.json``, prints a console summary, and returns the report
with an ``overall_pass`` flag.

All thresholds come from ``config.VALIDATION`` (never hardcoded here).
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

from config import VALIDATION, VALIDATION_REPORT_PATH, ensure_dirs
from src.cleaner import (
    canonical_customer_id,
    normalize_category,
    normalize_order_id,
    parse_money,
)

# Normalized-key pattern for fuzzy order-id comparison (drops whitespace/format).
_ALNUM_RE = re.compile(r"[^a-z0-9]")
# Missing/sentinel tokens treated as "no quantity" by the qty sanity classifier.
_QTY_MISSING = frozenset({"", "nan", "<na>", "none", "null", "n/a", "na", "unknown"})


# ── small helpers ──────────────────────────────────────────────────────────
def _round(value, ndigits: int = 4):
    """Round to a JSON-friendly float; map NaN/inf/None to None, pass through rest."""
    try:
        if value is None:
            return None
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, ndigits)
    except (TypeError, ValueError):
        return value


def _key(index_value) -> str:
    """Stringify a (possibly non-str) DataFrame index label for the JSON report."""
    return str(index_value)


# ── input assembly (shared by the CLI gate and the dashboard) ──────────────
def build_category_counts(
    clean_df: pd.DataFrame,
    customer_index: pd.Index,
    id_col: str = "customer_id",
    category_col: str = "product_category",
) -> pd.DataFrame:
    """
    Per-customer ORDER COUNTS by category (one column per category, plus an
    "Uncategorized" bucket for null categories).

    Built from the cleaned orders so the columns sum, per customer, to that
    customer's order frequency — exactly the invariant check_category_consistency
    verifies. (The clustering feature table stores a one-hot of the TOP category,
    which does NOT sum to frequency, so we recompute real counts here.)
    """
    cc = clean_df[[id_col, category_col]].copy()
    cc[category_col] = cc[category_col].astype("string").fillna("Uncategorized")
    counts = cc.groupby([id_col, category_col]).size().unstack(fill_value=0)
    counts.columns = [f"catcount_{c}" for c in counts.columns]
    return counts.reindex(customer_index, fill_value=0)


def prepare_validation_inputs(
    *,
    clean_df: pd.DataFrame,
    cluster_result: pd.DataFrame,
    X,
    cluster_labels,
    raw_orders_df: pd.DataFrame,
    labeled_df: pd.DataFrame | None = None,
    stored_silhouette=None,
) -> dict:
    """
    Assemble the keyword args for run_validation() from pipeline outputs.

    ``cluster_result`` is run_clustering()["result"]; ``labeled_df`` is the
    label_customers() output (with persona/action/priority) when available.
    Per-category count columns are attached so category_consistency is meaningful.
    """
    df = (labeled_df if labeled_df is not None else cluster_result).copy()
    catcounts = build_category_counts(clean_df, df.index)
    df = df.join(catcounts)
    return {
        "df": df,
        "X": X,
        "cluster_labels": cluster_labels,
        "raw_orders_df": raw_orders_df,
        "cat_cols": list(catcounts.columns),
        "stored_silhouette": stored_silhouette,
    }


# ── 1) cluster separation ──────────────────────────────────────────────────
def check_cluster_separation(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    rfm_cols=None,
    min_variance_ratio: float = 2.0,
) -> tuple[bool, dict]:
    """
    FAIL if clusters do not differ meaningfully on RFM (random/fake clustering).

    Computes a one-way ANOVA F-statistic (between-cluster vs within-cluster
    variance ratio) for each RFM feature across the cluster groups. A genuine
    segmentation separates strongly on at least one RFM dimension (F well above
    1); random labels give F ~ 1. Passes if the maximum F-stat across the RFM
    columns meets ``min_variance_ratio``.
    """
    rfm_cols = list(rfm_cols) if rfm_cols else ["recency", "frequency", "monetary"]
    present = [c for c in rfm_cols if c in df.columns]
    detail: dict = {
        "cluster_col": cluster_col,
        "rfm_cols": present,
        "threshold": min_variance_ratio,
        "per_feature": {},
    }
    if cluster_col not in df.columns or not present:
        detail["error"] = "missing cluster column or RFM columns"
        detail["actual"] = None
        return False, detail

    groups = list(df.groupby(cluster_col).groups.values())
    detail["n_clusters"] = len(groups)
    if len(groups) < 2:
        detail["error"] = f"need >=2 clusters, found {len(groups)}"
        detail["actual"] = None
        return False, detail

    f_by_feature: dict[str, float] = {}
    for col in present:
        samples = [df.loc[idx, col].astype(float).to_numpy() for idx in groups]
        samples = [s for s in samples if len(s) > 0]
        try:
            f_stat, p_value = f_oneway(*samples)
        except (ValueError, FloatingPointError):
            f_stat, p_value = float("nan"), float("nan")
        f_by_feature[col] = float(f_stat)
        detail["per_feature"][col] = {"f_stat": _round(f_stat), "p_value": _round(p_value)}

    valid = [v for v in f_by_feature.values() if not math.isnan(v)]
    max_f = max(valid) if valid else 0.0
    detail["max_f_stat"] = _round(max_f)
    detail["actual"] = detail["max_f_stat"]
    return bool(max_f >= min_variance_ratio), detail


# ── 2) silhouette score ────────────────────────────────────────────────────
def check_silhouette_score(
    X,
    cluster_labels,
    min_score: float = 0.25,
    stored_score=None,
    placeholder_tol: float = 0.05,
) -> tuple[bool, dict]:
    """
    Recompute the silhouette score independently and FAIL if it is below
    ``min_score`` or if a stored/claimed score does not match the recomputed one
    (a placeholder/fabricated metric). Silhouette needs 2 <= k <= n-1.
    """
    detail: dict = {"threshold": min_score, "stored_score": _round(stored_score)}
    X = np.asarray(X, dtype=float)
    labels = np.asarray(cluster_labels)
    n = len(labels)
    k = len(set(labels.tolist()))
    detail["n_samples"] = int(n)
    detail["n_clusters"] = int(k)

    if k < 2 or k > n - 1:
        detail["error"] = f"silhouette undefined for k={k}, n={n}"
        detail["actual"] = None
        return False, detail

    score = float(silhouette_score(X, labels))
    detail["recomputed_score"] = _round(score)
    detail["actual"] = detail["recomputed_score"]
    passed = score >= min_score

    if stored_score is not None:
        mismatch = abs(score - float(stored_score)) > placeholder_tol
        detail["stored_matches_recomputed"] = not mismatch
        if mismatch:
            detail["placeholder_warning"] = (
                f"stored silhouette {stored_score} does not match recomputed "
                f"{round(score, 4)} (tol {placeholder_tol}); claimed metric may "
                "be a placeholder / not actually computed"
            )
            passed = False

    return bool(passed), detail


# ── 3) label independence ──────────────────────────────────────────────────
def check_label_independence(
    df: pd.DataFrame, col_a: str = "segment", col_b: str = "persona"
) -> tuple[bool, dict]:
    """
    FAIL if ``col_b`` is a 1:1 (bijective) relabel of ``col_a`` — i.e. the second
    labeling adds zero information beyond the first. A legitimate labeling
    coarsens segments (several segments share a persona), so the relationship is
    NOT bijective. Generalizes to any two label columns.
    """
    detail: dict = {"col_a": col_a, "col_b": col_b}
    if col_a not in df.columns or col_b not in df.columns:
        detail["skipped"] = True
        detail["error"] = "one or both columns absent"
        detail["actual"] = None
        detail["threshold"] = "non-bijective"
        return True, detail  # not applicable -> do not block

    sub = df[[col_a, col_b]].dropna()
    if sub.empty:
        detail["skipped"] = True
        detail["error"] = "no non-null rows to compare"
        detail["actual"] = None
        detail["threshold"] = "non-bijective"
        return True, detail

    ct = pd.crosstab(sub[col_a], sub[col_b])
    n_a, n_b = ct.shape
    max_b_per_a = int((ct > 0).sum(axis=1).max())
    max_a_per_b = int((ct > 0).sum(axis=0).max())
    bijective = (n_a == n_b) and (max_b_per_a == 1) and (max_a_per_b == 1)

    detail.update(
        {
            "n_distinct_a": int(n_a),
            "n_distinct_b": int(n_b),
            "max_b_per_a": max_b_per_a,
            "max_a_per_b": max_a_per_b,
            "bijective": bool(bijective),
            "normalized_mutual_info": _round(
                normalized_mutual_info_score(
                    sub[col_a].astype(str), sub[col_b].astype(str)
                )
            ),
            "threshold": "non-bijective (col_b must not be a 1:1 relabel of col_a)",
            "actual": "bijective_1to1" if bijective else "adds_grouping",
        }
    )
    return (not bijective), detail


# ── 4) category consistency ────────────────────────────────────────────────
def check_category_consistency(
    df: pd.DataFrame,
    cat_cols,
    freq_col: str = "frequency",
    tolerance: float = 0,
    max_bad_frac: float = 0.05,
) -> tuple[bool, dict]:
    """
    FAIL if the per-category count columns do not sum to ``freq_col`` for more
    than ``max_bad_frac`` of rows (fabricated/inconsistent category data).
    Reports the worst-offending rows.
    """
    cat_cols = list(cat_cols)
    detail: dict = {
        "freq_col": freq_col,
        "tolerance": tolerance,
        "threshold": max_bad_frac,
        "cat_cols": cat_cols,
    }
    missing = [c for c in cat_cols + [freq_col] if c not in df.columns]
    if missing or not cat_cols:
        detail["error"] = f"missing columns: {missing or 'no cat_cols'}"
        detail["actual"] = None
        return False, detail

    cat_sum = df[cat_cols].sum(axis=1).astype(float)
    freq = df[freq_col].astype(float)
    diff = (cat_sum - freq).abs()
    bad_mask = diff > tolerance
    bad_frac = float(bad_mask.mean())

    worst = diff[bad_mask].sort_values(ascending=False).head(10)
    detail.update(
        {
            "bad_row_count": int(bad_mask.sum()),
            "total_rows": int(len(df)),
            "bad_row_frac": _round(bad_frac),
            "actual": _round(bad_frac),
            "worst_offenders": [
                {
                    "id": _key(i),
                    "cat_sum": _round(cat_sum.loc[i]),
                    "freq": _round(freq.loc[i]),
                    "diff": _round(d),
                }
                for i, d in worst.items()
            ],
        }
    )
    return bool(bad_frac <= max_bad_frac), detail


# ── 5) top-category accuracy ───────────────────────────────────────────────
def check_top_category_accuracy(
    df: pd.DataFrame,
    top_category_col: str,
    raw_orders_df: pd.DataFrame,
    id_col: str,
    raw_category_col: str,
    min_match_rate: float = 0.40,
) -> tuple[bool, dict]:
    """
    Recompute each customer's true top category directly from raw orders (mode of
    the normalized category) and compare to the claimed ``top_category_col``.
    FAIL if the match rate is at/below random chance (1/n_categories) or below
    ``min_match_rate`` — this specifically catches fabricated category assignment.
    """
    detail: dict = {"top_category_col": top_category_col, "min_match_rate": min_match_rate}
    if top_category_col not in df.columns:
        detail["error"] = "missing top_category_col"
        detail["actual"] = None
        detail["threshold"] = min_match_rate
        return False, detail

    raw = raw_orders_df[[id_col, raw_category_col]].copy()
    raw["_cat"] = raw[raw_category_col].map(normalize_category)
    # Resolve identity to the canonical customer so mixed id formats aggregate
    # into one customer (same entity resolution the pipeline applies).
    raw["_id"] = raw[id_col].map(canonical_customer_id)
    raw = raw.dropna(subset=["_cat", "_id"])
    n_categories = int(raw["_cat"].nunique())
    baseline = 1.0 / max(n_categories, 1)

    # True top category per canonical customer (mode; ties -> alphabetical).
    counts = raw.groupby(["_id", "_cat"]).size().reset_index(name="n")
    counts = counts.sort_values(["_id", "n", "_cat"], ascending=[True, False, True])
    true_top = counts.groupby("_id").first()["_cat"]

    # Align the claimed top_category (df indexed by representative id) onto the
    # same canonical key before comparing.
    claimed_by_canon: dict = {}
    for idx, val in df[top_category_col].items():
        canon = canonical_customer_id(idx)
        if canon is not None:
            claimed_by_canon[canon] = val
    joined = pd.DataFrame({"claimed": pd.Series(claimed_by_canon)})
    joined["true"] = true_top.reindex(joined.index)
    comparable = joined.dropna(subset=["true", "claimed"])

    threshold = max(min_match_rate, baseline)
    detail.update(
        {"n_categories": n_categories, "chance_baseline": _round(baseline), "threshold": _round(threshold)}
    )
    if comparable.empty:
        detail["error"] = "no comparable customers (no overlap of ids/categories)"
        detail["actual"] = None
        return False, detail

    matches = comparable["true"].astype(str) == comparable["claimed"].astype(str)
    match_rate = float(matches.mean())
    detail.update(
        {
            "match_rate": _round(match_rate),
            "actual": _round(match_rate),
            "n_compared": int(len(comparable)),
            "n_mismatched": int((~matches).sum()),
        }
    )
    passed = match_rate > baseline and match_rate >= min_match_rate
    return bool(passed), detail


# ── 6) frequency completeness ──────────────────────────────────────────────
def check_frequency_completeness(
    df: pd.DataFrame,
    freq_col: str,
    raw_orders_df: pd.DataFrame,
    id_col: str,
    canonical_id_col: str,
    tolerance: float = 0.05,
    order_id_col: str = "order_id",
) -> tuple[bool, dict]:
    """
    Recompute true frequency per canonical customer from ALL raw id-format
    variants (resolved via ``canonical_customer_id``) and compare to the computed
    ``freq_col``. FAIL if the average capture ratio (computed/true) falls below
    ``1 - tolerance`` — catches partial entity resolution / undercounting (e.g.
    the 38%-capture bug). Reports the full distribution of capture ratios.
    """
    detail: dict = {"freq_col": freq_col, "tolerance": tolerance, "threshold": _round(1 - tolerance)}
    if freq_col not in df.columns:
        detail["error"] = "missing freq_col"
        detail["actual"] = None
        return False, detail

    raw = raw_orders_df.copy()
    if canonical_id_col not in raw.columns:
        raw[canonical_id_col] = raw[id_col].map(canonical_customer_id)
    raw = raw.dropna(subset=[canonical_id_col])

    if order_id_col in raw.columns:
        raw["_oid"] = raw[order_id_col].map(normalize_order_id)
        raw = raw.dropna(subset=["_oid"])
        true_freq = raw.groupby(canonical_id_col)["_oid"].nunique()
    else:
        true_freq = raw.groupby(canonical_id_col).size()

    comp = pd.DataFrame({"computed": df[freq_col].astype(float)})
    comp["_canon"] = [canonical_customer_id(i) for i in comp.index]
    comp = comp.dropna(subset=["_canon"])
    comp["true"] = comp["_canon"].map(true_freq).astype(float)
    comp = comp.dropna(subset=["true"])
    comp = comp[comp["true"] > 0]

    if comp.empty:
        detail["error"] = "no customers with a resolvable true frequency"
        detail["actual"] = None
        return False, detail

    ratio = (comp["computed"] / comp["true"]).astype(float)
    avg = float(ratio.mean())
    detail.update(
        {
            "avg_capture_ratio": _round(avg),
            "actual": _round(avg),
            "n_customers_checked": int(len(ratio)),
            "n_undercounted": int((ratio < (1 - tolerance)).sum()),
            "capture_ratio_distribution": {
                "min": _round(ratio.min()),
                "p05": _round(ratio.quantile(0.05)),
                "p25": _round(ratio.quantile(0.25)),
                "p50": _round(ratio.quantile(0.50)),
                "p75": _round(ratio.quantile(0.75)),
                "p95": _round(ratio.quantile(0.95)),
                "max": _round(ratio.max()),
            },
        }
    )
    return bool(avg >= (1 - tolerance)), detail


# ── 7) unattributed revenue ────────────────────────────────────────────────
def check_unattributed_revenue(
    raw_orders_df: pd.DataFrame,
    id_col: str,
    amount_col: str,
    max_unattributed_pct: float = 0.05,
) -> tuple[bool, dict]:
    """
    FAIL if the share of orders with an unresolvable customer id (``canonical_
    customer_id`` returns None) exceeds ``max_unattributed_pct`` by ROW count or
    by REVENUE. Reports the dollar value and row count rather than dropping them
    silently.
    """
    detail: dict = {"threshold": max_unattributed_pct}
    unresolved = raw_orders_df[id_col].map(canonical_customer_id).isna()
    amounts = raw_orders_df[amount_col].map(parse_money)

    total_rows = int(len(raw_orders_df))
    total_rev = float(np.nansum(amounts.to_numpy(dtype=float)))
    unattr_rows = int(unresolved.sum())
    unattr_rev = float(np.nansum(amounts[unresolved].to_numpy(dtype=float)))
    row_pct = unattr_rows / total_rows if total_rows else 0.0
    rev_pct = unattr_rev / total_rev if total_rev else 0.0

    detail.update(
        {
            "unattributed_rows": unattr_rows,
            "total_rows": total_rows,
            "unattributed_row_pct": _round(row_pct),
            "unattributed_revenue": _round(unattr_rev, 2),
            "total_revenue": _round(total_rev, 2),
            "unattributed_revenue_pct": _round(rev_pct),
            "actual": _round(max(row_pct, rev_pct)),
        }
    )
    return bool(row_pct <= max_unattributed_pct and rev_pct <= max_unattributed_pct), detail


# ── 8) near-duplicate orders ───────────────────────────────────────────────
def check_near_duplicate_orders(
    raw_orders_df: pd.DataFrame,
    order_id_col: str,
    amount_col: str,
    date_col: str,
    id_col: str,
    similarity_threshold: float = 0.9,
    max_flagged_pct: float = 0.02,
) -> tuple[bool, dict]:
    """
    Flag near-duplicate order ids using a NORMALIZED (whitespace/format/case-
    insensitive) comparison rather than exact match: two rows whose order ids
    collapse to the same alphanumeric key but whose raw text differs are
    near-dupes. Reports the flagged count for human review and does NOT drop
    anything. FAIL if the flagged fraction exceeds ``max_flagged_pct``.
    """
    detail: dict = {"similarity_threshold": similarity_threshold, "threshold": max_flagged_pct}
    n = int(len(raw_orders_df))

    def _norm(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return _ALNUM_RE.sub("", str(value).strip().lower()) or None

    keys = raw_orders_df[order_id_col].map(_norm)
    frame = pd.DataFrame({"raw": raw_orders_df[order_id_col], "key": keys}).dropna(subset=["key"])

    flagged_idx: set = set()
    examples: list[dict] = []
    for key, grp in frame.groupby("key"):
        raw_variants = grp["raw"].astype(str).unique()
        # near-dup = same normalized key but >1 distinct raw spelling
        if len(grp) > 1 and len(raw_variants) > 1:
            flagged_idx.update(grp.index.tolist())
            if len(examples) < 10:
                examples.append(
                    {"normalized": key, "variants": raw_variants[:6].tolist(), "count": int(len(grp))}
                )

    flagged = len(flagged_idx)
    frac = flagged / n if n else 0.0
    detail.update(
        {
            "flagged_near_duplicates": flagged,
            "total_rows": n,
            "flagged_pct": _round(frac),
            "actual": _round(frac),
            "examples": examples,
            "note": "flagged for human review; NOT auto-dropped",
        }
    )
    return bool(frac <= max_flagged_pct), detail


# ── 9) quantity sanity ─────────────────────────────────────────────────────
def check_qty_sanity(
    raw_orders_df: pd.DataFrame,
    qty_col: str,
    min_val: int = 1,
    max_val=None,
    outlier_pct: float = 99.5,
    max_bad_pct: float = 0.05,
    require_integer: bool = True,
) -> tuple[bool, dict]:
    """
    Classify raw quantities and report counts of: missing, non-positive /
    below-min, non-integer (when integers expected), non-numeric text, and
    statistical outliers (> ``outlier_pct`` percentile). Nothing is imputed.
    FAIL if the hard-invalid fraction exceeds ``max_bad_pct`` (outliers are
    flagged but not, by themselves, a failure).
    """
    detail: dict = {
        "qty_col": qty_col,
        "min_val": min_val,
        "max_val": max_val,
        "outlier_pct": outlier_pct,
        "require_integer": require_integer,
        "threshold": max_bad_pct,
    }
    col = raw_orders_df[qty_col]
    n = int(len(col))

    def classify(value) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "missing"
        s = str(value).strip().lower()
        if s in _QTY_MISSING:
            return "missing"
        if re.fullmatch(r"-?\d+", s):
            return "below_min" if int(s) < min_val else "ok_int"
        if re.fullmatch(r"-?\d+\.\d+", s):
            return "below_min" if float(s) < min_val else "non_integer"
        return "non_numeric"

    kinds = col.map(classify)
    counts = {k: int(v) for k, v in kinds.value_counts().to_dict().items()}

    # Numeric extraction for outlier / max checks (best-effort first number).
    numeric = pd.to_numeric(
        col.astype("string").str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce"
    )
    positive = numeric[numeric.notna() & (numeric >= min_val)]
    hi = float(np.percentile(positive, outlier_pct)) if len(positive) else None
    outliers = int((positive > hi).sum()) if hi is not None else 0
    over_max = int((numeric > max_val).sum()) if max_val is not None else 0

    bad_kinds = ["missing", "below_min", "non_numeric"]
    if require_integer:
        bad_kinds.append("non_integer")
    bad = sum(counts.get(k, 0) for k in bad_kinds) + over_max
    bad_pct = bad / n if n else 0.0

    detail.update(
        {
            "total_rows": n,
            "kind_counts": counts,
            "missing": counts.get("missing", 0),
            "non_positive_or_below_min": counts.get("below_min", 0),
            "non_integer": counts.get("non_integer", 0),
            "non_numeric_text": counts.get("non_numeric", 0),
            "over_max": over_max,
            "outlier_threshold_value": _round(hi),
            "statistical_outliers": outliers,
            "bad_pct": _round(bad_pct),
            "actual": _round(bad_pct),
            "note": "rows flagged, not imputed",
        }
    )
    return bool(bad_pct <= max_bad_pct), detail


# ── aggregator ─────────────────────────────────────────────────────────────
def _resolve(config: dict | None) -> dict:
    """Merge a caller-supplied override dict over the config defaults (shallow)."""
    merged = {k: dict(v) for k, v in VALIDATION.items()}
    if config:
        for k, v in config.items():
            merged.setdefault(k, {}).update(v or {})
    return merged


def _safe(name: str, func, *args, **kwargs) -> tuple[bool, dict]:
    """Run a check; a raised exception is itself a (loud) FAIL, never swallowed."""
    try:
        return func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — surface ANY check error as a failure
        return False, {"error": f"{type(exc).__name__}: {exc}", "actual": None, "threshold": None}


def run_validation(
    *,
    df: pd.DataFrame,
    X=None,
    cluster_labels=None,
    raw_orders_df: pd.DataFrame | None = None,
    cat_cols=None,
    id_col: str = "customer_id",
    raw_category_col: str = "product_category",
    order_id_col: str = "order_id",
    amount_col: str = "order_value",
    date_col: str = "order_date",
    qty_col: str = "quantity",
    canonical_id_col: str = "canonical_id",
    top_category_col: str = "top_category",
    stored_silhouette=None,
    config: dict | None = None,
    report_path=None,
    write_report: bool = True,
    print_summary: bool = True,
    extra_report: dict | None = None,
) -> dict:
    """
    Run every applicable check, aggregate into a report, write
    ``validation_report.json``, print a console summary, and flag overall pass.

    Returns ``{"checks": {name: {pass, detail, threshold, actual}}, "overall_pass":
    bool, "n_failed": int, "failed": [names]}``. Checks whose required inputs are
    absent are skipped (recorded, not counted as failures). Any unexpected error
    inside a check is reported as a FAIL — never silently ignored.
    """
    cfg = _resolve(config)
    checks: dict[str, tuple[bool, dict]] = {}

    # 1) cluster separation
    cs = cfg["cluster_separation"]
    checks["cluster_separation"] = _safe(
        "cluster_separation", check_cluster_separation,
        df, "cluster", cs["rfm_cols"], cs["min_variance_ratio"],
    )

    # 2) silhouette (needs X + labels)
    if X is not None and cluster_labels is not None:
        sc = cfg["silhouette"]
        checks["silhouette_score"] = _safe(
            "silhouette_score", check_silhouette_score,
            X, cluster_labels, sc["min_score"], stored_silhouette, sc["placeholder_tol"],
        )

    # 3) label independence (one entry per configured pair)
    for col_a, col_b in cfg["label_independence"]["pairs"]:
        if col_a in df.columns and col_b in df.columns:
            checks[f"label_independence[{col_a}->{col_b}]"] = _safe(
                "label_independence", check_label_independence, df, col_a, col_b,
            )

    # 4) category consistency
    if cat_cols:
        cc = cfg["category_consistency"]
        checks["category_consistency"] = _safe(
            "category_consistency", check_category_consistency,
            df, cat_cols, cc["freq_col"], cc["tolerance"], cc["max_bad_frac"],
        )

    # raw-orders-dependent checks (5-9)
    if raw_orders_df is not None:
        tc = cfg["top_category_accuracy"]
        checks["top_category_accuracy"] = _safe(
            "top_category_accuracy", check_top_category_accuracy,
            df, top_category_col, raw_orders_df, id_col, raw_category_col, tc["min_match_rate"],
        )

        fc = cfg["frequency_completeness"]
        checks["frequency_completeness"] = _safe(
            "frequency_completeness", check_frequency_completeness,
            df, cfg["category_consistency"]["freq_col"], raw_orders_df, id_col,
            canonical_id_col, fc["tolerance"], order_id_col,
        )

        ur = cfg["unattributed_revenue"]
        checks["unattributed_revenue"] = _safe(
            "unattributed_revenue", check_unattributed_revenue,
            raw_orders_df, id_col, amount_col, ur["max_unattributed_pct"],
        )

        nd = cfg["near_duplicate_orders"]
        checks["near_duplicate_orders"] = _safe(
            "near_duplicate_orders", check_near_duplicate_orders,
            raw_orders_df, order_id_col, amount_col, date_col, id_col,
            nd["similarity_threshold"], nd["max_flagged_pct"],
        )

        if qty_col in raw_orders_df.columns:
            qs = cfg["qty_sanity"]
            checks["qty_sanity"] = _safe(
                "qty_sanity", check_qty_sanity,
                raw_orders_df, qty_col, qs["min_val"], qs["max_val"],
                qs["outlier_pct"], qs["max_bad_pct"], qs["require_integer"],
            )

    # Build the report.
    report_checks: dict[str, dict] = {}
    failed: list[str] = []
    for name, (passed, detail) in checks.items():
        skipped = bool(detail.get("skipped"))
        report_checks[name] = {
            "pass": bool(passed),
            "skipped": skipped,
            "threshold": detail.get("threshold"),
            "actual": detail.get("actual"),
            "detail": detail,
        }
        if not passed and not skipped:
            failed.append(name)

    report = {
        "overall_pass": len(failed) == 0,
        "n_checks": len(report_checks),
        "n_failed": len(failed),
        "failed": failed,
        "checks": report_checks,
    }
    # Non-check diagnostics (e.g. the k-selection table + recommended_k) get
    # recorded alongside the checks so the report is a single source of truth.
    if extra_report:
        report.update(extra_report)

    if write_report:
        path = report_path or VALIDATION_REPORT_PATH
        ensure_dirs()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        report["report_path"] = str(path)

    if print_summary:
        print_validation_summary(report)

    return report


def print_validation_summary(report: dict) -> None:
    """Print a per-check PASS/FAIL/SKIP summary and the overall verdict."""
    print("=" * 62)
    print("SEGMENTATION VALIDATION REPORT")
    print("=" * 62)
    for name, entry in report["checks"].items():
        if entry["skipped"]:
            status = "SKIP"
        elif entry["pass"]:
            status = "PASS"
        else:
            status = "FAIL"
        actual = entry["actual"]
        thresh = entry["threshold"]
        line = f"[{status}] {name}"
        if actual is not None or thresh is not None:
            line += f"  (actual={actual}, threshold={thresh})"
        print(line)
        if status == "FAIL":
            det = entry["detail"]
            if det.get("error"):
                print(f"        error: {det['error']}")
            for key in ("placeholder_warning", "bad_row_count", "flagged_near_duplicates",
                        "n_mismatched", "n_undercounted", "unattributed_rows"):
                if key in det:
                    print(f"        {key}: {det[key]}")
    print("-" * 62)
    verdict = "PASS — safe to proceed" if report["overall_pass"] else (
        f"FAIL — {report['n_failed']} check(s) failed: {', '.join(report['failed'])}"
    )
    print(f"OVERALL: {verdict}")
    if not report["overall_pass"]:
        print("Pipeline HALTED — human sign-off required before persona-labeling /")
        print("dashboard / API steps. Outputs were NOT promoted.")
    print("=" * 62)
