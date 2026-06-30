"""
revenue.py — Revenue Impact analytics over the labeled customer table.

Pure, importable functions (no Streamlit) so they are unit-testable:
  - compute_revenue_concentration : per-segment revenue share (Pareto view)
  - compute_clv_at_risk           : CLV sitting in at-risk segments

Both tolerate missing/zero monetary or clv values by excluding them from the
percentage math and reporting how many rows were excluded.
"""

from __future__ import annotations

import pandas as pd


def _normalize(label) -> str:
    """Lowercase and treat '-'/'_' as spaces, for robust label matching."""
    return str(label).lower().replace("-", " ").replace("_", " ").strip()


def format_currency(value: float) -> str:
    """Format a number as USD with thousands separators, e.g. $1,234.56."""
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def format_percent(value: float) -> str:
    """Format a number as a percentage with one decimal, e.g. 12.3%."""
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def compute_revenue_concentration(
    df: pd.DataFrame, segment_col: str, monetary_col: str
) -> pd.DataFrame:
    """
    Per-segment revenue concentration, sorted by descending % of revenue.

    Columns: segment, customer_count, pct_of_customers, total_revenue,
    pct_of_revenue, avg_revenue_per_customer.

    Rows with missing or non-positive monetary are excluded from all math; the
    excluded count and overall totals are attached via DataFrame.attrs
    ("excluded_count", "total_revenue", "total_customers").
    """
    work = df[[segment_col, monetary_col]].copy()
    work[monetary_col] = pd.to_numeric(work[monetary_col], errors="coerce")

    valid_mask = work[monetary_col].notna() & (work[monetary_col] > 0)
    valid = work[valid_mask]
    excluded = int((~valid_mask).sum())

    total_customers = int(len(valid))
    total_revenue = float(valid[monetary_col].sum())

    cols = [
        "segment", "customer_count", "pct_of_customers",
        "total_revenue", "pct_of_revenue", "avg_revenue_per_customer",
    ]
    if total_customers == 0:
        out = pd.DataFrame(columns=cols)
    else:
        grp = valid.groupby(segment_col)[monetary_col].agg(["count", "sum"])
        grp = grp.rename(columns={"count": "customer_count", "sum": "total_revenue"})
        grp["pct_of_customers"] = grp["customer_count"] / total_customers * 100
        grp["pct_of_revenue"] = (
            grp["total_revenue"] / total_revenue * 100 if total_revenue else 0.0
        )
        grp["avg_revenue_per_customer"] = grp["total_revenue"] / grp["customer_count"]
        out = grp.reset_index().rename(columns={segment_col: "segment"})
        out = out.sort_values("pct_of_revenue", ascending=False).reset_index(drop=True)
        out = out[cols]
        for c in ("total_revenue", "pct_of_customers", "pct_of_revenue",
                  "avg_revenue_per_customer"):
            out[c] = out[c].round(2)

    out.attrs["excluded_count"] = excluded
    out.attrs["total_revenue"] = round(total_revenue, 2)
    out.attrs["total_customers"] = total_customers
    return out


def compute_clv_at_risk(
    df: pd.DataFrame,
    segment_col: str,
    clv_col: str,
    at_risk_segment_names: list[str],
) -> dict:
    """
    Total CLV sitting in at-risk segments.

    `at_risk_segment_names` are substring patterns matched (case-insensitive,
    '-'/'_' treated as spaces) against the actual segment labels, so it works
    regardless of exact persona naming. Rows with missing/non-positive clv are
    excluded from CLV sums (counted in "excluded_count").

    Returns a dict with total_clv_at_risk, customer_count_at_risk,
    pct_of_total_clv_at_risk, pct_of_total_customers_at_risk, a per-segment
    breakdown, the matched_segments list, and any_at_risk (False when no segment
    matches the patterns).
    """
    combined = pd.DataFrame({
        "segment": df[segment_col],
        "_clv": pd.to_numeric(df[clv_col], errors="coerce"),
    })

    total_customers = int(len(combined))
    valid_mask = combined["_clv"].notna() & (combined["_clv"] > 0)
    excluded = int((~valid_mask).sum())
    total_clv = float(combined.loc[valid_mask, "_clv"].sum())

    patterns = [_normalize(p) for p in at_risk_segment_names]
    labels = [lbl for lbl in combined["segment"].dropna().unique()]
    matched = [
        lbl for lbl in labels
        if any(p in _normalize(lbl) for p in patterns)
    ]

    breakdown: dict[str, dict] = {}
    total_at_risk_clv = 0.0
    count_at_risk = 0
    for lbl in matched:
        seg_mask = combined["segment"] == lbl
        seg_count = int(seg_mask.sum())
        seg_clv = float(combined.loc[seg_mask & valid_mask, "_clv"].sum())
        breakdown[str(lbl)] = {
            "customer_count": seg_count,
            "total_clv": round(seg_clv, 2),
            "pct_of_total_clv": round(seg_clv / total_clv * 100, 2) if total_clv else 0.0,
        }
        total_at_risk_clv += seg_clv
        count_at_risk += seg_count

    return {
        "any_at_risk": bool(matched),
        "matched_segments": [str(m) for m in matched],
        "total_clv_at_risk": round(total_at_risk_clv, 2),
        "customer_count_at_risk": count_at_risk,
        "pct_of_total_clv_at_risk": (
            round(total_at_risk_clv / total_clv * 100, 2) if total_clv else 0.0
        ),
        "pct_of_total_customers_at_risk": (
            round(count_at_risk / total_customers * 100, 2) if total_customers else 0.0
        ),
        "breakdown": breakdown,
        "total_clv": round(total_clv, 2),
        "excluded_count": excluded,
    }
