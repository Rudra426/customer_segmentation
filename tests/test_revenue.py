"""
test_revenue.py — unit tests for the Revenue Impact compute functions.

Verifies revenue-concentration percentages and CLV-at-risk math against a small
synthetic fixture with known answers, plus edge cases (zero/missing values, no
at-risk segment match).

Run:  python -m pytest tests/test_revenue.py     (or)  python tests/test_revenue.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.revenue import compute_clv_at_risk, compute_revenue_concentration


def _fixture() -> pd.DataFrame:
    """4 customers; one has zero monetary/clv (must be excluded from math)."""
    return pd.DataFrame(
        {
            "customer_id": ["A", "B", "C", "D"],
            "persona": [
                "Loyal Big Spenders",
                "At-Risk High Value",
                "One-Time Buyers",
                "New Customers",
            ],
            "monetary": [700.0, 200.0, 100.0, 0.0],  # D excluded (zero)
            "clv": [210.0, 60.0, 30.0, 0.0],          # D excluded (zero)
        }
    )


def test_revenue_concentration_math():
    df = _fixture()
    out = compute_revenue_concentration(df, "persona", "monetary")

    # Zero-monetary row excluded; totals over the 3 valid customers.
    assert out.attrs["excluded_count"] == 1
    assert out.attrs["total_revenue"] == 1000.0
    assert out.attrs["total_customers"] == 3

    # Sorted by descending pct_of_revenue -> Loyal first.
    top = out.iloc[0]
    assert top["segment"] == "Loyal Big Spenders"
    assert top["total_revenue"] == 700.0
    assert top["pct_of_revenue"] == 70.0
    assert round(top["pct_of_customers"], 2) == 33.33
    assert top["avg_revenue_per_customer"] == 700.0

    # Order of the rest.
    assert list(out["segment"]) == [
        "Loyal Big Spenders", "At-Risk High Value", "One-Time Buyers"
    ]
    assert list(out["pct_of_revenue"]) == [70.0, 20.0, 10.0]


def test_clv_at_risk_math():
    df = _fixture()
    res = compute_clv_at_risk(df, "persona", "clv", ["at risk", "dormant", "churn"])

    assert res["any_at_risk"] is True
    assert res["matched_segments"] == ["At-Risk High Value"]
    # total valid CLV = 210 + 60 + 30 = 300; at-risk = 60.
    assert res["total_clv"] == 300.0
    assert res["total_clv_at_risk"] == 60.0
    assert res["pct_of_total_clv_at_risk"] == 20.0
    assert res["customer_count_at_risk"] == 1
    assert res["pct_of_total_customers_at_risk"] == 25.0  # 1 of 4 total customers
    assert res["excluded_count"] == 1
    assert res["breakdown"]["At-Risk High Value"]["total_clv"] == 60.0


def test_no_at_risk_segment_match():
    df = _fixture()
    res = compute_clv_at_risk(df, "persona", "clv", ["nonexistent-pattern"])
    assert res["any_at_risk"] is False
    assert res["matched_segments"] == []
    assert res["total_clv_at_risk"] == 0.0
    assert res["pct_of_total_clv_at_risk"] == 0.0


def test_all_zero_monetary_does_not_crash():
    df = _fixture().assign(monetary=0.0)
    out = compute_revenue_concentration(df, "persona", "monetary")
    assert out.empty
    assert out.attrs["excluded_count"] == 4
    assert out.attrs["total_revenue"] == 0.0


if __name__ == "__main__":
    test_revenue_concentration_math()
    test_clv_at_risk_math()
    test_no_at_risk_segment_match()
    test_all_zero_monetary_does_not_crash()
    print("All revenue tests passed.")
