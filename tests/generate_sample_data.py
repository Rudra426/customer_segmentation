"""
generate_sample_data.py — create a small, deliberately messy e-commerce export
used as the end-to-end test fixture throughout the pipeline.

Run:  python tests/generate_sample_data.py
Writes: data/raw/sample_messy.csv  (50 order rows, 6 messy columns)

The messiness is intentional so later phases (mapping, cleaning, validation)
have something realistic to handle:
  - Cryptic column names    -> "Cust ID", "Order #", "Total $", "SKU Category"
  - Mixed date formats      -> "2024-03-01", "03/15/2024", "Apr 2 2024"
  - Currency symbols/commas -> "$1,250.00", "99.5"
  - A few null / blank cells in optional columns
  - A duplicate order number and a negative refund row
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "raw" / "sample_messy.csv"

CATEGORIES = ["Apparel", "Footwear", "Accessories", "Home", "Beauty", ""]
DATE_STYLES = [
    lambda y, m, d: f"{y:04d}-{m:02d}-{d:02d}",          # 2024-03-01
    lambda y, m, d: f"{m:02d}/{d:02d}/{y:04d}",          # 03/15/2024
    lambda y, m, d: pd.Timestamp(y, m, d).strftime("%b %d %Y"),  # Mar 01 2024
]


def money(value: float) -> str:
    """Format a number as a messy money string (sometimes with $ and commas)."""
    style = random.random()
    if style < 0.5:
        return f"${value:,.2f}"        # "$1,250.00"
    if style < 0.8:
        return f"{value:.2f}"          # "1250.00"
    return f"{value:.1f}"              # "1250.0"


def build_rows() -> list[dict]:
    random.seed(SEED)
    rows: list[dict] = []
    n_customers = 15
    order_counter = 1000

    for cust in range(1, n_customers + 1):
        cust_id = f"C{cust:03d}"
        n_orders = random.randint(1, 6)          # varied frequency
        for _ in range(n_orders):
            order_counter += 1
            month = random.randint(1, 6)
            day = random.randint(1, 28)
            date_fn = random.choice(DATE_STYLES)
            value = round(random.uniform(15, 500), 2)
            qty = random.randint(1, 5)
            cat = random.choice(CATEGORIES)
            rows.append(
                {
                    "Cust ID": cust_id,
                    "Order #": f"ORD{order_counter}",
                    "Order Date": date_fn(2024, month, day),
                    "Total $": money(value),
                    "SKU Category": cat,
                    "Qty": qty,
                }
            )

    # --- inject specific messy edge cases for later phases ---
    # 1) duplicate order number (same as the first order)
    rows.append({**rows[0]})
    # 2) a negative refund row
    rows.append(
        {
            "Cust ID": "C002",
            "Order #": "ORD9999",
            "Order Date": "04/10/2024",
            "Total $": "-$45.00",
            "SKU Category": "Apparel",
            "Qty": 1,
        }
    )
    # 3) a row with a blank Total $ (null required field)
    rows.append(
        {
            "Cust ID": "C003",
            "Order #": "ORD9998",
            "Order Date": "2024-05-05",
            "Total $": "",
            "SKU Category": "Home",
            "Qty": 2,
        }
    )

    # trim/pad to exactly 50 rows for a predictable fixture
    return rows[:50]


def main() -> None:
    rows = build_rows()
    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows x {df.shape[1]} cols -> {OUT_PATH}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
