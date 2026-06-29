"""
features.py — Phase 4: aggregate clean order rows into one numeric row per
customer (RFM + monetary + tenure + category affinity).

Input is the cleaned, typed DataFrame from cleaner.clean_data(). Output (built
up across steps 4.1–4.7) is a customer-level feature matrix indexed by
customer_id, suitable for scaling + clustering in Phase 5.
"""

from __future__ import annotations

import pandas as pd

from config import CLV_MARGIN


def _snapshot_date(clean_df: pd.DataFrame, snapshot=None) -> pd.Timestamp:
    """Resolve the RFM snapshot date (default = latest order date in the data)."""
    if snapshot is not None:
        return pd.Timestamp(snapshot)
    return clean_df["order_date"].max()


def compute_recency(clean_df: pd.DataFrame, snapshot=None) -> pd.Series:
    """
    Recency = days since each customer's most recent order, relative to the
    snapshot date. Lower means more recently active.

    Returns a Series named "recency" indexed by customer_id.
    """
    snap = _snapshot_date(clean_df, snapshot)
    last_order = clean_df.groupby("customer_id")["order_date"].max()
    recency = (snap - last_order).dt.days
    recency.name = "recency"
    return recency.astype("int64")


def compute_frequency(clean_df: pd.DataFrame) -> pd.Series:
    """
    Frequency = number of distinct orders per customer.

    Counts unique order_id (not rows) so any residual row repeats can't inflate
    the value. Returns a Series named "frequency" indexed by customer_id.
    """
    freq = clean_df.groupby("customer_id")["order_id"].nunique()
    freq.name = "frequency"
    return freq.astype("int64")


def compute_monetary(clean_df: pd.DataFrame) -> pd.Series:
    """
    Monetary = total spend per customer (sum of order_value).

    Returns a Series named "monetary" indexed by customer_id.
    """
    monetary = clean_df.groupby("customer_id")["order_value"].sum()
    monetary.name = "monetary"
    return monetary.astype("float64").round(2)


def compute_aov_clv(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average Order Value and a simple CLV proxy per customer.

      aov = monetary / frequency          (avg spend per order)
      clv = aov * frequency * CLV_MARGIN  (= monetary * margin: expected profit
            contribution to date; a deliberately simple v1 proxy, not a churn
            model)

    Returns a DataFrame indexed by customer_id with columns ["aov", "clv"].
    """
    monetary = compute_monetary(clean_df)
    frequency = compute_frequency(clean_df)

    aov = (monetary / frequency).round(2)
    aov.name = "aov"
    clv = (monetary * CLV_MARGIN).round(2)
    clv.name = "clv"

    return pd.concat([aov, clv], axis=1)


def compute_tenure(clean_df: pd.DataFrame, snapshot=None) -> pd.Series:
    """
    Tenure = days a customer has been active, relative to the snapshot date.

    Start point is the customer's signup_date if that optional field is present
    and populated, otherwise their first order date. Returns a Series named
    "tenure" indexed by customer_id.
    """
    snap = _snapshot_date(clean_df, snapshot)
    first_order = clean_df.groupby("customer_id")["order_date"].min()

    start = first_order
    if "signup_date" in clean_df.columns:
        signup = clean_df.groupby("customer_id")["signup_date"].min()
        # Use signup where available, else fall back to first order.
        start = signup.fillna(first_order)

    tenure = (snap - start).dt.days.clip(lower=0)
    tenure.name = "tenure"
    return tenure.astype("int64")


def compute_top_category(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Each customer's dominant product category + a one-hot numeric encoding.

    Returns a DataFrame indexed by customer_id with:
      - top_category : string label of the most-frequent category (display only),
                       <NA> if the customer has no category data
      - cat_<name>   : one-hot 0/1 columns (numeric, for clustering)

    Returns an EMPTY DataFrame (no rows/cols) if product_category is absent or
    entirely null — the assembler then simply omits category features.
    """
    if "product_category" not in clean_df.columns:
        return pd.DataFrame()

    cats = clean_df[["customer_id", "product_category"]].dropna(
        subset=["product_category"]
    )
    if cats.empty:
        return pd.DataFrame()

    # Most-frequent category per customer (ties broken alphabetically for stability).
    counts = (
        cats.groupby(["customer_id", "product_category"]).size().reset_index(name="n")
    )
    counts = counts.sort_values(["customer_id", "n", "product_category"],
                                ascending=[True, False, True])
    top = counts.groupby("customer_id").first()["product_category"]
    top.name = "top_category"

    # One-hot encode the TOP category (numeric features for clustering).
    onehot = pd.get_dummies(top, prefix="cat").astype("int64")

    result = pd.concat([top.astype("string"), onehot], axis=1)
    # Reindex to all customers so customers with no category get NA / all-zero.
    all_customers = clean_df["customer_id"].dropna().unique()
    result = result.reindex(all_customers)
    onehot_cols = [c for c in result.columns if c.startswith("cat_")]
    result[onehot_cols] = result[onehot_cols].fillna(0).astype("int64")
    result.index.name = "customer_id"
    return result


# Non-numeric / identifier columns excluded from the clustering matrix.
_NON_NUMERIC_COLS = ("top_category",)


def engineer_features(clean_df: pd.DataFrame, snapshot=None) -> pd.DataFrame:
    """
    Build the full per-customer feature table (one row per customer).

    Combines: recency, frequency, monetary, aov, clv, tenure, and (if available)
    the top_category label + cat_* one-hot columns. Indexed by customer_id.
    Every column is numeric except the optional "top_category" display label.
    """
    parts = [
        compute_recency(clean_df, snapshot),
        compute_frequency(clean_df),
        compute_monetary(clean_df),
        compute_aov_clv(clean_df),
        compute_tenure(clean_df, snapshot),
    ]
    category = compute_top_category(clean_df)
    if not category.empty:
        parts.append(category)

    features = pd.concat(parts, axis=1)
    features.index.name = "customer_id"

    # Safety: numeric features should be fully populated; fill any gaps with 0.
    numeric_cols = [c for c in features.columns if c not in _NON_NUMERIC_COLS]
    features[numeric_cols] = features[numeric_cols].fillna(0)
    return features


def feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Return the numeric-only view (drops display labels) for scaling/clustering."""
    cols = [c for c in features.columns if c not in _NON_NUMERIC_COLS]
    return features[cols].astype("float64")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from config import RAW_DIR
    from src.cleaner import clean_data
    from src.schema_mapper import apply_mapping, load_raw_file

    mapping = {
        "Cust ID": "customer_id", "Order #": "order_id", "Order Date": "order_date",
        "Total $": "order_value", "SKU Category": "product_category", "Qty": "quantity",
    }
    clean_df, _ = clean_data(apply_mapping(load_raw_file(RAW_DIR / "sample_messy.csv"), mapping))
    feats = engineer_features(clean_df)
    print("feature table shape:", feats.shape)
    print(feats.round(2).to_string())
    print("\nnumeric matrix cols:", list(feature_matrix(feats).columns))
