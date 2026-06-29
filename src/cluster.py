"""
cluster.py — Phase 5: scale features, choose k, fit K-Means, validate, and
persist artifacts.

Input is the numeric feature matrix from features.feature_matrix(). Output is a
fitted scaler + K-Means model (saved with joblib), per-customer cluster labels,
validation metrics (silhouette, Davies-Bouldin), and a UMAP 2D projection.

Built step by step: 5.1 scaling (this piece) -> ... -> 5.5 persistence.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from config import (
    K_MAX,
    K_MIN,
    KMEANS_PATH,
    METADATA_PATH,
    RANDOM_STATE,
    SCALER_PATH,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    ensure_dirs,
)

# Cap the per-feature baseline sample stored for drift detection (Phase 10).
_BASELINE_SAMPLE_CAP = 2000


def scale_features(matrix: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize features to zero mean / unit variance.

    K-Means uses Euclidean distance, so unscaled features (e.g. monetary in the
    thousands vs. one-hot 0/1) would dominate. Returns the scaled DataFrame
    (same index/columns) and the fitted StandardScaler for reuse at scoring time.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix.to_numpy())
    scaled_df = pd.DataFrame(scaled, index=matrix.index, columns=matrix.columns)
    return scaled_df, scaler


def fit_kmeans(scaled_df: pd.DataFrame) -> tuple[KMeans, pd.Series, dict]:
    """
    Fit K-Means for k in [K_MIN, K_MAX] and pick k with the best silhouette.

    Silhouette needs 2 <= k <= n_samples-1, so the k range is clamped to the
    data size (important for tiny datasets). Returns:
      - the fitted KMeans model for the chosen k
      - a Series of cluster labels indexed by customer_id
      - info dict: {"best_k", "silhouette_by_k": {k: score}, "best_silhouette"}
    """
    X = scaled_df.to_numpy()
    n = X.shape[0]
    k_hi = min(K_MAX, n - 1)
    if k_hi < K_MIN:
        raise ValueError(
            f"Too few customers ({n}) to cluster with k>={K_MIN}. Need more data."
        )

    silhouette_by_k: dict[int, float] = {}
    # Start below silhouette's -1 floor so the first k always wins, guaranteeing a
    # non-None model even on degenerate data where every k scores -1.0.
    best_k, best_score, best_model = None, -2.0, None
    for k in range(K_MIN, k_hi + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X)
        score = float(silhouette_score(X, labels)) if len(set(labels)) > 1 else -1.0
        silhouette_by_k[k] = round(score, 4)
        if score > best_score:
            best_k, best_score, best_model = k, score, model

    labels = pd.Series(best_model.labels_, index=scaled_df.index, name="cluster")
    info = {
        "best_k": best_k,
        "silhouette_by_k": silhouette_by_k,
        "best_silhouette": round(best_score, 4),
    }
    return best_model, labels, info


def cluster_metrics(scaled_df: pd.DataFrame, labels: pd.Series) -> dict:
    """
    Compute validation metrics for a clustering.

    Returns {"k", "silhouette", "davies_bouldin", "cluster_sizes"}.
      - silhouette    : higher is better ( -1..1 )
      - davies_bouldin: lower is better ( >=0 )
    Both require >=2 distinct clusters; otherwise they are reported as None.
    """
    X = scaled_df.to_numpy()
    y = labels.to_numpy()
    n_clusters = len(set(y.tolist()))

    if n_clusters > 1:
        sil = round(float(silhouette_score(X, y)), 4)
        dbi = round(float(davies_bouldin_score(X, y)), 4)
    else:
        sil = dbi = None

    sizes = labels.value_counts().sort_index().to_dict()
    return {
        "k": n_clusters,
        "silhouette": sil,
        "davies_bouldin": dbi,
        "cluster_sizes": {int(k): int(v) for k, v in sizes.items()},
    }


def compute_umap(scaled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Project scaled features to 2D with UMAP for the dashboard scatter plot.

    n_neighbors is clamped to n_samples-1 (UMAP errors if it exceeds the data
    size). Seeded with RANDOM_STATE for reproducible coordinates. Returns a
    DataFrame indexed by customer_id with columns ["umap_x", "umap_y"].
    """
    import umap  # imported lazily — UMAP import is slow (numba JIT)

    n = scaled_df.shape[0]
    n_neighbors = max(2, min(UMAP_N_NEIGHBORS, n - 1))

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=UMAP_MIN_DIST,
        random_state=RANDOM_STATE,
    )
    coords = reducer.fit_transform(scaled_df.to_numpy())
    return pd.DataFrame(
        coords, index=scaled_df.index, columns=["umap_x", "umap_y"]
    )


def _baseline(matrix: pd.DataFrame) -> dict:
    """Per-feature stats + a capped value sample, stored for drift detection."""
    stats = {}
    sample = {}
    for col in matrix.columns:
        s = matrix[col].astype(float)
        stats[col] = {
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std(ddof=0)), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "median": round(float(s.median()), 4),
        }
        sample[col] = s.head(_BASELINE_SAMPLE_CAP).round(4).tolist()
    return {"stats": stats, "sample": sample}


def save_artifacts(scaler: StandardScaler, model: KMeans, metadata: dict) -> None:
    """Persist scaler + model (joblib) and metadata (JSON) into models/."""
    ensure_dirs()
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(model, KMEANS_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def run_clustering(features: pd.DataFrame, save: bool = True) -> dict:
    """
    Full Phase 5 pipeline: scale -> select k -> metrics -> UMAP -> persist.

    `features` is the engineer_features() output (numeric columns + the optional
    top_category label). Returns a dict:
      result   : features + cluster label + umap_x/umap_y (per customer)
      scaler   : fitted StandardScaler
      model    : fitted KMeans
      metrics  : cluster_metrics() dict
      metadata : everything persisted (feature order, metrics, drift baseline)
    """
    from src.features import feature_matrix

    matrix = feature_matrix(features)
    scaled, scaler = scale_features(matrix)
    model, labels, selection = fit_kmeans(scaled)
    metrics = cluster_metrics(scaled, labels)
    coords = compute_umap(scaled)

    result = features.copy()
    result["cluster"] = labels
    result = result.join(coords)

    metadata = {
        "feature_columns": list(matrix.columns),
        "best_k": selection["best_k"],
        "silhouette_by_k": selection["silhouette_by_k"],
        "metrics": metrics,
        "n_customers": int(matrix.shape[0]),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "drift_baseline": _baseline(matrix),
    }

    if save:
        save_artifacts(scaler, model, metadata)

    return {
        "result": result,
        "scaler": scaler,
        "model": model,
        "metrics": metrics,
        "metadata": metadata,
    }


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from config import RAW_DIR
    from src.cleaner import clean_data
    from src.features import engineer_features, feature_matrix
    from src.schema_mapper import apply_mapping, load_raw_file

    mapping = {
        "Cust ID": "customer_id", "Order #": "order_id", "Order Date": "order_date",
        "Total $": "order_value", "SKU Category": "product_category", "Qty": "quantity",
    }
    clean_df, _ = clean_data(apply_mapping(load_raw_file(RAW_DIR / "sample_messy.csv"), mapping))
    matrix = feature_matrix(engineer_features(clean_df))
    scaled, scaler = scale_features(matrix)
    print("scaled shape:", scaled.shape)
    print("means ~0:", scaled.mean().round(3).abs().max(), "| stds ~1:", scaled.std(ddof=0).round(3).min())
