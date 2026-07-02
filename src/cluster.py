"""
cluster.py — Phase 5: scale features, choose k, fit K-Means, validate, and
persist artifacts.

Input is the numeric feature matrix from features.feature_matrix(). Output is a
fitted scaler + K-Means model (saved with joblib), per-customer cluster labels,
validation metrics (silhouette, Davies-Bouldin), and a UMAP 2D projection.

k-selection (5.2) is multi-metric and stability-checked: select_optimal_k()
searches K_SEARCH_RANGE, refits each k with STABILITY_SEEDS different seeds, and
combines average silhouette + Davies-Bouldin with label stability (pairwise
Adjusted Rand Index), disqualifying any k that fragments into a cluster smaller
than MIN_CLUSTER_PCT. The recommended k is surfaced for human confirmation; the
final model is then re-fit cleanly on the FULL dataset with the confirmed k.

Built step by step: 5.1 scaling -> 5.2 k-selection -> ... -> 5.5 persistence.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from config import (
    DIAGNOSTIC_PLOT_PATH,
    K_SEARCH_RANGE,
    KMEANS_PATH,
    METADATA_PATH,
    MIN_CLUSTER_PCT,
    MIN_STABILITY_ARI,
    RANDOM_STATE,
    SCALER_PATH,
    STABILITY_SEEDS,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    ensure_dirs,
)

# Columns that are heavily right-skewed (a handful of high-spend customers can
# be 100-1000x the median). Left un-transformed, a single extreme outlier
# dominates Euclidean distance after scaling and K-Means degenerates into a
# "1 outlier vs. everyone else" split instead of a real behavioral segmentation.
_LOG_TRANSFORM_COLS = ("monetary", "aov", "clv", "frequency")


def scale_features(matrix: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Log-transform skewed monetary columns, then standardize to zero mean /
    unit variance.

    K-Means uses Euclidean distance, so unscaled features (e.g. monetary in the
    thousands vs. one-hot 0/1) would dominate. Monetary-ish columns are also
    heavily right-skewed (a few whale customers vs. a median of single-digit
    dollars) — scaling those raw lets one extreme customer dominate the whole
    distance space and collapse K-Means into a degenerate outlier-vs-rest split.
    log1p compresses that tail before scaling so clusters separate on overall
    behavior instead of being hijacked by one outlier.

    Returns the scaled DataFrame (same index/columns) and the fitted
    StandardScaler for reuse at scoring time. NOTE: scoring code must apply the
    same log1p to these columns before calling scaler.transform().
    """
    transformed = matrix.copy()
    log_cols = [c for c in _LOG_TRANSFORM_COLS if c in transformed.columns]
    transformed[log_cols] = np.log1p(transformed[log_cols].clip(lower=0))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(transformed.to_numpy())
    scaled_df = pd.DataFrame(scaled, index=matrix.index, columns=matrix.columns)
    return scaled_df, scaler


def _medoid_labeling(seed_labels: list[np.ndarray]) -> np.ndarray:
    """
    Pick the "modal" (most representative) labeling among the per-seed runs.

    Cluster ids are arbitrary across seeds, so we cannot vote per-point. Instead
    we take the medoid labeling: the seed whose labels are, on average, most
    similar (highest mean pairwise Adjusted Rand Index) to all the other seeds'
    labels. That is the most typical/common partition at this k, used to report
    the smallest cluster's share.
    """
    if len(seed_labels) == 1:
        return seed_labels[0]
    best_idx, best_mean = 0, -2.0
    for i, li in enumerate(seed_labels):
        others = [adjusted_rand_score(li, lj) for j, lj in enumerate(seed_labels) if j != i]
        mean_ari = float(np.mean(others)) if others else 0.0
        if mean_ari > best_mean:
            best_idx, best_mean = i, mean_ari
    return seed_labels[best_idx]


def select_optimal_k(
    X_scaled,
    k_range: tuple[int, int] = K_SEARCH_RANGE,
    n_seeds: int = STABILITY_SEEDS,
    min_cluster_pct: float = MIN_CLUSTER_PCT,
    min_stability_ari: float = MIN_STABILITY_ARI,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Multi-metric, stability-checked k-selection.

    For each k in ``k_range`` (inclusive, clamped to 2..n-1 so silhouette is
    defined) the model is fit ``n_seeds`` times with different random_state
    values (n_init=10 each) and we compute:
      - avg_silhouette     : mean silhouette across seeds (higher better)
      - avg_davies_bouldin : mean Davies-Bouldin across seeds (lower better)
      - avg_stability_ari  : mean pairwise Adjusted Rand Index between the seed
                             labelings (1.0 = identical partitions every seed)
      - min_cluster_pct    : smallest cluster's share under the modal labeling
      - disqualified       : True if min_cluster_pct < ``min_cluster_pct`` (the
                             threshold) — guards against fragmenting into tiny,
                             near-duplicate clusters.

    Returns a dict:
      table              : DataFrame [k, avg_silhouette, avg_davies_bouldin,
                           avg_stability_ari, min_cluster_pct, disqualified]
      records            : JSON-safe list-of-dicts version of ``table``
      recommended_k      : highest avg_silhouette among non-disqualified k whose
                           avg_stability_ari > ``min_stability_ari`` (with graceful
                           fallbacks if none qualify)
      recommendation_basis, k_range, n_seeds, min_cluster_pct, min_stability_ari
    """
    X = np.asarray(X_scaled, dtype=float)
    n = X.shape[0]
    k_lo = max(2, int(k_range[0]))
    k_hi = min(int(k_range[1]), n - 1)
    if k_hi < k_lo:
        raise ValueError(
            f"Too few customers ({n}) to search k in {tuple(k_range)} "
            f"(need at least {k_lo + 1} rows)."
        )

    rows: list[dict] = []
    for k in range(k_lo, k_hi + 1):
        seed_labels: list[np.ndarray] = []
        sils: list[float] = []
        dbis: list[float] = []
        for i in range(n_seeds):
            model = KMeans(n_clusters=k, random_state=random_state + i, n_init=10)
            labels = model.fit_predict(X)
            seed_labels.append(labels)
            if len(set(labels.tolist())) > 1:
                sils.append(float(silhouette_score(X, labels)))
                dbis.append(float(davies_bouldin_score(X, labels)))

        # Stability: average pairwise ARI over every distinct seed pair.
        pair_aris = [
            adjusted_rand_score(a, b) for a, b in combinations(seed_labels, 2)
        ]
        avg_ari = float(np.mean(pair_aris)) if pair_aris else 1.0

        # Smallest cluster share under the modal (medoid) labeling.
        modal = _medoid_labeling(seed_labels)
        sizes = np.bincount(modal, minlength=k)
        min_pct = float(sizes[sizes > 0].min() / n) if n else 0.0

        rows.append(
            {
                "k": int(k),
                "avg_silhouette": round(float(np.mean(sils)), 4) if sils else None,
                "avg_davies_bouldin": round(float(np.mean(dbis)), 4) if dbis else None,
                "avg_stability_ari": round(avg_ari, 4),
                "min_cluster_pct": round(min_pct, 4),
                "disqualified": bool(min_pct < min_cluster_pct),
            }
        )

    table = pd.DataFrame(
        rows,
        columns=[
            "k", "avg_silhouette", "avg_davies_bouldin",
            "avg_stability_ari", "min_cluster_pct", "disqualified",
        ],
    )

    recommended_k, basis = _recommend_k(table, min_stability_ari)
    return {
        "table": table,
        "records": rows,
        "recommended_k": recommended_k,
        "recommendation_basis": basis,
        "k_range": [k_lo, k_hi],
        "n_seeds": int(n_seeds),
        "min_cluster_pct": float(min_cluster_pct),
        "min_stability_ari": float(min_stability_ari),
    }


def _recommend_k(table: pd.DataFrame, min_stability_ari: float) -> tuple[int, str]:
    """
    Choose the recommended k: highest avg_silhouette among non-disqualified k
    whose avg_stability_ari exceeds ``min_stability_ari``. Falls back (with a
    recorded reason) so a usable k is always returned even on awkward data.
    """
    has_sil = table["avg_silhouette"].notna()
    stable = table["avg_stability_ari"] > min_stability_ari
    keep = ~table["disqualified"]

    primary = table[keep & stable & has_sil]
    if not primary.empty:
        k = int(primary.loc[primary["avg_silhouette"].idxmax(), "k"])
        return k, f"highest avg_silhouette among non-disqualified k with avg_stability_ari > {min_stability_ari}"

    relaxed = table[keep & has_sil]
    if not relaxed.empty:
        k = int(relaxed.loc[relaxed["avg_silhouette"].idxmax(), "k"])
        return k, f"fallback: no k exceeded ARI {min_stability_ari}; highest avg_silhouette among non-disqualified k"

    any_sil = table[has_sil]
    if not any_sil.empty:
        k = int(any_sil.loc[any_sil["avg_silhouette"].idxmax(), "k"])
        return k, "fallback: every k disqualified; highest avg_silhouette overall"

    return int(table.iloc[0]["k"]), "fallback: no valid silhouette computed; smallest k in range"


def save_diagnostic_plot(
    table: pd.DataFrame,
    recommended_k: int,
    path=DIAGNOSTIC_PLOT_PATH,
    min_cluster_pct: float = MIN_CLUSTER_PCT,
) -> Path:
    """
    Render the k-selection diagnostic as a saved PNG artifact (not just numbers):
    three stacked subplots sharing the k axis — avg silhouette, avg Davies-Bouldin,
    and smallest-cluster % — with the recommended k marked by a vertical line and
    disqualified k shaded red. Uses the headless Agg backend (no display needed).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dirs()
    ks = table["k"].to_numpy()
    sil = pd.to_numeric(table["avg_silhouette"], errors="coerce").to_numpy()
    dbi = pd.to_numeric(table["avg_davies_bouldin"], errors="coerce").to_numpy()
    pct = pd.to_numeric(table["min_cluster_pct"], errors="coerce").to_numpy() * 100.0

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
    axes[0].plot(ks, sil, marker="o", color="C0")
    axes[0].set_ylabel("Avg silhouette\n(higher = better)")
    axes[1].plot(ks, dbi, marker="o", color="C1")
    axes[1].set_ylabel("Avg Davies-Bouldin\n(lower = better)")
    axes[2].plot(ks, pct, marker="o", color="C2")
    axes[2].axhline(min_cluster_pct * 100.0, color="red", linestyle=":", alpha=0.7,
                    label=f"min {min_cluster_pct * 100:.0f}%")
    axes[2].set_ylabel("Smallest cluster %")
    axes[2].set_xlabel("k (number of clusters)")
    axes[2].legend(loc="best", fontsize=8)

    disq_ks = table.loc[table["disqualified"], "k"].tolist()
    for ax in axes:
        ax.axvline(recommended_k, color="green", linestyle="--", alpha=0.9,
                   label=f"recommended k={recommended_k}")
        for dk in disq_ks:
            ax.axvspan(dk - 0.5, dk + 0.5, color="red", alpha=0.07)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("k-selection diagnostics  (green = recommended, red shade = disqualified)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


def fit_final_kmeans(
    scaled_df: pd.DataFrame, k: int, random_state: int = RANDOM_STATE
) -> tuple[KMeans, pd.Series]:
    """
    Clean final fit on the FULL dataset with the confirmed k (n_init=10).

    This is deliberately a fresh KMeans on all rows — NOT a reused model from one
    of the stability seeds — so the shipped clustering is fit once, cleanly, on
    every customer. Returns the model and labels indexed by customer_id.
    """
    X = scaled_df.to_numpy()
    model = KMeans(n_clusters=int(k), random_state=random_state, n_init=10)
    model.fit(X)
    labels = pd.Series(model.labels_, index=scaled_df.index, name="cluster")
    return model, labels


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


def save_artifacts(scaler: StandardScaler, model: KMeans, metadata: dict) -> None:
    """Persist scaler + model (joblib) and metadata (JSON) into models/."""
    ensure_dirs()
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(model, KMEANS_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)


def run_clustering(
    features: pd.DataFrame,
    save: bool = True,
    k_override: int | None = None,
    selection: dict | None = None,
    make_plot: bool = True,
) -> dict:
    """
    Full Phase 5 pipeline: scale -> select k -> (confirm) -> fit -> metrics ->
    UMAP -> persist.

    k is chosen by select_optimal_k() (multi-metric + stability). The final model
    is re-fit cleanly on the FULL dataset with the confirmed k:
      - ``k_override`` (when not None) fits that k instead of the recommended one
        — this is how the human confirmation gate passes an override through.
      - ``selection`` lets a caller reuse an already-computed k-selection result
        (e.g. the dashboard, which shows the diagnostic before confirming) so the
        expensive seed sweep is not repeated.

    `features` is the engineer_features() output. Returns a dict:
      result    : features + cluster label + umap_x/umap_y (per customer)
      scaler    : fitted StandardScaler
      model      : fitted KMeans (final, full-data fit)
      metrics    : cluster_metrics() dict
      metadata   : everything persisted (feature order, k-selection, metrics)
      selection  : the k-selection result (table, recommended_k, ...)
    """
    from src.features import feature_matrix

    matrix = feature_matrix(features)
    scaled, scaler = scale_features(matrix)

    if selection is None:
        selection = select_optimal_k(scaled.to_numpy())

    plot_path = None
    if make_plot:
        plot_path = str(save_diagnostic_plot(selection["table"], selection["recommended_k"]))

    final_k = int(k_override) if k_override is not None else int(selection["recommended_k"])
    model, labels = fit_final_kmeans(scaled, final_k)
    metrics = cluster_metrics(scaled, labels)
    coords = compute_umap(scaled)

    result = features.copy()
    result["cluster"] = labels
    result = result.join(coords)

    metadata = {
        "feature_columns": list(matrix.columns),
        "recommended_k": int(selection["recommended_k"]),
        "final_k": final_k,
        "best_k": final_k,  # backward-compat alias for older readers
        "k_overridden": k_override is not None and int(k_override) != int(selection["recommended_k"]),
        "recommendation_basis": selection["recommendation_basis"],
        "k_selection": selection["records"],
        "diagnostic_plot_path": plot_path,
        "metrics": metrics,
        "n_customers": int(matrix.shape[0]),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }

    if save:
        save_artifacts(scaler, model, metadata)

    return {
        "result": result,
        "scaler": scaler,
        "model": model,
        "metrics": metrics,
        "metadata": metadata,
        "selection": selection,
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