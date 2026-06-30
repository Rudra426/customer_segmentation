"""
test_cluster.py — unit tests for the multi-metric, stability-checked k-selection.

The headline test builds a synthetic fixture with 3 well-separated blobs and
asserts that select_optimal_k() recovers k=3 even though K_SEARCH_RANGE goes up
to 15. Supporting tests cover the returned table contract, the tiny-cluster
disqualification rule, and the clean full-data final fit.

Run:  python -m pytest tests/test_cluster.py   (or)  python tests/test_cluster.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cluster import fit_final_kmeans, scale_features, select_optimal_k


def _three_blobs(n_samples: int = 300, random_state: int = 42) -> np.ndarray:
    """Three compact, well-separated Gaussian blobs in 2D (the true k is 3)."""
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=[[0.0, 0.0], [10.0, 10.0], [20.0, 0.0]],
        cluster_std=0.6,
        random_state=random_state,
    )
    return X


def test_recovers_k_three_despite_wide_range():
    """select_optimal_k must pick k=3 on 3 well-separated blobs, range up to 15."""
    X = _three_blobs()
    sel = select_optimal_k(X, k_range=(2, 15), n_seeds=8)

    assert sel["recommended_k"] == 3
    # The true partition should be perfectly stable and clearly the best silhouette.
    row3 = sel["table"].set_index("k").loc[3]
    assert row3["avg_stability_ari"] > 0.7
    assert not bool(row3["disqualified"])
    best_sil_k = int(sel["table"].loc[sel["table"]["avg_silhouette"].idxmax(), "k"])
    assert best_sil_k == 3


def test_table_contract():
    """The diagnostic table must expose exactly the agreed columns, one row per k."""
    X = _three_blobs(n_samples=150)
    sel = select_optimal_k(X, k_range=(2, 8), n_seeds=4)
    table = sel["table"]

    assert list(table.columns) == [
        "k",
        "avg_silhouette",
        "avg_davies_bouldin",
        "avg_stability_ari",
        "min_cluster_pct",
        "disqualified",
    ]
    # One row per k in the (clamped) search range, in ascending order.
    assert list(table["k"]) == [2, 3, 4, 5, 6, 7, 8]
    # records mirror the table for JSON serialization.
    assert len(sel["records"]) == len(table)
    assert sel["recommendation_basis"]
    assert sel["k_range"] == [2, 8]
    assert sel["n_seeds"] == 4


def test_tiny_cluster_disqualified():
    """A k that fragments into a sub-threshold cluster is flagged disqualified."""
    # Three well-separated blobs, but one is tiny (25 of 465 rows ~ 5.4%). With an
    # 8% floor the true k=3 — which has the BEST silhouette — must be disqualified.
    X, _ = make_blobs(
        n_samples=[220, 220, 25],
        centers=[[0.0, 0.0], [12.0, 12.0], [24.0, 0.0]],
        cluster_std=0.6,
        random_state=42,
    )
    sel = select_optimal_k(X, k_range=(2, 6), n_seeds=4, min_cluster_pct=0.08)
    table = sel["table"].set_index("k")

    # k=3 isolates the tiny blob (~5.4%) below the 8% floor -> disqualified,
    # even though it has the single highest avg_silhouette.
    assert bool(table.loc[3, "disqualified"])
    best_sil_k = int(sel["table"].loc[sel["table"]["avg_silhouette"].idxmax(), "k"])
    assert best_sil_k == 3
    # The recommendation must NOT land on the disqualified high-silhouette k.
    assert sel["recommended_k"] != 3
    rec_row = table.loc[sel["recommended_k"]]
    assert not bool(rec_row["disqualified"])


def test_final_fit_uses_full_dataset():
    """fit_final_kmeans re-fits cleanly on every row and labels the full index."""
    X = _three_blobs(n_samples=240)
    matrix = pd.DataFrame(X, columns=["f0", "f1"], index=[f"c{i}" for i in range(len(X))])
    scaled, _ = scale_features(matrix)

    model, labels = fit_final_kmeans(scaled, k=3)

    assert model.n_clusters == 3
    # Every customer in the input is labeled (full-data fit, no sampling).
    assert len(labels) == len(matrix)
    assert list(labels.index) == list(matrix.index)
    assert set(labels.unique()) == {0, 1, 2}


def test_raises_when_too_few_rows():
    """A search range that cannot satisfy k+1 rows fails loudly, not silently."""
    X = _three_blobs(n_samples=4)
    try:
        select_optimal_k(X, k_range=(10, 15), n_seeds=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for too few rows")


if __name__ == "__main__":
    test_recovers_k_three_despite_wide_range()
    test_table_contract()
    test_tiny_cluster_disqualified()
    test_final_fit_uses_full_dataset()
    test_raises_when_too_few_rows()
    print("All cluster tests passed.")
