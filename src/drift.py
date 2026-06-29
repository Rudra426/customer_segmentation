"""
drift.py — Phase 10: detect data/segmentation drift and signal retraining.

Compares newly-arrived customer features against the training baseline saved in
model_metadata.json (by cluster.run_clustering) and flags drift via:
  10.2 per-feature KS-test
  10.3 silhouette-degradation check
  10.4 a scheduler hook that logs warnings
  10.5 cron/Airflow wiring (documented in README)

The training baseline (per-feature stats + samples) is captured at train time, so
this module only needs to LOAD it here in 10.1.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import silhouette_score

from config import (
    DRIFT_ALPHA,
    DRIFT_LOG_PATH,
    KMEANS_PATH,
    METADATA_PATH,
    SCALER_PATH,
    SILHOUETTE_DROP_FRAC,
    ensure_dirs,
)


class BaselineNotFound(RuntimeError):
    """Raised when no training baseline is available (train a model first)."""


def load_baseline() -> dict:
    """
    Load the training drift baseline from model_metadata.json.

    Returns a dict:
      feature_columns : ordered feature names
      stats           : {feature: {mean, std, min, max, median}}
      sample          : {feature: [training values]}  (for the KS-test)
      train_metrics   : the clustering metrics recorded at train time

    Raises BaselineNotFound if metadata or its drift baseline is missing.
    """
    if not METADATA_PATH.exists():
        raise BaselineNotFound(
            "No model_metadata.json found. Run the pipeline (clustering) first."
        )
    with open(METADATA_PATH, encoding="utf-8") as fh:
        metadata = json.load(fh)

    baseline = metadata.get("drift_baseline")
    if not baseline or "sample" not in baseline:
        raise BaselineNotFound("Metadata has no drift baseline. Retrain to capture it.")

    return {
        "feature_columns": metadata.get("feature_columns", []),
        "stats": baseline.get("stats", {}),
        "sample": baseline.get("sample", {}),
        "train_metrics": metadata.get("metrics", {}),
    }


def ks_drift(new_features: pd.DataFrame, baseline: dict | None = None,
             alpha: float = DRIFT_ALPHA) -> dict:
    """
    Per-feature Kolmogorov-Smirnov drift test: new data vs the training sample.

    For each feature present in both the new data and the baseline, runs a
    two-sample KS-test. A feature is "drifted" when its p-value < alpha (its
    distribution differs significantly from training).

    Returns:
      per_feature      : {feature: {statistic, p_value, drifted}}
      drifted_features : [feature, ...] that drifted
      n_drifted        : count
      alpha            : threshold used
    """
    if baseline is None:
        baseline = load_baseline()
    sample = baseline["sample"]

    per_feature: dict[str, dict] = {}
    drifted: list[str] = []
    for col in baseline["feature_columns"]:
        if col not in new_features.columns or col not in sample:
            continue
        ref = pd.Series(sample[col], dtype="float64").dropna()
        cur = new_features[col].astype("float64").dropna()
        if ref.empty or cur.empty:
            continue
        stat, p = ks_2samp(cur, ref)
        is_drift = bool(p < alpha)
        per_feature[col] = {
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 4),
            "drifted": is_drift,
        }
        if is_drift:
            drifted.append(col)

    return {
        "per_feature": per_feature,
        "drifted_features": drifted,
        "n_drifted": len(drifted),
        "alpha": alpha,
    }


def silhouette_degradation(new_features: pd.DataFrame, baseline: dict | None = None,
                           drop_frac: float = SILHOUETTE_DROP_FRAC) -> dict:
    """
    Check whether clustering quality degraded on new data.

    Assigns new customers to the existing clusters with the saved scaler+model,
    computes the silhouette on the new data, and compares it to the training
    baseline. Flags `degraded=True` when the silhouette falls by more than
    drop_frac (e.g. 0.25 = a 25% relative drop).

    Returns {baseline_silhouette, current_silhouette, drop_frac_observed,
    degraded}. current_silhouette is None if it cannot be computed (e.g. all new
    points land in one cluster).
    """
    if baseline is None:
        baseline = load_baseline()
    base_sil = baseline["train_metrics"].get("silhouette")

    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(KMEANS_PATH)

    cols = baseline["feature_columns"]
    # Align to the trained feature columns: add any missing one-hot/category
    # columns as 0 and drop unseen ones, so a new export with different (or no)
    # categories does not raise KeyError. Mirrors the scoring API's behavior.
    aligned = new_features.reindex(columns=cols, fill_value=0.0).astype("float64")
    X = scaler.transform(aligned.to_numpy())
    labels = model.predict(X)

    if len(set(labels.tolist())) < 2 or base_sil in (None, 0):
        current = None
        observed = None
        degraded = False
    else:
        current = round(float(silhouette_score(X, labels)), 4)
        observed = round((base_sil - current) / abs(base_sil), 4)
        degraded = bool(observed > drop_frac)

    return {
        "baseline_silhouette": base_sil,
        "current_silhouette": current,
        "drop_frac_observed": observed,
        "degraded": degraded,
    }


def _log_drift(message: str) -> None:
    """Append a timestamped line to the drift log."""
    ensure_dirs()
    stamp = pd.Timestamp.now().isoformat(timespec="seconds")
    with open(DRIFT_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(f"{stamp}  {message}\n")


def run_drift_check(new_features: pd.DataFrame, log: bool = True) -> dict:
    """
    Run all drift checks and produce a retraining signal (scheduler entry point).

    Combines the per-feature KS-test and the silhouette-degradation check.
    Recommends retraining if any feature drifted OR clustering quality degraded.
    Prints a warning and appends a line to outputs/drift.log when `log` is True.

    Returns {ks, silhouette, retrain_recommended, reasons}.
    """
    baseline = load_baseline()
    ks = ks_drift(new_features, baseline)
    sil = silhouette_degradation(new_features, baseline)

    reasons: list[str] = []
    if ks["n_drifted"] > 0:
        reasons.append(f"{ks['n_drifted']} feature(s) drifted: {ks['drifted_features']}")
    if sil["degraded"]:
        reasons.append(
            f"silhouette dropped {sil['drop_frac_observed']:.0%} "
            f"({sil['baseline_silhouette']} -> {sil['current_silhouette']})"
        )
    retrain = bool(reasons)

    report = {
        "ks": ks,
        "silhouette": sil,
        "retrain_recommended": retrain,
        "reasons": reasons,
        "n_rows_checked": int(len(new_features)),
    }

    if log:
        if retrain:
            msg = "DRIFT DETECTED - retraining recommended: " + "; ".join(reasons)
            print("WARNING:", msg)
        else:
            msg = "No significant drift detected."
        _log_drift(f"[rows={len(new_features)}] {msg}")

    return report


if __name__ == "__main__":
    base = load_baseline()
    print("baseline features:", base["feature_columns"])
    print("train silhouette :", base["train_metrics"].get("silhouette"))
    print("sample sizes     :", {k: len(v) for k, v in base["sample"].items()})
