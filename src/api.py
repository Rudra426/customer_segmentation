"""
api.py — Phase 8: FastAPI real-time scoring endpoint.

Loads the persisted scaler + K-Means model + metadata + cluster->persona map and
exposes a POST /score endpoint that assigns a single customer's feature vector to
a segment and returns the recommended action.

Built step by step: 8.1 artifact loader (this piece) -> ... -> 8.4 endpoints.

Run:  uvicorn src.api:app --reload
"""

from __future__ import annotations

import json

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import (
    CLV_MARGIN,
    KMEANS_PATH,
    METADATA_PATH,
    SCALER_PATH,
    SEGMENT_MAP_PATH,
)


class CustomerFeatures(BaseModel):
    """Input for /score. Required RFM fields; the rest are derived if omitted."""

    recency: float = Field(..., ge=0, description="Days since last order")
    frequency: int = Field(..., ge=1, description="Number of orders")
    monetary: float = Field(..., ge=0, description="Total spend")
    aov: float | None = Field(None, ge=0, description="Avg order value (derived if omitted)")
    clv: float | None = Field(None, ge=0, description="CLV proxy (derived if omitted)")
    tenure: float | None = Field(None, ge=0, description="Days active (default 0)")
    top_category: str | None = Field(
        None, description="Dominant category label, e.g. 'Apparel' (optional)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "recency": 12,
                "frequency": 5,
                "monetary": 1400.0,
                "top_category": "Accessories",
            }
        }
    }


class ScoreResponse(BaseModel):
    """Output of /score: the assigned segment and its recommended action."""

    cluster: int
    persona: str
    action: str
    channel: str
    priority: str
    priority_score: int


class ArtifactsNotFound(RuntimeError):
    """Raised when model artifacts are missing (run clustering first)."""


def load_artifacts() -> dict:
    """
    Load all scoring artifacts from models/.

    Returns a dict with:
      scaler         : fitted StandardScaler
      model          : fitted KMeans
      metadata       : model_metadata.json (feature order, metrics, drift baseline)
      feature_columns: ordered feature names the scaler/model expect
      segment_map    : {cluster_id(int): {persona, action, channel, priority, ...}}
                       (empty dict if the map has not been saved yet)

    Raises ArtifactsNotFound if the scaler/model/metadata are missing.
    """
    missing = [p.name for p in (SCALER_PATH, KMEANS_PATH, METADATA_PATH) if not p.exists()]
    if missing:
        raise ArtifactsNotFound(
            f"Missing artifacts: {missing}. Run the pipeline (clustering) first."
        )

    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(KMEANS_PATH)
    with open(METADATA_PATH, encoding="utf-8") as fh:
        metadata = json.load(fh)

    segment_map: dict[int, dict] = {}
    if SEGMENT_MAP_PATH.exists():
        with open(SEGMENT_MAP_PATH, encoding="utf-8") as fh:
            segment_map = {int(k): v for k, v in json.load(fh).items()}

    return {
        "scaler": scaler,
        "model": model,
        "metadata": metadata,
        "feature_columns": metadata.get("feature_columns", []),
        "segment_map": segment_map,
    }


# ── Feature vector construction ────────────────────────────────────────────

def build_feature_vector(feat: CustomerFeatures, feature_columns: list[str]) -> list[float]:
    """
    Build a feature vector aligned to the model's training columns.

    Derives aov (monetary/frequency) and clv (monetary * CLV_MARGIN) when not
    provided, defaults tenure to 0, and one-hot encodes top_category into the
    cat_* columns. Any training column not supplied defaults to 0.
    """
    aov = feat.aov if feat.aov is not None else (feat.monetary / max(feat.frequency, 1))
    clv = feat.clv if feat.clv is not None else (feat.monetary * CLV_MARGIN)
    tenure = feat.tenure if feat.tenure is not None else 0.0

    base = {
        "recency": float(feat.recency),
        "frequency": float(feat.frequency),
        "monetary": float(feat.monetary),
        "aov": float(aov),
        "clv": float(clv),
        "tenure": float(tenure),
    }
    if feat.top_category:
        base[f"cat_{feat.top_category}"] = 1.0

    return [float(base.get(col, 0.0)) for col in feature_columns]


# ── FastAPI app ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Customer Segmentation Scoring API",
    description="Assign a customer to a segment and return the recommended action.",
    version="1.0.0",
)

_artifacts: dict | None = None


def get_artifacts() -> dict:
    """Load artifacts once and cache them for the process lifetime."""
    global _artifacts
    if _artifacts is None:
        _artifacts = load_artifacts()
    return _artifacts


@app.get("/")
def root() -> dict:
    """Basic API info and available endpoints."""
    return {
        "service": "Customer Segmentation Scoring API",
        "version": "1.0.0",
        "endpoints": {
            "POST /score": "Score one customer -> segment + recommended action",
            "GET /health": "Artifact / readiness status",
            "GET /docs": "Interactive API documentation",
        },
    }


@app.get("/health")
def health() -> dict:
    """
    Readiness check. Returns ok=True only when scoring artifacts are loadable.
    Never raises — reports status so the API is usable before any model exists.
    """
    try:
        arts = get_artifacts()
    except ArtifactsNotFound as err:
        return {"ok": False, "ready": False, "detail": str(err)}
    except Exception as err:  # unexpected load failure
        return {"ok": False, "ready": False, "detail": f"Artifact load error: {err}"}

    return {
        "ok": True,
        "ready": True,
        "k": int(arts["model"].n_clusters),
        "n_features": len(arts["feature_columns"]),
        "segments_mapped": len(arts["segment_map"]),
        "trained_at": arts["metadata"].get("created_utc"),
    }


@app.post("/score", response_model=ScoreResponse)
def score(features: CustomerFeatures) -> ScoreResponse:
    """Score a single customer and return their segment + recommended action."""
    try:
        arts = get_artifacts()
    except ArtifactsNotFound as err:
        raise HTTPException(status_code=503, detail=str(err)) from err

    vector = build_feature_vector(features, arts["feature_columns"])
    scaled = arts["scaler"].transform([vector])
    cluster = int(arts["model"].predict(scaled)[0])

    seg = arts["segment_map"].get(cluster)
    if not seg:
        raise HTTPException(
            status_code=500,
            detail=f"No persona mapping for cluster {cluster}. Re-run labeling.",
        )

    return ScoreResponse(
        cluster=cluster,
        persona=seg["persona"],
        action=seg["action"],
        channel=seg["channel"],
        priority=seg["priority"],
        priority_score=seg["priority_score"],
    )


if __name__ == "__main__":
    arts = load_artifacts()
    print("feature_columns:", arts["feature_columns"])
    print("k:", arts["model"].n_clusters)
    print("segment_map clusters:", list(arts["segment_map"].keys()))
