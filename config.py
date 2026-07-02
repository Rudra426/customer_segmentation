"""
config.py — central configuration for the customer-segmentation pipeline.

Holds the FIXED internal schema, the hardcoded ACTION_MAP, LLM settings, and
filesystem paths. Every other module imports from here so there is a single
source of truth. Nothing in this file performs I/O at import time except
loading environment variables from .env.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env (if present) ─────────────────────────────────────────────────
load_dotenv()


def _get_secret(name: str, default: str = "") -> str:
    """
    Resolve a config value from the environment first, then Streamlit secrets.

    Locally, `.env` -> os.getenv covers everything. On Streamlit Community
    Cloud there is no .env file; secrets are entered in the app's Settings ->
    Secrets panel and exposed via st.secrets instead. Falling back to
    st.secrets here (rather than requiring it) keeps this file safe to import
    from non-Streamlit contexts too (src/api.py, scripts, tests) where
    streamlit may not even be installed/running.
    """
    val = os.getenv(name)
    if val:
        return val
    try:
        import streamlit as st  # local import: optional dependency here
        return str(st.secrets.get(name, default))
    except Exception:
        return default

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# Named artifact locations written/read by the clustering phases.
SCALER_PATH = MODELS_DIR / "scaler.joblib"
KMEANS_PATH = MODELS_DIR / "kmeans.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
SEGMENT_MAP_PATH = MODELS_DIR / "segment_map.json"
DRIFT_LOG_PATH = OUTPUTS_DIR / "drift.log"

# ── LLM (OpenRouter — OpenAI-compatible API) ───────────────────────────────
OPENROUTER_API_KEY = _get_secret("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = _get_secret("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Optional attribution headers OpenRouter uses for app rankings (harmless if unset).
OPENROUTER_APP_URL = _get_secret("OPENROUTER_APP_URL", "http://localhost")
OPENROUTER_APP_NAME = _get_secret("OPENROUTER_APP_NAME", "customer-segmentation")

# ── Fixed internal schema (DO NOT CHANGE) ──────────────────────────────────
REQUIRED_FIELDS: dict[str, str] = {
    "customer_id": "string",
    "order_id": "string",
    "order_date": "datetime",
    "order_value": "float",
}

OPTIONAL_FIELDS: dict[str, str] = {
    "quantity": "int",
    "unit_price": "float",
    "product_category": "string",
    "customer_email": "string",
    "signup_date": "datetime",
    "support_tickets": "int",
}

# Convenience: the full schema and the ordered list of all known fields.
ALL_FIELDS: dict[str, str] = {**REQUIRED_FIELDS, **OPTIONAL_FIELDS}

# ── Action map (hardcoded for v1) ──────────────────────────────────────────
# Persona name -> recommended action. The persona labeler (Phase 6) is
# constrained to choose names from these keys so every cluster maps cleanly.
ACTION_MAP: dict[str, dict[str, str]] = {
    "Loyal Big Spenders": {
        "action": "Send early access to new products / VIP perks",
        "channel": "Email + SMS",
        "priority": "retain",
    },
    "At-Risk High Value": {
        "action": "Win-back discount within 7 days",
        "channel": "Email",
        "priority": "urgent",
    },
    "New Customers": {
        "action": "Onboarding email series + second-purchase incentive",
        "channel": "Email",
        "priority": "convert",
    },
    "One-Time Buyers": {
        "action": "Light remarketing, low-cost nudge offer",
        "channel": "Retargeting ads",
        "priority": "monitor",
    },
    "Low Engagement": {
        "action": "Remove from paid spend, keep on passive email only",
        "channel": "Passive email",
        "priority": "deprioritize",
    },
}

# Allowed persona names, exposed for prompt construction / validation.
PERSONA_NAMES: list[str] = list(ACTION_MAP.keys())

# Maps each priority label to a numeric weight (higher = more attention).
PRIORITY_SCORE: dict[str, int] = {
    "urgent": 5,
    "retain": 4,
    "convert": 3,
    "monitor": 2,
    "deprioritize": 1,
}

# ── Clustering: multi-metric, stability-checked k-selection ────────────────
# k is no longer chosen by silhouette alone over a fixed K_MIN..K_MAX range.
# select_optimal_k() searches K_SEARCH_RANGE, refits each k STABILITY_SEEDS times,
# and combines silhouette + Davies-Bouldin + label stability (pairwise Adjusted
# Rand Index) while disqualifying any k that fragments into a cluster smaller
# than MIN_CLUSTER_PCT of customers. The recommended k is the highest-silhouette
# survivor whose stability exceeds MIN_STABILITY_ARI.
K_SEARCH_RANGE = (2, 15)  # inclusive (k_min, k_max) searched during k-selection
MIN_CLUSTER_PCT = 0.03    # disqualify k whose smallest cluster < this fraction
STABILITY_SEEDS = 8       # KMeans refits per k (distinct seeds) for stability/avg
MIN_STABILITY_ARI = 0.7   # recommended k must have avg pairwise ARI above this
RANDOM_STATE = 42         # base seed for reproducibility (KMeans / UMAP)
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

# k-selection diagnostic plot (silhouette / Davies-Bouldin / min-cluster-% vs k).
DIAGNOSTIC_PLOT_PATH = OUTPUTS_DIR / "k_selection_diagnostic.png"

# Assumed gross margin used for the simple CLV proxy (profit contribution).
CLV_MARGIN = 0.30

# ── Drift detection (Phase 10) ─────────────────────────────────────────────
DRIFT_ALPHA = 0.05            # KS-test p-value below this flags feature drift
SILHOUETTE_DROP_FRAC = 0.25   # silhouette dropping >25% vs baseline flags drift

# ── Revenue Impact view ────────────────────────────────────────────────────
# Substring patterns (normalized: lowercased, '-'/'_' -> space) used to detect
# "at-risk" segments from whatever persona labels the pipeline produced.
AT_RISK_PATTERNS = ["at risk", "at-risk", "dormant", "churn", "lapsed"]

# ── LLM call settings ──────────────────────────────────────────────────────
LLM_SAMPLE_VALUES = 5     # sample values per column sent to the schema mapper
LLM_MAX_RETRIES = 3       # retries on transient API / JSON-parse failures

# ── Schema-mapping validation thresholds ───────────────────────────────────
# Mappings at/above this confidence are trusted; below it we ask the user.
MAPPING_CONFIDENCE_THRESHOLD = 0.6
# If fewer than this many of the 4 REQUIRED fields are confidently mapped, we
# treat the upload as "probably not e-commerce data" and reject with guidance.
MIN_REQUIRED_FOR_ECOMMERCE = 2


def ensure_dirs() -> None:
    """Create all output directories if they do not already exist."""
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def has_api_key() -> bool:
    """True if a non-placeholder OpenRouter API key is configured."""
    return bool(OPENROUTER_API_KEY) and OPENROUTER_API_KEY != "your_openrouter_key_here"