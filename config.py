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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Optional attribution headers OpenRouter uses for app rankings (harmless if unset).
OPENROUTER_APP_URL = os.getenv("OPENROUTER_APP_URL", "http://localhost")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "customer-segmentation")

# ── Fixed internal schema (DO NOT CHANGE) ──────────────────────────────────
REQUIRED_FIELDS: dict[str, str] = {
    "customer_id": "string",
    "order_id": "string",
    "order_date": "datetime",
    "order_value": "float",
}

OPTIONAL_FIELDS: dict[str, str] = {
    "quantity": "int",
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

# ── Clustering parameters ──────────────────────────────────────────────────
K_MIN = 3                 # smallest k tried during silhouette search
K_MAX = 8                 # largest k tried
RANDOM_STATE = 42         # reproducibility for KMeans / UMAP
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

# Assumed gross margin used for the simple CLV proxy (profit contribution).
CLV_MARGIN = 0.30

# ── Drift detection (Phase 10) ─────────────────────────────────────────────
DRIFT_ALPHA = 0.05            # KS-test p-value below this flags feature drift
SILHOUETTE_DROP_FRAC = 0.25   # silhouette dropping >25% vs baseline flags drift

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
