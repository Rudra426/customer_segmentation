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

# ── Validation / QA gate (Phase 6.5) ───────────────────────────────────────
# Independent checks run AFTER feature-engineering/clustering/labeling and
# BEFORE the segmentation is trusted by the dashboard / scoring API / exports.
# All thresholds live here (never hardcoded in the check functions) so they can
# be tuned per dataset without touching code. Any failing check HALTS the
# pipeline (non-zero exit) and requires human sign-off — checks never auto-fix.
VALIDATION_REPORT_PATH = OUTPUTS_DIR / "validation_report.json"

VALIDATION: dict = {
    # 1) clusters must separate on RFM (random/fake clustering -> F ~ 1).
    "cluster_separation": {
        "rfm_cols": ["recency", "frequency", "monetary"],
        "min_variance_ratio": 2.0,      # min ANOVA F-stat on at least one RFM dim
    },
    # 2) silhouette recomputed from stored labels; catches placeholder metrics.
    "silhouette": {
        "min_score": 0.25,
        "placeholder_tol": 0.05,        # |stored - recomputed| above this = fabricated
    },
    # 3) a labeling column that is a 1:1 relabel of the segment adds no signal.
    "label_independence": {
        "pairs": [                      # (segment-ish col, derived label col)
            ["cluster", "persona"],
            ["cluster", "action"],
            ["cluster", "priority"],
        ],
    },
    # 4) per-category count columns must sum to the order frequency.
    "category_consistency": {
        "freq_col": "frequency",
        "tolerance": 0,                 # allowed |sum(cat) - freq| per row
        "max_bad_frac": 0.05,           # FAIL if > this fraction of rows violate
    },
    # 5) claimed top category must match recomputed mode of real purchases.
    "top_category_accuracy": {
        "min_match_rate": 0.40,         # also required to beat 1/n_categories chance
    },
    # 6) computed frequency must capture true orders across ALL id variants.
    "frequency_completeness": {
        "tolerance": 0.05,              # FAIL if avg(computed/true) < 1 - tolerance
    },
    # 7) orders with an unresolvable customer id must be a small minority.
    "unattributed_revenue": {
        "max_unattributed_pct": 0.05,   # by rows AND by revenue
    },
    # 8) whitespace/format near-duplicate order ids (flagged, not auto-dropped).
    "near_duplicate_orders": {
        "similarity_threshold": 0.9,
        "max_flagged_pct": 0.02,        # FAIL if more than this fraction are near-dupes
    },
    # 9) quantity sanity: missing / non-positive / non-integer / outliers.
    "qty_sanity": {
        "min_val": 1,
        "max_val": None,
        "outlier_pct": 99.5,            # values above this percentile flagged as outliers
        "max_bad_pct": 0.05,            # FAIL if hard-invalid rows exceed this fraction
        "require_integer": True,
    },
}

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
