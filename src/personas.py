
"""
personas.py — Phase 6: label clusters with personas and attach actions.

Takes the clustered customer table from cluster.run_clustering() and:
  6.1 profiles each cluster (size + feature averages + dataset-relative bands +
      dominant category)
  6.2 asks the LLM to name each cluster using ONLY the fixed ACTION_MAP personas,
      judging each cluster by its bands relative to THIS dataset (so it works on
      any dataset) and citing that evidence in its reasoning
  6.3 joins persona -> ACTION_MAP (action, channel, priority)
  6.4 attaches a numeric priority score per customer
  6.5 attaches persona / action columns onto the customer table
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pydantic import BaseModel

from config import ACTION_MAP, PERSONA_NAMES, PRIORITY_SCORE

# RFM-style fields summarized for the LLM (only those present are used).
_PROFILE_FIELDS = ["recency", "frequency", "monetary", "aov", "clv", "tenure"]

# Fallback persona when the LLM returns an invalid/missing name.
_FALLBACK_PERSONA = "Low Engagement"

# Ordered relative bands. A cluster's mean for a feature is placed into one of
# these by comparing it to that feature's distribution ACROSS THIS DATASET's
# customers (quintile edges). This gives the LLM a dataset-relative sense of
# "high/low" so persona labeling works on any dataset regardless of the absolute
# magnitudes (a $37k mean is meaningless without knowing the population spread).
_BANDS = ["very_low", "low", "mid", "high", "very_high"]


def _relative_bands(result: pd.DataFrame, present: list[str]) -> dict[str, dict]:
    """
    Per-feature machinery for placing a cluster mean on a dataset-relative band.

    Returns {feature: band} where `band` is a callable mapping a value to one of
    _BANDS by comparing it to that feature's [q20, q40, q60, q80] quintile edges
    over the dataset's customers. Whether a feature actually separates the
    clusters is derived later from the resulting bands (see _separating_features).
    """
    info: dict[str, dict] = {}
    for f in present:
        col = result[f].to_numpy(dtype="float64")
        edges = np.quantile(col, [0.2, 0.4, 0.6, 0.8])

        def band(value: float, _edges=edges) -> str:
            return _BANDS[min(int(np.searchsorted(_edges, value, side="right")), 4)]

        info[f] = {"band": band}
    return info


def profile_clusters(result: pd.DataFrame) -> list[dict]:
    """
    Summarize each cluster for the persona labeler.

    `result` is the run_clustering() output (has a "cluster" column, numeric
    features, and optionally "top_category"). Returns one dict per cluster:
      cluster        : cluster id (int)
      size           : number of customers
      share_pct      : % of all customers
      averages       : {feature: rounded mean} for present RFM/monetary fields
      relative       : {feature: band} — where the cluster's mean sits relative
                       to THIS dataset's customers (very_low..very_high). This is
                       the frame the LLM uses to judge "high/low" on any dataset.
      top_category   : most common dominant category in the cluster (or None)
    Clusters are ordered by descending mean monetary when available.
    """
    total = len(result)
    present = [f for f in _PROFILE_FIELDS if f in result.columns]
    bands = _relative_bands(result, present)
    profiles: list[dict] = []

    for cid, grp in result.groupby("cluster"):
        averages = {f: round(float(grp[f].mean()), 2) for f in present}
        relative = {f: bands[f]["band"](averages[f]) for f in present}
        top_category = None
        if "top_category" in grp.columns and grp["top_category"].notna().any():
            top_category = grp["top_category"].mode(dropna=True)
            top_category = str(top_category.iloc[0]) if not top_category.empty else None
        profiles.append(
            {
                "cluster": int(cid),
                "size": int(len(grp)),
                "share_pct": round(100 * len(grp) / total, 1),
                "averages": averages,
                "relative": relative,
                "top_category": top_category,
            }
        )

    sort_key = "monetary" if "monetary" in present else None
    if sort_key:
        profiles.sort(key=lambda p: p["averages"].get(sort_key, 0), reverse=True)
    return profiles


class ClusterLabel(BaseModel):
    """One cluster -> persona decision from the LLM."""

    cluster: int
    persona: str  # must be one of PERSONA_NAMES (validated post-hoc)
    reasoning: str


class ClusterLabelList(BaseModel):
    """Object wrapper so the response is a JSON object (JSON-mode friendly)."""

    labels: list[ClusterLabel]


def _separating_features(profiles: list[dict]) -> tuple[list[str], list[str]]:
    """
    Split present features into those that DO vs DON'T separate the clusters.

    A feature separates the clusters if its relative band is not identical for
    every cluster. Reporting this stops the LLM from inventing behavioral
    distinctions on a dimension where all clusters actually sit in the same band
    (the failure mode where 12 near-identical clusters all became one persona).
    """
    present = list(profiles[0].get("relative", {})) if profiles else []
    separating, flat = [], []
    for f in present:
        bands = {p["relative"].get(f) for p in profiles}
        (separating if len(bands) > 1 else flat).append(f)
    return separating, flat


def _build_persona_prompt(profiles: list[dict]) -> str:
    """Describe the fixed personas + cluster profiles for the LLM."""
    persona_lines = []
    for name in PERSONA_NAMES:
        info = ACTION_MAP[name]
        persona_lines.append(f"- {name}: priority={info['priority']}, action={info['action']}")

    separating, flat = _separating_features(profiles)
    sep_txt = ", ".join(separating) if separating else "NONE"
    flat_txt = ", ".join(flat) if flat else "none"

    lines = [
        "You assign a marketing persona to each customer cluster.",
        "",
        "Choose the persona name for each cluster ONLY from this fixed list:",
        *persona_lines,
        "",
        "Each cluster has an `averages` block (raw means) AND a `relative` block.",
        "The `relative` bands (very_low < low < mid < high < very_high) place each",
        "cluster's mean against THIS dataset's own customer distribution. ALWAYS",
        "judge a cluster by its `relative` bands, never by raw magnitudes — a",
        "monetary of 37000 may be very_low in one dataset and very_high in another.",
        "",
        "Field meanings:",
        "- recency = days since last order; LOW band = recently active, HIGH = gone quiet.",
        "- frequency = number of orders; monetary = total spend; aov = avg order.",
        "- tenure = how long the customer has been active.",
        "",
        "Map bands to personas (relative, so this works on any dataset):",
        "- monetary & frequency high/very_high AND recency low/very_low => Loyal Big Spenders.",
        "- monetary high/very_high BUT recency high/very_high (gone quiet) => At-Risk High Value.",
        "- tenure low/very_low AND frequency low/very_low (just arrived) => New Customers.",
        "- frequency very_low (≈1 order) with low monetary => One-Time Buyers.",
        "- monetary & frequency low/very_low AND recency high => Low Engagement.",
        "",
        "How to differentiate the clusters:",
        f"- Bands that DIFFER across clusters (use these to tell segments apart): {sep_txt}.",
        f"- Bands that are IDENTICAL across all clusters (carry no signal): {flat_txt}.",
        "- If two clusters share the same bands on every separating feature, give them",
        "  the SAME persona — do not invent a distinction from noise.",
        "- top_category is a product preference, NOT a lifecycle stage: never let it",
        "  alone change the persona. If clusters differ ONLY by top_category, they",
        "  describe the same behavior and should share a persona.",
        "",
        "Cluster profiles:",
        json.dumps(profiles, indent=2),
        "",
        "For each cluster return `reasoning` as ONE sentence that CITES the specific",
        "bands driving the choice (e.g. \"recency=very_low, frequency=high,",
        "monetary=very_high => Loyal Big Spenders\"). Do not restate the persona",
        "definition or mention top_category as a reason. Return one label per cluster.",
    ]
    return "\n".join(lines)


def label_clusters(profiles: list[dict]) -> dict[int, dict]:
    """
    Ask the LLM to name each cluster, constrained to the fixed personas.

    Returns {cluster_id: {"persona": str, "reasoning": str}}. Any persona not in
    PERSONA_NAMES is replaced with the fallback, and any cluster the LLM omits is
    filled with the fallback so every cluster is always covered.
    """
    from src.llm import generate_structured

    prompt = _build_persona_prompt(profiles)
    result = generate_structured(prompt, response_schema=ClusterLabelList)

    valid = set(PERSONA_NAMES)
    labels: dict[int, dict] = {}
    for item in result.labels:
        persona = item.persona if item.persona in valid else _FALLBACK_PERSONA
        labels[int(item.cluster)] = {"persona": persona, "reasoning": item.reasoning}

    # Ensure every profiled cluster has a label.
    for p in profiles:
        labels.setdefault(
            p["cluster"], {"persona": _FALLBACK_PERSONA, "reasoning": "No label returned."}
        )
    return labels


def map_actions(labels: dict[int, dict]) -> dict[int, dict]:
    """
    Enrich cluster labels with their ACTION_MAP entry.

    For each {cluster: {persona, reasoning}} adds action, channel, and priority
    from ACTION_MAP. Returns a new dict {cluster: {persona, reasoning, action,
    channel, priority}}. Unknown personas fall back to the fallback persona's
    action so a row is always produced.
    """
    enriched: dict[int, dict] = {}
    for cid, info in labels.items():
        persona = info["persona"]
        action = ACTION_MAP.get(persona, ACTION_MAP[_FALLBACK_PERSONA])
        enriched[cid] = {
            "persona": persona,
            "reasoning": info.get("reasoning", ""),
            "action": action["action"],
            "channel": action["channel"],
            "priority": action["priority"],
        }
    return enriched


def priority_score(priority: str) -> int:
    """
    Map a priority label to its numeric weight (higher = more attention).

    Uses config.PRIORITY_SCORE: urgent=5, retain=4, convert=3, monitor=2,
    deprioritize=1. Unknown labels score 0.
    """
    return int(PRIORITY_SCORE.get(priority, 0))


def label_customers(result: pd.DataFrame) -> dict:
    """
    Full Phase 6 pipeline: profile clusters -> LLM personas -> actions ->
    attach per-customer columns.

    `result` is the run_clustering() output (must have a "cluster" column).
    Returns a dict:
      labeled  : customer table + persona, action, channel, priority,
                 priority_score columns
      segments : per-cluster summary (profile + persona + action) for the UI
    """
    profiles = profile_clusters(result)
    labels = label_clusters(profiles)
    enriched = map_actions(labels)

    labeled = result.copy()
    cluster = labeled["cluster"]
    labeled["persona"] = cluster.map(lambda c: enriched[c]["persona"])
    labeled["action"] = cluster.map(lambda c: enriched[c]["action"])
    labeled["channel"] = cluster.map(lambda c: enriched[c]["channel"])
    labeled["priority"] = cluster.map(lambda c: enriched[c]["priority"])
    labeled["priority_score"] = labeled["priority"].map(priority_score)

    # Per-segment summary (profile joined with persona/action) for the dashboard.
    segments = []
    for p in profiles:
        cid = p["cluster"]
        seg = {**p, **enriched[cid]}
        seg["priority_score"] = priority_score(seg["priority"])
        segments.append(seg)
    segments.sort(key=lambda s: s["priority_score"], reverse=True)

    return {"labeled": labeled, "segments": segments}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    import warnings

    warnings.filterwarnings("ignore")
    from config import RAW_DIR
    from src.cleaner import clean_data
    from src.cluster import run_clustering
    from src.features import engineer_features
    from src.schema_mapper import apply_mapping, load_raw_file

    mapping = {
        "Cust ID": "customer_id", "Order #": "order_id", "Order Date": "order_date",
        "Total $": "order_value", "SKU Category": "product_category", "Qty": "quantity",
    }
    clean_df, _ = clean_data(apply_mapping(load_raw_file(RAW_DIR / "sample_messy.csv"), mapping))
    out = run_clustering(engineer_features(clean_df), save=False)
    labeled = label_customers(out["result"])
    cols = ["cluster", "persona", "priority", "priority_score", "monetary", "frequency"]
    print(labeled["labeled"][cols].to_string())
