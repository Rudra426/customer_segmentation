
"""
personas.py — Phase 6: label clusters with personas and attach actions.

Takes the clustered customer table from cluster.run_clustering() and:
  6.1 profiles each cluster (size + feature averages + dominant category)
  6.2 asks the LLM to name each cluster using ONLY the fixed ACTION_MAP personas
  6.3 joins persona -> ACTION_MAP (action, channel, priority)
  6.4 attaches a numeric priority score per customer
  6.5 attaches persona / action columns onto the customer table
"""

from __future__ import annotations

import json

import pandas as pd
from pydantic import BaseModel

from config import ACTION_MAP, PERSONA_NAMES, PRIORITY_SCORE

# RFM-style fields summarized for the LLM (only those present are used).
_PROFILE_FIELDS = ["recency", "frequency", "monetary", "aov", "clv", "tenure"]

# Fallback persona when the LLM returns an invalid/missing name.
_FALLBACK_PERSONA = "Low Engagement"


def profile_clusters(result: pd.DataFrame) -> list[dict]:
    """
    Summarize each cluster for the persona labeler.

    `result` is the run_clustering() output (has a "cluster" column, numeric
    features, and optionally "top_category"). Returns one dict per cluster:
      cluster        : cluster id (int)
      size           : number of customers
      share_pct      : % of all customers
      averages       : {feature: rounded mean} for present RFM/monetary fields
      top_category   : most common dominant category in the cluster (or None)
    Clusters are ordered by descending mean monetary when available.
    """
    total = len(result)
    present = [f for f in _PROFILE_FIELDS if f in result.columns]
    profiles: list[dict] = []

    for cid, grp in result.groupby("cluster"):
        averages = {f: round(float(grp[f].mean()), 2) for f in present}
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


def _build_persona_prompt(profiles: list[dict]) -> str:
    """Describe the fixed personas + cluster profiles for the LLM."""
    persona_lines = []
    for name in PERSONA_NAMES:
        info = ACTION_MAP[name]
        persona_lines.append(f"- {name}: priority={info['priority']}, action={info['action']}")

    lines = [
        "You assign a marketing persona to each customer cluster.",
        "",
        "Choose the persona name for each cluster ONLY from this fixed list:",
        *persona_lines,
        "",
        "Guidance:",
        "- recency = days since last order (LOW = recently active).",
        "- frequency = number of orders; monetary = total spend; aov = avg order.",
        "- High frequency + high monetary + low recency => Loyal Big Spenders.",
        "- High monetary but HIGH recency (gone quiet) => At-Risk High Value.",
        "- Very new / short tenure / few orders => New Customers.",
        "- Exactly one purchase, little since => One-Time Buyers.",
        "- Low spend, low frequency, high recency => Low Engagement.",
        "- Two clusters MAY share a persona if they fit the same profile.",
        "",
        "Cluster profiles:",
        json.dumps(profiles, indent=2),
        "",
        "Return one label per cluster with a one-sentence reasoning.",
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


def label_customers(result: pd.DataFrame, save_map: bool = True) -> dict:
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

    # Persist cluster -> persona/action map so the scoring API (Phase 8) can use it.
    if save_map:
        save_segment_map(enriched)

    return {"labeled": labeled, "segments": segments}


def save_segment_map(enriched: dict[int, dict]) -> None:
    """Write the cluster -> {persona, action, channel, priority} map to disk."""
    from config import SEGMENT_MAP_PATH, ensure_dirs

    ensure_dirs()
    payload = {
        str(cid): {
            "persona": info["persona"],
            "action": info["action"],
            "channel": info["channel"],
            "priority": info["priority"],
            "priority_score": priority_score(info["priority"]),
        }
        for cid, info in enriched.items()
    }
    with open(SEGMENT_MAP_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


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
