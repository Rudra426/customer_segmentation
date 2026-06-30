"""
test_validation.py — unit tests for the segmentation QA gate (src/validation.py).

Run (no pytest required):
    python -m unittest tests.test_validation -v

Strategy: build ONE "good" fixture (a segmented customer table + matching raw
orders) that must pass all nine checks, then derive a "bad" fixture per known
failure mode that must fail exactly its corresponding check. Each failure mode
mirrors a real prior bug:
  1 random/fake clustering            -> cluster_separation
  2 overlapping clusters / placeholder-> silhouette
  3 persona == 1:1 relabel of segment -> label_independence
  4 category counts != frequency      -> category_consistency
  5 fabricated top category           -> top_category_accuracy
  6 partial entity resolution (38%)   -> frequency_completeness
  7 unresolvable customer ids         -> unattributed_revenue
  8 whitespace/format near-dupe orders-> near_duplicate_orders
  9 messy quantities                  -> qty_sanity
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src import validation as V  # noqa: E402

CATS = ["Books", "Toys", "Sports", "Beauty"]
ID_FORMATS = [
    lambda n: f"CUST{n:05d}",
    lambda n: f"C-{n}",
    lambda n: f"cid{n:05d}",
    lambda n: f"{n:06d}",
]
# Per-cluster (recency_base, frequency, monetary_per_order) — well separated.
CLUSTER_SPECS = {
    0: {"recency": 5, "freq": 10, "value": 300.0, "persona": "Loyal Big Spenders"},
    1: {"recency": 40, "freq": 5, "value": 120.0, "persona": "Loyal Big Spenders"},
    2: {"recency": 200, "freq": 1, "value": 25.0, "persona": "Low Engagement"},
}
PERSONA_ACTION = {
    "Loyal Big Spenders": ("VIP perks", "retain"),
    "Low Engagement": ("Passive email", "deprioritize"),
}
N_PER_CLUSTER = 20


def build_good():
    """Return a dict of run_validation kwargs for a fully-consistent fixture."""
    cust_rows = []
    raw_rows = []
    core = 0
    for cluster, spec in CLUSTER_SPECS.items():
        for j in range(N_PER_CLUSTER):
            core += 1
            freq = spec["freq"]
            rep_id = ID_FORMATS[0](core)  # representative format for the customer table

            # Build `freq` orders with a clear dominant category.
            top_cat = CATS[core % len(CATS)]
            cat_counts = {c: 0 for c in CATS}
            for k in range(freq):
                # 60%+ of orders in the top category -> unambiguous mode.
                cat = top_cat if k < max(1, int(freq * 0.6) + 1) else CATS[(core + k) % len(CATS)]
                cat_counts[cat] += 1
                # Rotate id format across this customer's orders (all same entity).
                raw_id = ID_FORMATS[k % len(ID_FORMATS)](core)
                raw_rows.append(
                    {
                        "customer_id": raw_id,
                        "order_id": f"ORD{core:05d}{k:03d}",
                        "order_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=core + k),
                        "order_value": spec["value"],
                        "product_category": cat,
                        "quantity": 1 + (k % 5),
                    }
                )
            # Recompute the true top category from the generated counts.
            true_top = max(cat_counts, key=lambda c: (cat_counts[c], c == top_cat))
            monetary = round(spec["value"] * freq, 2)
            recency = spec["recency"] + j  # small spread within cluster
            action, priority = PERSONA_ACTION[spec["persona"]]
            row = {
                "customer_id": rep_id,
                "recency": recency,
                "frequency": freq,
                "monetary": monetary,
                "top_category": true_top,
                "cluster": cluster,
                "persona": spec["persona"],
                "action": action,
                "priority": priority,
            }
            for c in CATS:
                row[f"cat_{c}"] = cat_counts[c]
            cust_rows.append(row)

    df = pd.DataFrame(cust_rows).set_index("customer_id")
    raw = pd.DataFrame(raw_rows)

    cat_cols = [f"cat_{c}" for c in CATS]
    X = StandardScaler().fit_transform(df[["recency", "frequency", "monetary"]].to_numpy())
    labels = df["cluster"].to_numpy()
    stored = float(silhouette_score(X, labels))  # honest stored metric

    return {
        "df": df,
        "X": X,
        "cluster_labels": labels,
        "raw_orders_df": raw,
        "cat_cols": cat_cols,
        "stored_silhouette": stored,
        "write_report": False,
        "print_summary": False,
    }


class GoodFixtureTests(unittest.TestCase):
    def test_good_fixture_passes_all_checks(self):
        report = V.run_validation(**build_good())
        failing = [n for n, e in report["checks"].items() if not e["pass"] and not e["skipped"]]
        self.assertTrue(report["overall_pass"], f"unexpected failures: {failing}")
        # All nine logical checks should be present (label_independence -> 3 pairs).
        names = set(report["checks"])
        self.assertIn("cluster_separation", names)
        self.assertIn("silhouette_score", names)
        self.assertIn("category_consistency", names)
        self.assertIn("top_category_accuracy", names)
        self.assertIn("frequency_completeness", names)
        self.assertIn("unattributed_revenue", names)
        self.assertIn("near_duplicate_orders", names)
        self.assertIn("qty_sanity", names)
        self.assertTrue(any(n.startswith("label_independence") for n in names))


class CheckFailureTests(unittest.TestCase):
    """Each test breaks exactly one thing and asserts the matching check fails."""

    def setUp(self):
        self.good = build_good()
        self.df = self.good["df"]
        self.raw = self.good["raw_orders_df"]
        self.cat_cols = self.good["cat_cols"]

    # 1
    def test_cluster_separation_fails_on_random_clusters(self):
        rng = np.random.default_rng(0)
        df = self.df.copy()
        df["cluster"] = rng.integers(0, 3, size=len(df))
        passed, detail = V.check_cluster_separation(df, "cluster", ["recency", "frequency", "monetary"])
        self.assertFalse(passed)
        self.assertLess(detail["max_f_stat"], 2.0)

    def test_cluster_separation_passes_on_real_clusters(self):
        passed, _ = V.check_cluster_separation(self.df, "cluster", ["recency", "frequency", "monetary"])
        self.assertTrue(passed)

    # 2
    def test_silhouette_fails_when_below_threshold(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(60, 3))  # no structure
        labels = rng.integers(0, 3, size=60)
        passed, detail = V.check_silhouette_score(X, labels, min_score=0.25)
        self.assertFalse(passed)
        self.assertLess(detail["recomputed_score"], 0.25)

    def test_silhouette_fails_on_placeholder_mismatch(self):
        passed, detail = V.check_silhouette_score(
            self.good["X"], self.good["cluster_labels"], min_score=0.25, stored_score=0.0
        )
        # recomputed is high (passes threshold) but stored 0.0 is a placeholder
        self.assertFalse(passed)
        self.assertIn("placeholder_warning", detail)

    # 3
    def test_label_independence_fails_on_bijective_relabel(self):
        df = self.df.copy()
        # persona becomes a unique 1:1 rename of each cluster -> no new signal
        df["persona"] = df["cluster"].map({0: "A", 1: "B", 2: "C"})
        passed, detail = V.check_label_independence(df, "cluster", "persona")
        self.assertFalse(passed)
        self.assertTrue(detail["bijective"])

    def test_label_independence_passes_on_coarsening(self):
        passed, detail = V.check_label_independence(self.df, "cluster", "persona")
        self.assertTrue(passed)
        self.assertFalse(detail["bijective"])

    # 4
    def test_category_consistency_fails_when_counts_not_summing_to_freq(self):
        df = self.df.copy()
        # Corrupt category counts for ~30% of rows.
        idx = df.index[: int(len(df) * 0.3)]
        df.loc[idx, "cat_Books"] = df.loc[idx, "cat_Books"] + 7
        passed, detail = V.check_category_consistency(df, self.cat_cols, "frequency", 0, 0.05)
        self.assertFalse(passed)
        self.assertGreater(detail["bad_row_frac"], 0.05)

    # 5
    def test_top_category_accuracy_fails_on_fabricated_labels(self):
        df = self.df.copy()
        df["top_category"] = "Beauty"  # constant fabricated label, ignores history
        passed, detail = V.check_top_category_accuracy(
            df, "top_category", self.raw, "customer_id", "product_category", 0.40
        )
        self.assertFalse(passed)
        self.assertLessEqual(detail["match_rate"], detail["threshold"])

    def test_top_category_accuracy_passes_on_real_labels(self):
        passed, _ = V.check_top_category_accuracy(
            self.df, "top_category", self.raw, "customer_id", "product_category", 0.40
        )
        self.assertTrue(passed)

    # 6
    def test_frequency_completeness_fails_on_undercount(self):
        df = self.df.copy()
        # Simulate partial entity resolution: only ~38% of true orders captured.
        df["frequency"] = np.ceil(df["frequency"] * 0.38).astype(int)
        passed, detail = V.check_frequency_completeness(
            df, "frequency", self.raw, "customer_id", "canonical_id", 0.05
        )
        self.assertFalse(passed)
        self.assertLess(detail["avg_capture_ratio"], 0.95)

    def test_frequency_completeness_passes_when_all_variants_resolved(self):
        passed, detail = V.check_frequency_completeness(
            self.df, "frequency", self.raw, "customer_id", "canonical_id", 0.05
        )
        self.assertTrue(passed)
        self.assertGreaterEqual(detail["avg_capture_ratio"], 0.95)

    # 7
    def test_unattributed_revenue_fails_with_many_unresolved_ids(self):
        raw = self.raw.copy()
        # Make ~15% of orders have unresolvable customer ids.
        n = int(len(raw) * 0.15)
        raw.loc[raw.index[:n], "customer_id"] = "UNKNOWN"
        passed, detail = V.check_unattributed_revenue(raw, "customer_id", "order_value", 0.05)
        self.assertFalse(passed)
        self.assertGreater(detail["unattributed_row_pct"], 0.05)

    # 8
    def test_near_duplicate_orders_fails_on_format_variants(self):
        raw = self.raw.copy()
        # Inject whitespace/format near-duplicates of existing order ids.
        extra = []
        for oid in raw["order_id"].iloc[:50]:
            extra.append({**raw.iloc[0].to_dict(), "order_id": f"  {oid.lower()} "})
        raw = pd.concat([raw, pd.DataFrame(extra)], ignore_index=True)
        passed, detail = V.check_near_duplicate_orders(
            raw, "order_id", "order_value", "order_date", "customer_id", 0.9, 0.02
        )
        self.assertFalse(passed)
        self.assertGreater(detail["flagged_near_duplicates"], 0)

    def test_near_duplicate_orders_passes_when_clean(self):
        passed, _ = V.check_near_duplicate_orders(
            self.raw, "order_id", "order_value", "order_date", "customer_id", 0.9, 0.02
        )
        self.assertTrue(passed)

    # 9
    def test_qty_sanity_fails_on_messy_quantities(self):
        raw = self.raw.copy()
        raw["quantity"] = raw["quantity"].astype(object)  # allow mixed messy values
        n = len(raw)
        bad = int(n * 0.2)
        # mix of missing, negative, text, and fractional
        raw.loc[raw.index[:bad], "quantity"] = [
            [None, -3, "one", 2.5][i % 4] for i in range(bad)
        ]
        passed, detail = V.check_qty_sanity(raw, "quantity", 1, None, 99.5, 0.05, True)
        self.assertFalse(passed)
        self.assertGreater(detail["bad_pct"], 0.05)

    def test_qty_sanity_passes_on_clean_quantities(self):
        passed, _ = V.check_qty_sanity(self.raw, "quantity", 1, None, 99.5, 0.05, True)
        self.assertTrue(passed)


class EndToEndGateTests(unittest.TestCase):
    """run_validation should HALT (overall_pass False) on any single bad mode."""

    def test_gate_halts_on_fabricated_top_category(self):
        good = build_good()
        good["df"] = good["df"].copy()
        good["df"]["top_category"] = "Beauty"
        report = V.run_validation(**good)
        self.assertFalse(report["overall_pass"])
        self.assertIn("top_category_accuracy", report["failed"])

    def test_gate_halts_on_bijective_persona(self):
        good = build_good()
        df = good["df"].copy()
        df["persona"] = df["cluster"].map({0: "A", 1: "B", 2: "C"})
        df["action"] = df["cluster"].map({0: "a", 1: "b", 2: "c"})
        df["priority"] = df["cluster"].map({0: "p", 1: "q", 2: "r"})
        good["df"] = df
        report = V.run_validation(**good)
        self.assertFalse(report["overall_pass"])
        self.assertTrue(any(f.startswith("label_independence") for f in report["failed"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
