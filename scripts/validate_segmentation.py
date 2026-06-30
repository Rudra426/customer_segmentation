"""
validate_segmentation.py — Phase 6.5 gate: run the segmentation pipeline and
HALT (non-zero exit) before persona-labeling / dashboard / API steps are
trusted, if any QA check fails.

This is the pipeline ORCHESTRATOR for unattended / CI use. It runs the same
ingest -> clean -> feature -> cluster (-> personas) path the dashboard uses,
then calls validation.run_validation() and turns the verdict into a PROCESS
EXIT CODE so a scheduler / CI job can stop and require human sign-off:

    0  all checks passed            -> safe to promote outputs
    1  a check FAILED, or the run could not complete (bad file, etc.)

Personas (LLM step) are included only when an API key is configured and
--no-personas is not passed; without them the label-independence checks are
simply skipped (they are not applicable yet).

Usage:
    python scripts/validate_segmentation.py data/raw/orders.xlsx --mapping mapping.json
    python scripts/validate_segmentation.py orders.csv --no-personas --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from config import VALIDATION_REPORT_PATH, has_api_key  # noqa: E402
from src.cleaner import clean_data  # noqa: E402
from src.cluster import (  # noqa: E402
    run_clustering,
    save_diagnostic_plot,
    select_optimal_k,
)
from src.features import engineer_features, feature_matrix  # noqa: E402
from src.schema_mapper import (  # noqa: E402
    apply_mapping,
    finalize_mapping,
    load_raw_file,
    run_schema_mapping,
)
from src.validation import prepare_validation_inputs, run_validation  # noqa: E402

EXIT_OK = 0
EXIT_FAIL = 1


def _print_k_table(selection: dict) -> None:
    """Print the full multi-metric k-selection diagnostic table + recommendation."""
    print("\n" + "=" * 62)
    print("K-SELECTION DIAGNOSTICS (silhouette + Davies-Bouldin + stability)")
    print("=" * 62)
    print(selection["table"].to_string(index=False))
    print("-" * 62)
    print(f"Recommended k = {selection['recommended_k']}  "
          f"({selection['recommendation_basis']})")
    print(f"Disqualified k (cluster < {selection['min_cluster_pct'] * 100:.0f}% of customers): "
          f"{selection['table'].loc[selection['table']['disqualified'], 'k'].tolist() or 'none'}")
    print("=" * 62)


def _confirm_k(selection: dict, args) -> int | None:
    """
    Human confirmation gate for the chosen k (consistent with the approval-gate
    pattern). Returns the confirmed k, or None to ABORT the run.

    Resolution order:
      --k K        -> use K (explicit override, no prompt)
      --yes        -> accept the recommended k (non-interactive / CI)
      interactive  -> prompt y / n / <int-override>
      non-tty      -> abort (cannot obtain sign-off unattended; pass --yes or --k)
    """
    rec = int(selection["recommended_k"])
    k_lo, k_hi = selection["k_range"]
    valid = set(range(k_lo, k_hi + 1))

    if args.k is not None:
        if args.k not in valid:
            print(f"ERROR: --k {args.k} outside tested range [{k_lo}, {k_hi}].",
                  file=sys.stderr)
            return None
        print(f"Using forced k={args.k} (--k override).")
        return int(args.k)

    if args.yes:
        print(f"Auto-accepting recommended k={rec} (--yes).")
        return rec

    if not sys.stdin.isatty():
        print("ERROR: human confirmation required before fitting the final model. "
              "Re-run interactively, or pass --yes to accept the recommended k, "
              "or --k N to override.", file=sys.stderr)
        return None

    while True:
        resp = input(
            f"Proceed with recommended k={rec}? "
            f"[y]es / [n]o-abort / integer {k_lo}-{k_hi} to override: "
        ).strip().lower()
        if resp in ("", "y", "yes"):
            return rec
        if resp in ("n", "no"):
            return None
        if resp.lstrip("-").isdigit() and int(resp) in valid:
            return int(resp)
        print(f"  Please answer y, n, or an integer k in [{k_lo}, {k_hi}].")


def _normalize(path: Path, mapping_path: Path | None) -> pd.DataFrame:
    """Raw export -> schema-normalized DataFrame (explicit mapping or LLM)."""
    if mapping_path is not None:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        return apply_mapping(load_raw_file(path), mapping)
    mapped = run_schema_mapping(path)
    final = finalize_mapping(mapped["raw_df"], mapped["mappings"])
    if final["normalized_df"] is None:
        raise ValueError(
            "Schema mapping rejected; supply --mapping with an explicit column map."
        )
    return final["normalized_df"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a segmentation run; non-zero exit halts the pipeline."
    )
    parser.add_argument("data_file", help="Path to the CSV/Excel order export.")
    parser.add_argument("--mapping", type=Path, default=None,
                        help="JSON {raw_column: internal_field} for LLM-free mapping.")
    parser.add_argument("--no-personas", action="store_true",
                        help="Skip the LLM persona step (label-independence checks skipped).")
    parser.add_argument("--k", type=int, default=None,
                        help="Force this k (override the recommendation; no prompt).")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Auto-accept the recommended k without prompting (CI use).")
    parser.add_argument("--output", type=Path, default=VALIDATION_REPORT_PATH,
                        help=f"Where to write the report (default {VALIDATION_REPORT_PATH}).")
    parser.add_argument("--json", action="store_true",
                        help="Print the full report JSON instead of just the summary.")
    args = parser.parse_args(argv)

    data_path = Path(args.data_file)
    if not data_path.exists():
        print(f"ERROR: file not found: {data_path}", file=sys.stderr)
        return EXIT_FAIL

    try:
        # 1) ingest + clean
        raw_normalized = _normalize(data_path, args.mapping)
        clean_df, _ = clean_data(raw_normalized)
        if clean_df.empty:
            print("ERROR: no rows survived cleaning; cannot validate.", file=sys.stderr)
            return EXIT_FAIL

        # 2) features + k-selection (multi-metric, stability-checked)
        features = engineer_features(clean_df)
        matrix = feature_matrix(features)
        # Recompute X independently (don't trust the stored scaler) for both the
        # k-search and the silhouette QA check downstream.
        X = StandardScaler().fit_transform(matrix.to_numpy())
        selection = select_optimal_k(X)
        plot_path = save_diagnostic_plot(selection["table"], selection["recommended_k"])
        _print_k_table(selection)
        print(f"Diagnostic plot saved: {plot_path}")

        # 2b) human confirmation gate BEFORE fitting the final model
        confirmed_k = _confirm_k(selection, args)
        if confirmed_k is None:
            print("ABORTED: k not confirmed; no final model fit.", file=sys.stderr)
            return EXIT_FAIL

        # 2c) clean final fit on the FULL dataset with the confirmed k
        cluster_out = run_clustering(
            features, save=False, k_override=confirmed_k,
            selection=selection, make_plot=False,
        )
        cluster_result = cluster_out["result"]
        labels = cluster_result["cluster"].to_numpy()
        stored_sil = cluster_out["metrics"].get("silhouette")

        # 3) personas (only if we can): produces persona/action/priority columns
        labeled_df = None
        if not args.no_personas and has_api_key():
            from src.personas import label_customers
            labeled_df = label_customers(cluster_result, save_map=False)["labeled"]
        else:
            print("(personas skipped — label-independence checks will be skipped)")

        # 4) assemble inputs and RUN THE GATE
        kwargs = prepare_validation_inputs(
            clean_df=clean_df,
            cluster_result=cluster_result,
            X=X,
            cluster_labels=labels,
            raw_orders_df=raw_normalized,  # ORIGINAL id formats: needed for checks 6-9
            labeled_df=labeled_df,
            stored_silhouette=stored_sil,
        )
        # Record the k-selection diagnostics in validation_report.json too.
        k_section = {
            "k_selection": {
                "recommended_k": int(selection["recommended_k"]),
                "confirmed_k": int(confirmed_k),
                "recommendation_basis": selection["recommendation_basis"],
                "k_range": selection["k_range"],
                "n_seeds": selection["n_seeds"],
                "min_cluster_pct": selection["min_cluster_pct"],
                "min_stability_ari": selection["min_stability_ari"],
                "diagnostic_plot": str(plot_path),
                "table": selection["records"],
            }
        }
        report = run_validation(report_path=args.output, extra_report=k_section, **kwargs)

    except (ValueError, KeyError, OSError) as exc:
        print(f"ERROR: validation run failed: {exc}", file=sys.stderr)
        return EXIT_FAIL

    if args.json:
        print(json.dumps(report, indent=2, default=str))

    return EXIT_OK if report["overall_pass"] else EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
