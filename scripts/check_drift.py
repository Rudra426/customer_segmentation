"""
check_drift.py — Phase 10.5: scheduler entry point for drift detection.

A standalone, cron/Airflow-friendly CLI that takes a freshly-exported data file,
runs it through the same ingest → clean → feature pipeline used at train time,
then calls drift.run_drift_check() against the saved training baseline.

The retraining signal is surfaced as a PROCESS EXIT CODE so any scheduler can
branch on it without parsing stdout:

    0  checked, no significant drift          -> nothing to do
    2  checked, RETRAINING RECOMMENDED        -> kick off a retrain
    1  could not run (bad file, no baseline)  -> alert/investigate

Usage:
    python scripts/check_drift.py data/raw/new_orders.csv
    python scripts/check_drift.py new_orders.csv --mapping mapping.json
    python scripts/check_drift.py new_orders.csv --json

--mapping points at a JSON file of {raw_column: internal_field} so the run is
fully deterministic and needs no LLM/API key (recommended for unattended jobs,
since the source export format is stable). Without it, the LLM schema mapper is
used to auto-map columns, exactly as the dashboard does on upload.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/check_drift.py ...` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import DRIFT_LOG_PATH
from src.cleaner import clean_data
from src.drift import BaselineNotFound, run_drift_check
from src.features import engineer_features, feature_matrix
from src.schema_mapper import (
    apply_mapping,
    finalize_mapping,
    load_raw_file,
    run_schema_mapping,
)

# Exit codes consumed by cron / Airflow (see module docstring).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_RETRAIN = 2


def _normalize(path: Path, mapping_path: Path | None) -> pd.DataFrame:
    """
    Turn a raw export into a schema-normalized DataFrame.

    With `mapping_path`, applies a fixed {raw_column: internal_field} mapping
    (deterministic, no LLM). Otherwise runs the LLM auto-mapper and accepts the
    result unless it is outright rejected.
    """
    if mapping_path is not None:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        return apply_mapping(load_raw_file(path), mapping)

    mapped = run_schema_mapping(path)
    final = finalize_mapping(mapped["raw_df"], mapped["mappings"])
    if final["normalized_df"] is None:
        raise ValueError(
            "Schema mapping was rejected for this file; supply --mapping with an "
            "explicit column map. Mapper said: " + final["report"].get("status", "?")
        )
    return final["normalized_df"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a drift check on a new data export (scheduler entry point)."
    )
    parser.add_argument("data_file", help="Path to the new CSV/Excel export.")
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="JSON file of {raw_column: internal_field} for deterministic, "
        "LLM-free mapping (recommended for unattended jobs).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Do not append to outputs/drift.log (still prints to stdout).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full drift report as JSON instead of a summary line.",
    )
    args = parser.parse_args(argv)

    data_path = Path(args.data_file)
    if not data_path.exists():
        print(f"ERROR: file not found: {data_path}", file=sys.stderr)
        return EXIT_ERROR

    try:
        normalized = _normalize(data_path, args.mapping)
        clean_df, _ = clean_data(normalized)
        if clean_df.empty:
            print("ERROR: no rows survived cleaning; cannot check drift.",
                  file=sys.stderr)
            return EXIT_ERROR
        features = feature_matrix(engineer_features(clean_df))
        report = run_drift_check(features, log=not args.no_log)
    except BaselineNotFound as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except (ValueError, KeyError, OSError) as exc:
        print(f"ERROR: drift check failed: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        status = "RETRAIN RECOMMENDED" if report["retrain_recommended"] else "OK"
        print(f"[{status}] rows={report['n_rows_checked']} "
              f"drifted_features={report['ks']['n_drifted']} "
              f"silhouette_degraded={report['silhouette']['degraded']}")
        for reason in report["reasons"]:
            print("  -", reason)
        print(f"(log: {DRIFT_LOG_PATH})")

    return EXIT_RETRAIN if report["retrain_recommended"] else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
