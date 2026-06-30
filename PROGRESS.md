# 📊 Project Progress — E-Commerce Customer Segmentation Pipeline

**LLM provider:** OpenRouter · **Model:** `meta-llama/llama-3.3-70b-instruct`
**Python:** 3.13.3 · **Last updated:** Phase 10 COMPLETE — all phases done 🎉

Legend: ✅ done · 🟡 in progress · ⬜ not started

---

## Phase 1 — Project Setup  ✅
- [x] 1.1 Folder structure
- [x] 1.2 Virtual environment (`.venv`)
*- [x] 1.3 requirements.txt + i+nstall (all deps verified on Py 3.13)
- [x] 1.4 `.env.template` + `.gitignore` (+ `.env` with real key, gitignored)
- [x] 1.5 `config.py` (schema, ACTION_MAP, paths, clustering params)
- [x] 1.6 README skeleton36

## Phase 2 — LLM Schema Mapper  ✅
- [x] 2.1 File loader (CSV/Excel, delimiter/encoding sniff) + synthetic sample fixture
- [x] 2.2 Column profiler (names + sample values)
- [x] 2.3 LLM mapping call (JSON + pydantic validation) + `src/llm.py` wrapper
- [x] 2.4 Response parser + validation (ok / needs_confirmation / rejected + user guidance)
- [x] 2.5 Edge-case handling (dedup, user overrides, date-format detection)
- [x] 2.6 Apply rename → normalized DataFrame
- [x] 2.7 Human-confirm hook (`run_schema_mapping` + `finalize_mapping` for UI)

## Phase 3 — Data Cleaning & Validation  ✅
- [x] 3.1 Type coercion (money/date/int/string + `parse_money`)
- [x] 3.2 Required-field null handling
- [x] 3.3 Duplicate order_id removal
- [x] 3.4 Negative/zero order_value handling
- [x] 3.5 Date standardization (tz-naive + future-order removal)
- [x] 3.6 Multi-currency check (flag-only v1)
- [x] 3.7 Validation report dict + `clean_data()` orchestrator

## Phase 4 — Feature Engineering  ✅
- [x] 4.1 Recency
- [x] 4.2 Frequency
- [x] 4.3 Monetary
- [x] 4.4 AOV / CLV (proxy = monetary × margin)
- [x] 4.5 Tenure (signup_date or first-order fallback)
- [x] 4.6 Top category (label + one-hot, graceful if absent)
- [x] 4.7 Assemble numeric matrix (`engineer_features` + `feature_matrix`)

## Phase 5 — Clustering & Validation  ✅
- [x] 5.1 StandardScaler
- [x] 5.2 K-Means k-selection — **multi-metric + stability-checked** (see Phase 5.2★ below)
- [x] 5.3 Davies-Bouldin score (+ metrics dict)
- [x] 5.4 UMAP 2D projection (n_neighbors clamped to sample size)
- [x] 5.5 Persist artifacts (joblib + JSON metadata, drift baseline) + `run_clustering()`

## Phase 5.2★ — Multi-metric, stability-checked k-selection  ✅
Replaced the old fixed `K_MIN..K_MAX` silhouette-only selection.
- [x] `select_optimal_k(X_scaled, k_range, n_seeds, min_cluster_pct, min_stability_ari)`
  in `src/cluster.py` — per k: `n_seeds` KMeans refits (distinct seeds, n_init=10);
  avg silhouette, avg Davies-Bouldin, avg **pairwise Adjusted Rand Index** (stability),
  smallest-cluster % under the modal/medoid labeling; disqualifies any k whose smallest
  cluster < `MIN_CLUSTER_PCT`. Returns a DataFrame
  `[k, avg_silhouette, avg_davies_bouldin, avg_stability_ari, min_cluster_pct, disqualified]`
  + `recommended_k` (highest avg_silhouette among non-disqualified k with avg ARI >
  `MIN_STABILITY_ARI`, with graceful fallbacks).
- [x] `save_diagnostic_plot()` — 3 stacked subplots (silhouette / Davies-Bouldin /
  smallest-cluster %) vs k on a shared x-axis, recommended k marked with a vertical line,
  disqualified k shaded; saved to `outputs/k_selection_diagnostic.png` (headless Agg).
- [x] `config.py` — `K_SEARCH_RANGE=(2,15)`, `MIN_CLUSTER_PCT=0.03`, `STABILITY_SEEDS=8`,
  `MIN_STABILITY_ARI=0.7` (old `K_MIN`/`K_MAX` removed).
- [x] Pipeline integration in `scripts/validate_segmentation.py` — prints the full
  diagnostic table + recommended_k to console, writes the k-selection block into
  `outputs/validation_report.json`, then a human approval gate (`y` / `n` / integer
  override) before the final fit — consistent with the existing approval-gate pattern
  (`--yes`/`--k` for CI; aborts on non-tty without sign-off).
- [x] `fit_final_kmeans()` — clean re-fit on the FULL dataset with the confirmed k
  (fresh KMeans, n_init=10; not a reused stability-seed model).
- [x] `tests/test_cluster.py` — synthetic fixture with 3 well-separated blobs recovers
  k=3 despite the range going to 15; plus table-contract, tiny-cluster disqualification,
  full-data final-fit, and too-few-rows tests (5 tests, all green).

## Phase 6 — LLM Persona Labeling + Action Map  ✅
- [x] 6.1 Cluster profiling
- [x] 6.2 LLM persona call (constrained to ACTION_MAP keys)
- [x] 6.3 Map to actions (ACTION_MAP join)
- [x] 6.4 Priority score (numeric weight from PRIORITY_SCORE)
- [x] 6.5 Attach action columns + `label_customers()` orchestrator

## Phase 7 — Streamlit Dashboard  ✅
- [x] 7.1 Upload UI (+ session_state, schema-mapping on upload)
- [x] 7.2 Schema confirm (editable dropdowns + reject/needs-confirm guidance)
- [x] 7.3 Validation report view (metrics + drop breakdown)
- [x] 7.4 Cluster charts (sizes bar, recency-vs-monetary, UMAP — Plotly)
- [x] 7.5 Action recommendations (per-segment cards, LLM personas)
- [x] 7.6 CSV download (labeled customers export)

## Phase 8 — FastAPI Real-Time Scoring  ✅
- [x] 8.1 Artifact loader (+ persist cluster→persona segment_map.json)
- [x] 8.2 Request/response schemas (Pydantic, derived fields + validation)
- [x] 8.3 POST /score (feature-vector build + predict + segment lookup)
- [x] 8.4 Health + error handling (GET /health, GET /, graceful untrained state)

## Phase 9 — Chat Q&A Layer  ✅
- [x] 9.1 Chat UI + example prompts
- [x] 9.2 NL → pandas (LLM writes `result = ...` snippet)
- [x] 9.3 Safe execution sandbox (token denylist + whitelisted builtins)
- [x] 9.4 Explain result in plain English

## Phase 10 — Drift Detection & Retraining Hook  ✅
- [x] 10.1 Baseline capture (loader; baseline persisted at train time)
- [x] 10.2 KS-test per feature (new vs training sample)
- [x] 10.3 Silhouette degradation check (new vs baseline silhouette)
- [x] 10.4 Scheduler hook (`run_drift_check`: warn + log to drift.log)
- [x] 10.5 Cron/Airflow wiring (`scripts/check_drift.py` CLI w/ exit-code signal + README docs)

## Phase 6.5 — Segmentation QA Gate  ✅
- [x] `src/validation.py` — 9 independent checks (cluster separation, silhouette,
  label independence, category consistency, top-category accuracy, frequency
  completeness, unattributed revenue, near-dupe orders, qty sanity); each returns
  `(pass, detail)`, recomputes ground truth from source, never auto-fixes.
- [x] `run_validation()` aggregator → `outputs/validation_report.json` + console summary.
- [x] Thresholds in `config.VALIDATION` (no hardcoding); `prepare_validation_inputs()` helper.
- [x] `scripts/validate_segmentation.py` — CLI orchestrator, exit code 1 on any failure (CI/cron).
- [x] Dashboard wiring — `render_segmentation_gate()` HALTS app.py between clustering
  (7.4) and persona labeling (7.5); blocks actions/export/chat until pass or human sign-off.
- [x] `tests/test_validation.py` — good fixture (passes all) + one bad fixture per failure mode (19 tests).

## Extensions (post-pipeline)
- [x] Revenue Impact view — `src/revenue.py` (`compute_revenue_concentration`,
  `compute_clv_at_risk`) + dashboard section + `tests/test_revenue.py`

---

### Key decisions locked in
- LLM = **Llama-3.3-70B** via **OpenRouter** (`openai` SDK, OpenAI-compatible).
- Chat layer = custom "LLM-writes-pandas + safe-exec" (no `pandasai`).
- `shap` used for cluster explanations; `sqlalchemy` skipped in v1.
- Multi-currency = flag/warn only in v1.
- Sample data = 6 messy columns incl. `Order #` so all required fields exist.
- Personas constrained to the 5 fixed `ACTION_MAP` keys.

---

### 🔚 Last task — continue where we left
**Done (2026-06-30):** Phase 5.2★ multi-metric, stability-checked k-selection. The
clustering module (`src/cluster.py`), `config.py`, the diagnostic plot, and the CLI
approval gate (`scripts/validate_segmentation.py`) were already in place from the prior
session; this pass added the missing **unit-test deliverable** (`tests/test_cluster.py`)
and verified the whole thing end-to-end — `select_optimal_k` recovers k=3 on 3
well-separated blobs (silhouette 0.93) despite `K_SEARCH_RANGE` going to 15, the
diagnostic PNG renders (~88 KB), and the full suite is **28/28 green**.

**Next up (not started):** none required for this task. Possible follow-ups if desired —
wire the same console-style k-selection diagnostic/approval into the Streamlit
`render_clustering()` step (currently the dashboard auto-accepts `recommended_k`), and
add the diagnostic plot image to the dashboard.
