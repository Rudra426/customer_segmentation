# Q&A — E-Commerce Customer Segmentation Pipeline

A reference Q&A covering **how the project is built, how it scales, what it does and doesn't do, where its boundaries are, and what has to change before a real-world production deployment**. Answers are grounded in the current codebase (`app.py`, `config.py`, `src/`, `scripts/`, `tests/`).

> Scope note: this document describes the system as built — a single-tenant, file-driven, LLM-assisted segmentation tool with a Streamlit UI. Where a claim is aspirational or requires work, it is flagged **[GAP]** or **[TODO for prod]**.

---

## 1. Project overview & purpose

**Q: What does this project actually do?**
It turns a messy e-commerce order export (CSV/Excel) into labeled customer segments with recommended marketing actions, with no data team required. The flow is: ingest → LLM-assisted column mapping → clean/validate → feature engineering (RFM, AOV/CLV, tenure, category affinity) → K-Means clustering with stability-checked k-selection → LLM persona labeling → deliver via a Streamlit dashboard, a CSV export, and a natural-language chat Q&A layer.

**Q: Who is the intended user?**
A small-to-mid e-commerce store owner or marketer without an analytics team. This assumption drives most design trade-offs: plain-English output, LLM auto-mapping of arbitrary column names, and a "just upload a file" UX rather than a data warehouse integration.

**Q: What are the core building blocks?**
- `config.py` — single source of truth: fixed internal schema, `ACTION_MAP`, paths, LLM + clustering params.
- `src/schema_mapper.py` (Phase 2) — LLM maps raw columns → internal schema.
- `src/cleaner.py` (Phase 3) — type coercion, dedup, null/negative/future-date handling, entity resolution of customer IDs.
- `src/features.py` (Phase 4) — RFM, AOV/CLV proxy, tenure, top-category one-hot.
- `src/cluster.py` (Phase 5) — StandardScaler + K-Means, multi-metric stability-checked k-selection, UMAP projection, artifact persistence.
- `src/personas.py` (Phase 6) — LLM names each cluster (constrained to 5 fixed personas) + attaches actions/priority.
- `app.py` (Phase 7) — Streamlit dashboard.
- `src/chat.py` (Phase 9) — NL→pandas Q&A with a restricted exec sandbox.

**Q: What is the tech stack?**
pandas, scikit-learn, umap-learn, shap, the `openai` SDK pointed at OpenRouter (default model `meta-llama/llama-3.3-70b-instruct`), Streamlit, joblib, plotly, openpyxl, python-dotenv.

---

## 2. Architecture & how it's built

**Q: How does data flow through the system?**
Upload → `run_schema_mapping()` (LLM proposes column map) → human confirms/edits mapping → `clean_data()` → `engineer_features()` → `run_clustering()` (fit scaler + K-Means, save artifacts) → `label_customers()` (LLM personas + actions) → dashboard views / CSV export / chat.

**Q: What model artifacts are produced, and where?**
`models/scaler.joblib`, `models/kmeans.joblib`, and `models/model_metadata.json` (feature column order, metrics, `created_utc`). Paths are centralized in `config.py`.

**Q: How is the training path kept consistent so segments stay comparable?**
`model_metadata.json` stores the exact `feature_columns` order used at fit time. The same `engineer_features()` definitions — `aov = monetary/frequency` and `clv = monetary * CLV_MARGIN` — are used everywhere, so re-running the pipeline on the same input reproduces the same feature matrix. This is the mechanism that keeps feature construction consistent across runs.

**Q: How is the LLM integrated?**
All calls go through `src/llm.py`, a single OpenRouter (OpenAI-compatible) wrapper. Structured output is done by embedding the target JSON schema in the prompt, requesting JSON-object mode, then validating with pydantic and retrying up to `LLM_MAX_RETRIES` (3). Temperature defaults to 0.0 for determinism. The LLM is used in exactly three places: schema mapping, persona labeling, and chat.

**Q: Is the system deterministic?**
Partly. Clustering uses `RANDOM_STATE = 42` for KMeans/UMAP, and LLM temperature is 0. But LLM outputs are not guaranteed identical across runs or model versions, and OpenRouter may route to different backends — so persona *names* and chat code can vary. The numeric pipeline (clean → features → cluster) is reproducible given the same input.

---

## 3. Functionality & boundaries

**Q: What is the fixed internal schema?**
Required: `customer_id`, `order_id`, `order_date`, `order_value`. Optional: `quantity`, `unit_price`, `product_category`, `customer_email`, `signup_date`, `support_tickets`. Everything downstream assumes these names; the schema mapper's entire job is to coerce arbitrary uploads into this shape.

**Q: What are the segments, and are they fixed?**
Yes — five hardcoded personas in `ACTION_MAP`: **Loyal Big Spenders, At-Risk High Value, New Customers, One-Time Buyers, Low Engagement**. Each maps to a fixed action, channel, and priority. The LLM is *constrained* to choose from these names; it cannot invent new ones. This is a deliberate v1 boundary that guarantees every cluster maps cleanly to an action.

**Q: What is a hard boundary / explicitly out of scope in v1?**
- Multi-currency is **flag-and-warn only** — values are not converted to a common currency.
- `sqlalchemy` / database ingestion is skipped; input is files only.
- CLV is a **proxy** (`monetary × CLV_MARGIN`), not a predictive lifetime-value model.
- Personas are a fixed set of 5; no custom taxonomies.
- Chat answers only questions about the loaded customer table; anything else is declined as out-of-scope.
- The number of clusters `k` is auto-recommended within `K_SEARCH_RANGE = (2, 15)`.

**Q: How does the chat Q&A avoid arbitrary code execution?**
`src/chat.py` uses defense-in-depth: (1) the LLM is told to write pandas using only `df` and `pd`, no imports/IO; (2) a `_FORBIDDEN` token denylist rejects `import`, dunders, `open(`, `eval`, `os.`, `subprocess`, file/network calls, etc.; (3) `safe_exec()` runs in a sandbox exposing only a whitelist of safe builtins plus `pd`. Out-of-scope questions are declined without executing anything. **[Caveat — see §7]** a denylist sandbox is a mitigation, not a hard security guarantee.

**Q: How are results delivered to the user?**
Through the Streamlit dashboard: the validation report, cluster charts (sizes, recency-vs-monetary, UMAP), per-segment persona/action cards, a labeled-customer CSV download, and the chat Q&A panel. Everything the user sees comes from the in-memory pipeline output held in `st.session_state`.

---

## 4. Scalability

**Q: What are the current scale limits?**
The pipeline is **single-process, in-memory pandas**. Practical ceiling is a dataset that fits comfortably in one machine's RAM (roughly low-single-digit millions of order rows, fewer customers after aggregation). K-Means k-selection refits each candidate k `STABILITY_SEEDS = 8` times across `K_SEARCH_RANGE` (2–15) with `n_init=10`, plus a UMAP projection — this is the most CPU-intensive step and grows with customer count and k.

**Q: Where does it break first under growth?**
1. **Memory** — the whole order table and feature matrix live in RAM; Streamlit also holds copies in `session_state`.
2. **k-selection compute** — ~14 candidate k × 8 seeds × n_init=10 KMeans fits + pairwise ARI + UMAP. This is the wall-clock bottleneck on larger customer bases.
3. **LLM latency/cost** — schema mapping and persona labeling are network round-trips; chat is one to two LLM calls per question.
4. **Streamlit concurrency** — one Python process per session; not built for many simultaneous users.

**Q: How would you scale it for production?**
- Move ingestion off single-file uploads to a warehouse/object-store source (the `sqlalchemy` path deliberately skipped in v1).
- Push aggregation (RFM per customer) into SQL / a distributed engine so pandas only sees the already-reduced per-customer table.
- Cache/parallelize k-selection (it's embarrassingly parallel across seeds and k) or narrow the search range once a stable k is known.
- Separate the **training job** (batch, scheduled) from interactive use so the heavy fit isn't re-run per session.
- Replace Streamlit for multi-user use, or run it per-tenant.

**Q: How does retraining scale over time?**
Retraining is currently manual — re-run the pipeline on a fresh export to regenerate the artifacts in `models/`. There is no automated drift detection or scheduled retrain job in v1; deciding *when* to retrain is left to the operator. A production setup would add a monitored retrain trigger (e.g. scheduled batch + data/segmentation-quality checks). **[TODO for prod]**

---

## 5. Limitations

**Q: What are the modeling limitations?**
- **K-Means assumptions** — spherical, similarly-sized clusters in scaled Euclidean space; struggles with elongated/varying-density segments. `MIN_CLUSTER_PCT = 0.03` disqualifies fragmented k, and `MIN_STABILITY_ARI = 0.7` guards against unstable k, but the method itself is still K-Means.
- **CLV is a crude proxy** (`monetary × margin`), not survival/predictive modeling.
- **Category affinity** uses only the *top* category one-hot, not full basket composition.
- **Fixed 5 personas** — a store whose reality needs 7 segments is forced into 5.
- **Cold start** — a brand-new customer with one order lands in whatever cluster their sparse RFM matches; there's no dedicated new-customer model beyond the persona bucket.

**Q: What are the data limitations?**
- Multi-currency is not normalized — mixed-currency stores get warnings, not correct math.
- Entity resolution of `customer_id` variants is heuristic (`canonical_customer_id`), so it can under- or over-merge identities.
- Required-field nulls, negatives, zeros, duplicates, and future-dated orders are dropped — aggressive cleaning can discard meaningful rows if the source is unusual.
- Everything hinges on the schema mapping being correct; a wrong-but-plausible column map (e.g. tax mapped to `order_value`) silently corrupts every downstream number.

**Q: What are the operational limitations?**
- LLM dependence for two pipeline steps means an outage or model change can block mapping/labeling (chat degrades gracefully; mapping/labeling do not).
- Cost and latency scale with LLM calls; there is no batching or caching of LLM responses across uploads.
- Artifacts are local files — no model registry, versioning, or rollback beyond the filesystem.
- No authentication anywhere (Streamlit). **[GAP]**

**Q: What testing exists / doesn't?**
Unit tests cover clustering k-selection contract (`tests/test_cluster.py`) and the revenue view (`tests/test_revenue.py`). There is **no** end-to-end integration test of the full upload→export flow, and no load tests. **[TODO for prod]**

---

## 6. Real-world deployment — readiness Q&A

**Q: Can I deploy this as-is to production?**
For a single trusted internal user doing periodic batch segmentation, roughly yes. For a real multi-user, internet-facing product, **no** — the gaps in auth, secrets handling, multi-tenancy, and observability below must be closed first.

**Q: How are secrets handled, and is that production-safe?**
`config._get_secret()` reads from environment first, then falls back to `st.secrets` (Streamlit Cloud). The OpenRouter key is loaded from `.env` locally. For production: use a real secrets manager (Vault, AWS/GCP Secrets Manager), never bake keys into images, and scope the key. **[TODO for prod]**

**Q: What's the deployment topology?**
Two deployable concerns, best separated:
1. **Batch/training** — runs the pipeline, writes artifacts to shared storage (object store, not a local disk).
2. **Dashboard** — Streamlit, ideally per-tenant or behind SSO; the heaviest and least multi-user-friendly component.

**Q: Is it multi-tenant?**
No. Artifact paths in `config.py` are global singletons (`models/kmeans.joblib`, etc.), so two tenants would overwrite each other's models. Multi-tenancy needs per-tenant artifact namespacing and per-tenant data isolation. **[GAP]**

**Q: What observability is needed before go-live?**
Structured logging of every pipeline stage and LLM call (latency, tokens, cost, retries), and alerting on pipeline failures. Today there's console output only. **[TODO for prod]**

**Q: What about data privacy / PII?**
Uploads contain `customer_email` and IDs. Raw files are written to `data/raw/`, cleaned intermediates to `data/processed/`, and column samples are sent to a third-party LLM (OpenRouter) during schema mapping and (aggregate stats) during persona labeling. For production you must: get consent/legal basis, minimize what's sent to the LLM, consider redacting PII before LLM calls, set retention/deletion policies for `data/`, and confirm OpenRouter's data-handling terms meet your compliance needs (GDPR/CCPA). **[TODO for prod]**

**Q: What could silently produce wrong results in the wild?**
- A plausible-but-wrong LLM column mapping (biggest risk — it poisons everything downstream).
- Mixed currencies treated as one.
- Faulty `customer_id` entity resolution inflating/deflating frequency and monetary.
- An LLM persona label that misrepresents a cluster's real behavior.
Because these fail *quietly*, production needs guardrails: mapping confirmation UX, currency checks that block rather than warn for money math, and sanity checks on segment profiles. *(Note: a dedicated segmentation QA gate previously existed and was removed; some of that validation intent would need to return for production confidence.)*

**Q: What's the minimal production hardening checklist?**
1. AuthN/AuthZ on Streamlit.
2. Secrets in a manager, not `.env`.
3. Per-tenant artifact + data isolation.
4. Artifact storage in an object store + model versioning/rollback.
5. Structured logging, metrics, alerting on pipeline failures.
6. PII minimization + data retention/deletion policy; validate LLM-provider data terms.
7. Rate limiting + cost caps on LLM calls; timeouts and fallbacks for LLM outages.
8. End-to-end + load tests in CI.
9. Re-introduce automated output validation (sanity checks on segments/features) before promoting a model.
10. Reproducible, monitored retrain job with a defined trigger and rollback.

**Q: How do we roll out a retrained model safely?**
Write new artifacts to a *new* versioned location, run validation checks against a holdout, then flip the dashboard to the new version and keep the previous version for instant rollback. Never overwrite `models/*.joblib` in place while the app is live — the current single-path design does exactly that and must change. **[TODO for prod]**

---

## 7. Security Q&A

**Q: How safe is the chat sandbox, really?**
It's a **denylist + restricted-builtins** sandbox (`_FORBIDDEN` tokens, `_SAFE_BUILTINS`, only `df`/`pd` in scope). This blocks the obvious escapes (imports, dunders, file/network/system access) and is good defense-in-depth, but token-denylist sandboxes are historically bypassable and should not be treated as a hard security boundary against a determined adversary. For production, run generated code in a genuinely isolated environment (separate process/container with no network, seccomp, resource limits) or replace codegen with a constrained query builder. **[TODO for prod]**

**Q: What's exposed to the third-party LLM?**
Schema mapping sends column names + a few sample values per column; persona labeling sends aggregate cluster profiles; chat sends the schema summary + a small sample of rows. Sample rows can include PII. Minimize and/or redact before sending in production.

**Q: Any injection surface via the uploaded file?**
The LLM sees raw column names and sample cell values, so prompt-injection via crafted cell contents is a theoretical vector for the mapping/chat steps. Mitigations: keep temperature 0, validate all LLM output against pydantic schemas (already done), and constrain outputs to enums where possible (personas already are).

---

## 8. Quick-reference: key configuration knobs

| Knob | Location | Default | Effect |
|------|----------|---------|--------|
| `K_SEARCH_RANGE` | `config.py` | `(2, 15)` | Candidate cluster counts searched |
| `MIN_CLUSTER_PCT` | `config.py` | `0.03` | Disqualifies k with a too-small cluster |
| `STABILITY_SEEDS` | `config.py` | `8` | KMeans refits per k for stability/avg metrics |
| `MIN_STABILITY_ARI` | `config.py` | `0.7` | Min stability for a recommended k |
| `RANDOM_STATE` | `config.py` | `42` | Repeatable KMeans/UMAP |
| `LLM_MAX_RETRIES` | `config.py` | `3` | Retries on LLM/JSON/validation failure |
| `OPENROUTER_MODEL` | `config.py`/env | `llama-3.3-70b-instruct` | LLM used for map/label/chat |

---

*Generated as a deployment-readiness reference. Items marked **[GAP]** / **[TODO for prod]** are the shortest path from "works on my machine" to "safe in production."*
