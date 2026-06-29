# 🛍️ E-Commerce Customer Segmentation Pipeline

An automated, no-data-team-required customer segmentation tool for small-to-mid
e-commerce stores. Upload any messy CSV/Excel export (Shopify, WooCommerce,
custom) and get back labeled customer segments with recommended marketing actions.

**LLM provider:** OpenRouter · **Default model:** `meta-llama/llama-3.3-70b-instruct`

> ⚠️ Status: under active development — see [`PROGRESS.md`](PROGRESS.md) for what's built.

---

## ✨ What it does

| Stage | Capability |
|-------|------------|
| Ingest | Accepts any messy CSV/Excel export |
| Map | LLM (Llama-3.3-70B via OpenRouter) auto-maps raw column names → fixed internal schema |
| Clean | Validates, dedupes, type-coerces, reports issues |
| Engineer | Builds RFM, AOV/CLV, category-affinity features |
| Cluster | K-Means with auto-`k` (silhouette), UMAP visualization |
| Label | LLM names each segment + maps to recommended actions |
| Deliver | Streamlit dashboard · CSV export · FastAPI scoring · Chat Q&A |

---

## 🚀 Setup

```bash
# 1. Create & activate the virtual environment
python -m venv .venv
source .venv/Scripts/activate        # Git Bash on Windows
# .venv\Scripts\Activate.ps1         # PowerShell
# .venv\Scripts\activate.bat         # cmd

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.template .env
# then edit .env and paste your OpenRouter key (https://openrouter.ai/keys)
```

---

## ▶️ Running (interfaces — added in later phases)

```bash
# Streamlit dashboard (Phase 7)
streamlit run app.py

# FastAPI real-time scoring endpoint (Phase 8)
uvicorn src.api:app --reload
```

---

## 🔁 Scheduled drift detection (Phase 10)

Once a model is trained, `scripts/check_drift.py` compares a fresh data export
against the training baseline (saved in `models/model_metadata.json`) and signals
whether a retrain is warranted. It runs the same ingest → clean → feature
pipeline used at train time, then a per-feature KS-test plus a silhouette-
degradation check.

```bash
# Deterministic, no API key needed (recommended for unattended jobs):
python scripts/check_drift.py data/raw/new_orders.csv --mapping mapping.json

# Or let the LLM auto-map columns (needs OPENROUTER_API_KEY):
python scripts/check_drift.py data/raw/new_orders.csv
```

`mapping.json` is a fixed `{raw_column: internal_field}` map, e.g.:

```json
{ "Cust ID": "customer_id", "Order #": "order_id", "Order Date": "order_date",
  "Total $": "order_value", "SKU Category": "product_category", "Qty": "quantity" }
```

The retraining signal is the **process exit code**, so any scheduler can branch
on it without parsing output:

| Exit | Meaning | Scheduler action |
|------|---------|------------------|
| `0`  | no significant drift | nothing to do |
| `2`  | retraining recommended | trigger a retrain |
| `1`  | error (bad file / no baseline) | alert / investigate |

Every run also appends a timestamped line to `outputs/drift.log`.

### cron (Linux/macOS)

Run daily at 02:30 and retrain when drift is flagged (exit code `2`):

```cron
30 2 * * * cd /path/to/proctor && .venv/bin/python scripts/check_drift.py \
  data/raw/new_orders.csv --mapping mapping.json >> outputs/drift.log 2>&1 \
  || [ $? -eq 2 ] && python scripts/retrain.py   # wire up your own retrain step
```

On Windows, schedule the equivalent with Task Scheduler calling
`.venv\Scripts\python.exe scripts\check_drift.py ...`.

### Airflow

`check_drift.py` exits non-zero on "retrain", which `BashOperator` treats as
failure — so use `BranchPythonOperator` (which inspects the code) to fan out to a
retrain task instead:

```python
from airflow import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import pendulum, subprocess

PROJECT = "/path/to/proctor"

def _check_drift():
    code = subprocess.run(
        [f"{PROJECT}/.venv/bin/python", "scripts/check_drift.py",
         "data/raw/new_orders.csv", "--mapping", "mapping.json"],
        cwd=PROJECT,
    ).returncode
    if code == 2:
        return "retrain"          # drift detected
    if code == 0:
        return "no_op"            # all good
    raise RuntimeError(f"drift check failed (exit {code})")

with DAG(
    "segmentation_drift",
    schedule="30 2 * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
) as dag:
    check = BranchPythonOperator(task_id="check_drift", python_callable=_check_drift)
    retrain = EmptyOperator(task_id="retrain")   # replace with your retrain task
    no_op = EmptyOperator(task_id="no_op")
    check >> [retrain, no_op]
```

---

## 🗂️ Project structure

```
proctor/
├── config.py            # schema, ACTION_MAP, paths, model + clustering params
├── app.py               # Streamlit dashboard (Phase 7)
├── requirements.txt
├── .env.template        # copy to .env and add your key
├── data/
│   ├── raw/             # uploaded input files
│   └── processed/       # cleaned intermediates
├── models/              # joblib artifacts (scaler, kmeans, metadata)
├── outputs/             # labeled CSVs, drift logs
├── src/
│   ├── llm.py           # OpenRouter (Llama-3.3-70B) client wrapper
│   ├── schema_mapper.py # Phase 2
│   ├── cleaner.py       # Phase 3
│   ├── features.py      # Phase 4
│   ├── cluster.py       # Phase 5
│   ├── personas.py      # Phase 6
│   ├── api.py           # Phase 8
│   ├── chat.py          # Phase 9
│   └── drift.py         # Phase 10
├── scripts/
│   ├── generate_sample_data.py
│   └── check_drift.py   # Phase 10.5 — scheduler entry point (cron/Airflow)
└── tests/
```

---

## 🔒 Internal schema

**Required:** `customer_id`, `order_id`, `order_date`, `order_value`
**Optional:** `quantity`, `product_category`, `customer_email`, `signup_date`, `support_tickets`

---

## 🧩 Segments & actions

The pipeline assigns each customer to one of five personas, each mapped to a
recommended action, channel, and priority (see `ACTION_MAP` in `config.py`):
**Loyal Big Spenders · At-Risk High Value · New Customers · One-Time Buyers · Low Engagement**

---

## 📈 Tech stack

pandas · scikit-learn · openai (OpenRouter) · streamlit · fastapi · uvicorn · joblib ·
umap-learn · shap · openpyxl · plotly · python-dotenv
