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
| Deliver | Streamlit dashboard · CSV export · Chat Q&A |

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
├── outputs/             # labeled CSVs
├── src/
│   ├── llm.py           # OpenRouter (Llama-3.3-70B) client wrapper
│   ├── schema_mapper.py # Phase 2
│   ├── cleaner.py       # Phase 3
│   ├── features.py      # Phase 4
│   ├── cluster.py       # Phase 5
│   ├── personas.py      # Phase 6
│   ├── api.py           # Phase 8
│   └── chat.py          # Phase 9
├── scripts/
│   └── generate_sample_data.py
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

pandas · scikit-learn · openai (OpenRouter) · streamlit · joblib ·
umap-learn · shap · openpyxl · plotly · python-dotenv
