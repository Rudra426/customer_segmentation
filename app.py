"""
app.py — Streamlit dashboard for the customer-segmentation pipeline.

Flow (built across Phase 7 steps):
  7.1 upload            -> load file, run schema mapping
  7.2 schema confirm    -> review/edit the column mapping
  7.3 validation report -> clean data + show report
  7.4 cluster charts    -> features + clustering + plots
  7.5 action recs       -> personas + per-segment actions
  7.6 CSV download      -> export labeled customers

Run:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Make project imports work when launched via `streamlit run`.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import AT_RISK_PATTERNS, RAW_DIR, has_api_key  # noqa: E402
from src.cleaner import clean_data  # noqa: E402
from src.cluster import run_clustering, save_diagnostic_plot, select_optimal_k  # noqa: E402
from src.features import engineer_features  # noqa: E402
from src.chat import EXAMPLE_PROMPTS, answer_question  # noqa: E402
from src.personas import label_customers  # noqa: E402
from src.revenue import (  # noqa: E402
    compute_clv_at_risk,
    compute_revenue_concentration,
    format_currency,
    format_percent,
)
from src.schema_mapper import finalize_mapping, run_schema_mapping  # noqa: E402

st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide",
)


def _init_state() -> None:
    """Initialize session_state keys used across the flow."""
    defaults = {
        "uploaded_name": None,  # name of the file currently loaded
        "mapping_session": None,  #1 output of run_schema_mapping()
        "normalized_df": None,  # schema-normalized DataFrame (Phase 2 output)
        "clean_df": None,  # cleaned DataFrame (Phase 3)
        "report": None,  # validation report
        "k_selection": None,  # select_optimal_k() result (table + recommended_k)
        "k_plot_path": None,  # saved diagnostic plot artifact path
        "cluster_out": None,  # run_clustering() output
        "labeled": None,  # label_customers() output
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)


def _reset_downstream() -> None:
    """Clear cached results when a new file is uploaded."""
    for key in (
        "mapping_session", "normalized_df", "clean_df", "report",
        "k_selection", "k_plot_path", "cluster_out", "labeled",
    ):
        st.session_state[key] = None


def render_schema_confirm(session: dict) -> None:
    """Step 7.2 - show validation status and let the user confirm/edit mapping."""
    report = session["report"]
    status = report["status"]

    st.subheader("2 - Confirm column mapping")

    if status == "rejected":
        st.error("This file does not look like e-commerce order data.")
        for msg in report["messages"]:
            st.write("- " + msg)
        for tip in report["suggestions"]:
            st.info(tip)
        st.stop()

    if status == "needs_confirmation":
        st.warning("Some columns need your confirmation before continuing.")
        for msg in report["messages"]:
            st.write("- " + msg)
    else:
        st.success("All required fields mapped with high confidence.")

    # Editable mapping: one dropdown per raw column, defaulting to the proposal.
    choices = session["field_choices"]
    proposals = {m["raw_column"]: m for m in session["mappings"]}
    overrides: dict[str, str] = {}

    st.write("Map each uploaded column to an internal field (or `none` to ignore):")
    for raw, proposed in session["editable_mapping"].items():
        conf = proposals.get(raw, {}).get("confidence", 0.0)
        c1, c2 = st.columns([3, 2])
        with c1:
            picked = st.selectbox(
                f"`{raw}`",
                options=choices,
                index=choices.index(proposed) if proposed in choices else len(choices) - 1,
                key=f"map_{raw}",
            )
        with c2:
            st.caption(f"LLM proposal: **{proposed}** (confidence {conf})")
        overrides[raw] = picked

    if st.button("Confirm mapping and continue", type="primary"):
        result = finalize_mapping(session["raw_df"], session["mappings"], overrides)
        if result["status"] == "rejected":
            st.error("With these choices, required fields are still missing.")
            for msg in result["report"]["messages"]:
                st.write("- " + msg)
        else:
            st.session_state["normalized_df"] = result["normalized_df"]
            # Invalidate downstream so they recompute with the new mapping.
            for key in ("clean_df", "report", "cluster_out", "labeled"):
                st.session_state[key] = None
            st.success(
                f"Mapping confirmed - {result['normalized_df'].shape[1]} fields, "
                f"{result['normalized_df'].shape[0]} rows."
            )


def render_validation(normalized_df) -> None:
    """Step 7.3 - clean the data and render the validation report."""
    st.subheader("3 - Data validation")

    if st.session_state["clean_df"] is None:
        with st.spinner("Cleaning and validating..."):
            clean_df, report = clean_data(normalized_df)
            st.session_state["clean_df"] = clean_df
            st.session_state["report"] = report

    report = st.session_state["report"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows in", report["rows_in"])
    c2.metric("Rows kept", report["rows_out"], delta=-report["rows_dropped_total"])
    c3.metric("Customers", report["n_customers"])
    c4.metric("Orders", report["n_orders"])

    for w in report["warnings"]:
        st.warning(w)
    if report["currency"]["warning"]:
        st.warning(report["currency"]["warning"])

    with st.expander("Cleaning details"):
        drops = report["drops"]
        st.write(
            f"- **Null required fields removed:** {drops['null_required']['rows_dropped']} "
            f"({drops['null_required']['per_field']})"
        )
        st.write(
            f"- **Duplicate orders removed:** {drops['duplicate_orders']['duplicates_removed']}"
        )
        st.write(
            f"- **Negative/zero values removed:** {drops['nonpositive_values']['rows_dropped']} "
            f"(neg {drops['nonpositive_values']['negative_removed']}, "
            f"zero {drops['nonpositive_values']['zero_removed']})"
        )
        st.write(f"- **Future-dated orders removed:** {drops['future_orders']}")
        id_info = report.get("id_normalization", {})
        if id_info.get("merged_groups"):
            st.write(
                f"- **Customer ids unified:** {id_info['ids_before']} raw ids → "
                f"{id_info['ids_after']} customers "
                f"({id_info['merged_groups']} format group(s) merged)"
            )
            for ex in id_info.get("examples", []):
                st.write(f"    - {ex['variants']} → `{ex['merged_to']}`")
        if report["date_range"]:
            dr = report["date_range"]
            st.write(f"- **Date range:** {dr['min']} to {dr['max']} (as of {dr['reference']})")
        if report["optional_field_nulls"]:
            st.write(f"- **Optional-field nulls:** {report['optional_field_nulls']}")

    if report["rows_out"] == 0:
        st.error("No rows survived cleaning - cannot segment. Check the source data.")
        st.stop()


def render_clustering(clean_df) -> None:
    """Step 7.4 - run clustering on demand and show metrics + charts."""
    st.subheader("4 - Customer segments")

    if st.session_state["cluster_out"] is None:
        if not st.button("Segment customers", type="primary"):
            st.info("Click to engineer features and cluster customers.")
            return
        with st.spinner("Engineering features and clustering..."):
            features = engineer_features(clean_df)
            st.session_state["cluster_out"] = run_clustering(features, save=True)
            # New clustering -> recompute personas (7.5).
            st.session_state["labeled"] = None

    out = st.session_state["cluster_out"]
    result = out["result"]
    metrics = out["metrics"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Segments (k)", metrics["k"])
    c2.metric("Silhouette", metrics["silhouette"], help="Higher is better (-1 to 1)")
    c3.metric("Davies-Bouldin", metrics["davies_bouldin"], help="Lower is better")

    # Color by cluster id (rendered as a category, not a number).
    plot_df = result.copy()
    plot_df["cluster"] = plot_df["cluster"].astype(str)

    # 1) Segment sizes
    sizes = plot_df["cluster"].value_counts().sort_index().reset_index()
    sizes.columns = ["cluster", "customers"]
    st.plotly_chart(
        px.bar(sizes, x="cluster", y="customers", color="cluster",
               title="Segment sizes"),
        use_container_width=True,
    )

    col_a, col_b = st.columns(2)
    # 2) Recency vs Monetary
    with col_a:
        st.plotly_chart(
            px.scatter(
                plot_df.reset_index(), x="recency", y="monetary", color="cluster",
                hover_name="customer_id", title="Recency vs Monetary",
            ),
            use_container_width=True,
        )
    # 3) UMAP projection
    with col_b:
        st.plotly_chart(
            px.scatter(
                plot_df.reset_index(), x="umap_x", y="umap_y", color="cluster",
                hover_name="customer_id", title="UMAP projection",
            ),
            use_container_width=True,
        )


def render_actions(cluster_out) -> None:
    """Step 7.5 - label segments with personas and show recommended actions."""
    st.subheader("5 - Recommended actions per segment")

    if st.session_state["labeled"] is None:
        with st.spinner("Naming segments and assigning actions with the LLM..."):
            st.session_state["labeled"] = label_customers(cluster_out["result"])

    segments = st.session_state["labeled"]["segments"]

    for seg in segments:
        with st.container(border=True):
            top = st.columns([3, 1, 1])
            top[0].markdown(f"### {seg['persona']}")
            top[1].metric("Customers", seg["size"])
            top[2].metric("Share", f"{seg['share_pct']}%")
            st.markdown(
                f"**Action:** {seg['action']}  \n"
                f"**Channel:** {seg['channel']} | **Priority:** "
                f"`{seg['priority']}` (score {seg['priority_score']})"
            )
            avg = seg["averages"]
            st.caption(
                f"Avg - recency {avg.get('recency')}d, "
                f"frequency {avg.get('frequency')}, "
                f"monetary ${avg.get('monetary')}, AOV ${avg.get('aov')}"
            )
            if seg.get("reasoning"):
                st.caption(seg["reasoning"])


def render_revenue_impact(df) -> None:
    """Revenue Impact view: concentration (Pareto) + CLV-at-risk over segments."""
    st.subheader("Revenue Impact")

    conc = compute_revenue_concentration(df, "persona", "monetary")
    at_risk = compute_clv_at_risk(df, "persona", "clv", AT_RISK_PATTERNS)

    if conc.empty:
        st.info("No positive revenue values available to analyze.")
        return

    # Headline insight from the top revenue-concentrated segment.
    top = conc.iloc[0]
    st.markdown(
        f"## {top['segment']} = {format_percent(top['pct_of_customers'])} of "
        f"customers but {format_percent(top['pct_of_revenue'])} of revenue"
    )

    # KPI row.
    total_revenue = conc.attrs["total_revenue"]
    total_customers = conc.attrs["total_customers"]
    avg_rev = total_revenue / total_customers if total_customers else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total revenue", format_currency(total_revenue))
    k2.metric(
        "CLV at risk",
        format_currency(at_risk["total_clv_at_risk"]),
        delta=f"{format_percent(at_risk['pct_of_total_clv_at_risk'])} of CLV",
        delta_color="inverse",
    )
    k3.metric(
        "Customers at risk",
        f"{at_risk['customer_count_at_risk']:,}",
        delta=f"{format_percent(at_risk['pct_of_total_customers_at_risk'])} of base",
        delta_color="inverse",
    )
    k4.metric("Avg revenue / customer", format_currency(avg_rev))

    # At-risk callout, auto-generated.
    if at_risk["any_at_risk"]:
        st.error(
            f"{format_currency(at_risk['total_clv_at_risk'])} "
            f"({format_percent(at_risk['pct_of_total_clv_at_risk'])} of total CLV) "
            f"sits in at-risk segments across {at_risk['customer_count_at_risk']:,} "
            f"customers: {', '.join(at_risk['matched_segments'])}."
        )
    else:
        st.info("No at-risk segment found among the current personas.")

    # Grouped bar: % of customers vs % of revenue per segment.
    long = conc.melt(
        id_vars="segment",
        value_vars=["pct_of_customers", "pct_of_revenue"],
        var_name="metric",
        value_name="pct",
    )
    long["metric"] = long["metric"].map(
        {"pct_of_customers": "% of customers", "pct_of_revenue": "% of revenue"}
    )
    st.plotly_chart(
        px.bar(
            long, x="segment", y="pct", color="metric", barmode="group",
            title="Customer share vs revenue share by segment",
            labels={"pct": "Percent", "segment": "Segment"},
        ),
        use_container_width=True,
    )

    # Formatted table (no raw floats).
    display = pd.DataFrame({
        "Segment": conc["segment"],
        "Customers": conc["customer_count"].map(lambda v: f"{int(v):,}"),
        "% of customers": conc["pct_of_customers"].map(format_percent),
        "Total revenue": conc["total_revenue"].map(format_currency),
        "% of revenue": conc["pct_of_revenue"].map(format_percent),
        "Avg revenue / customer": conc["avg_revenue_per_customer"].map(format_currency),
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

    excluded = conc.attrs["excluded_count"]
    if excluded:
        st.caption(f"{excluded} customer(s) excluded from revenue math (missing/zero spend).")


def render_download() -> None:
    """Step 7.6 - export the labeled customer table as CSV."""
    st.subheader("6 - Export")

    labeled = st.session_state["labeled"]["labeled"].reset_index()
    csv_bytes = labeled.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download labeled customers (CSV)",
        data=csv_bytes,
        file_name="segmented_customers.csv",
        mime="text/csv",
        type="primary",
    )
    st.caption(f"{len(labeled)} customers, {labeled.shape[1]} columns.")
    with st.expander("Preview labeled table"):
        st.dataframe(labeled, use_container_width=True)


def _run_chat(df, question: str) -> None:
    """Run one Q&A turn and append it to the chat history."""
    st.session_state["chat_history"].append({"role": "user", "content": question})
    with st.spinner("Thinking..."):
        answer = answer_question(df, question)
    st.session_state["chat_history"].append({"role": "assistant", "answer": answer})


def render_chat() -> None:
    """Step 9.1 - chat box + example prompts over the labeled table."""
    st.subheader("7 - Ask questions about your customers")
    st.session_state.setdefault("chat_history", [])

    df = st.session_state["labeled"]["labeled"].reset_index()

    # Example prompt suggestions.
    st.caption("Try an example:")
    cols = st.columns(len(EXAMPLE_PROMPTS))
    for col, prompt in zip(cols, EXAMPLE_PROMPTS):
        if col.button(prompt, key=f"ex_{prompt}"):
            _run_chat(df, prompt)

    # Replay history.
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            ans = msg["answer"]
            with st.chat_message("assistant"):
                if ans.get("error"):
                    st.error(ans["error"])
                elif ans.get("out_of_scope"):
                    st.info(ans["explanation"])
                else:
                    st.write(ans["explanation"])
                    if ans.get("result") is not None:
                        st.write(ans["result"])
                if ans.get("code"):
                    with st.expander("Show pandas code"):
                        st.code(ans["code"], language="python")

    # Free-text input.
    question = st.chat_input("Ask about your segments, e.g. 'how many VIPs?'")
    if question:
        _run_chat(df, question)
        st.rerun()


def main() -> None:
    _init_state()

    st.title("E-Commerce Customer Segmentation")
    st.caption(
        "Upload a raw CSV/Excel export, auto-map columns, clean, segment "
        "customers, and get recommended actions."
    )

    if not has_api_key():
        st.warning(
            "No OpenRouter API key found. Copy `.env.template` to `.env` and set "
            "`OPENROUTER_API_KEY` before analyzing data."
        )

    # Sidebar: upload
    with st.sidebar:
        st.header("1 - Upload data")
        uploaded = st.file_uploader(
            "CSV or Excel export",
            type=["csv", "tsv", "txt", "xlsx", "xls"],
            help="Shopify, WooCommerce, or any custom export.",
        )

    if uploaded is None:
        st.info("Upload a file in the sidebar to begin.")
        return

    # Persist the upload to data/raw and (re)run schema mapping once per file.
    if st.session_state["uploaded_name"] != uploaded.name:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        dest = RAW_DIR / uploaded.name
        dest.write_bytes(uploaded.getbuffer())
        st.session_state["uploaded_name"] = uploaded.name
        _reset_downstream()
        with st.spinner("Reading file and auto-mapping columns with the LLM..."):
            st.session_state["mapping_session"] = run_schema_mapping(dest)

    session = st.session_state["mapping_session"]
    raw_df = session["raw_df"]

    st.subheader("Raw data preview")
    st.write(f"**{uploaded.name}** - {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
    st.dataframe(raw_df.head(10), use_container_width=True)

    # Step 7.2: schema confirmation
    render_schema_confirm(session)

    # Step 7.3: validation report (only after mapping confirmed)
    if st.session_state["normalized_df"] is None:
        st.info("Confirm the column mapping above to continue.")
        return
    render_validation(st.session_state["normalized_df"])

    # Step 7.4: clustering + charts
    render_clustering(st.session_state["clean_df"])

    if st.session_state["cluster_out"] is None:
        return

    # Step 7.5: persona labeling + action recommendations
    render_actions(st.session_state["cluster_out"])

    # Revenue Impact view (concentration + CLV-at-risk)
    if st.session_state["labeled"] is not None:
        render_revenue_impact(st.session_state["labeled"]["labeled"].reset_index())

    # Step 7.6: CSV export
    if st.session_state["labeled"] is not None:
        render_download()

    # Step 9: chat Q&A over the labeled table
    if st.session_state["labeled"] is not None:
        render_chat()


if __name__ == "__main__":
    main()