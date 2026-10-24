"""TransactionGuard — advanced local Streamlit fraud-review interface."""

from __future__ import annotations

import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transactionguard.inference import InferenceError, InferenceService

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
FRAUD, LEGIT = "Potential Fraud", "Likely Legitimate"
PALETTE = {FRAUD: "#F43F5E", LEGIT: "#10B981"}
RISK_COLORS = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#F43F5E"}
ACCENT = "#8B5CF6"

PAGES = {
    "📊  Overview": "overview",
    "📂  Bulk Analysis": "bulk",
    "🔍  Single Transaction": "single",
    "📈  Model Performance": "performance",
    "ℹ️  About": "about",
}

st.set_page_config(page_title="TransactionGuard", page_icon="🛡️", layout="wide")


# --------------------------------------------------------------------------- #
# Styling helpers
# --------------------------------------------------------------------------- #
def inject_style() -> None:
    st.markdown(
        """<style>
        #MainMenu, footer {visibility:hidden;}
        .stApp {background: radial-gradient(1200px 600px at 10% -10%, #1b2250 0%, #0B1020 55%);
                color:#F8FAFC;}
        .block-container {padding-top:1.6rem; max-width:1300px;}
        [data-testid="stSidebar"] {background:#0e1424; border-right:1px solid #1f2a44;}
        .tg-brand {font-size:1.35rem; font-weight:800; letter-spacing:.3px;}
        .tg-brand span {color:#8B5CF6;}
        .tg-hero {background:linear-gradient(135deg,#312e81 0%,#6d28d9 55%,#be185d 120%);
                  border-radius:18px; padding:26px 30px; margin-bottom:18px;
                  box-shadow:0 10px 30px rgba(109,40,217,.25);}
        .tg-hero h1 {margin:0; font-size:2rem;}
        .tg-hero p {margin:.35rem 0 0 0; color:#E0E7FF; font-size:.98rem;}
        .tg-card {background:#141B2D; border:1px solid #283552; border-radius:14px;
                  padding:16px 18px; height:100%;
                  transition:transform .15s ease, border-color .15s ease;}
        .tg-card:hover {transform:translateY(-2px); border-color:#8B5CF6;}
        .tg-label {color:#94A3B8; font-size:.8rem; text-transform:uppercase;
                   letter-spacing:.06em; margin-bottom:.25rem;}
        .tg-value {font-size:1.8rem; font-weight:800; line-height:1.15;}
        .tg-sub {color:#94A3B8; font-size:.8rem; margin-top:.2rem;}
        .tg-pill {display:inline-block; padding:3px 12px; border-radius:999px;
                  font-size:.78rem; font-weight:700; border:1px solid;}
        .tg-verdict {border-radius:14px; padding:18px 22px; font-size:1.3rem; font-weight:800;}
        .tg-muted {color:#94A3B8;}
        div[data-testid="stTabs"] button {font-weight:600;}
        div.stButton > button[kind="primary"], div.stDownloadButton > button {border-radius:10px;}
        </style>""",
        unsafe_allow_html=True,
    )


def kpi(label: str, value, sub: str = "", color: str = "#F8FAFC") -> str:
    return (
        f'<div class="tg-card"><div class="tg-label">{label}</div>'
        f'<div class="tg-value" style="color:{color}">{value}</div>'
        f'<div class="tg-sub">{sub}</div></div>'
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="tg-hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, title: str, height: int = 340) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=15)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_service() -> InferenceService:
    return InferenceService(ROOT / "artifacts").load()


def read_upload(uploaded) -> pd.DataFrame:
    if uploaded.size > MAX_UPLOAD_BYTES:
        raise InferenceError("The upload exceeds the 25 MB safety limit.")
    suffix = Path(uploaded.name).suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(uploaded)
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            raise InferenceError("The CSV file could not be read.") from exc
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(uploaded)
        except (ImportError, OSError, ValueError) as exc:
            raise InferenceError("The Excel workbook could not be read.") from exc
    raise InferenceError("Upload a CSV, XLSX, or XLS file.")


def template_bytes(features: list[str]) -> bytes:
    return pd.DataFrame(columns=features).to_csv(index=False).encode("utf-8")


def export_excel(frame: pd.DataFrame) -> bytes:
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False, sheet_name="predictions")
    except ImportError as exc:
        raise InferenceError("Excel export requires the openpyxl package.") from exc
    return output.getvalue()


def risk_pill(risk: str) -> str:
    color = RISK_COLORS.get(str(risk), "#94A3B8")
    return f'<span class="tg-pill" style="color:{color};border-color:{color}">{risk} risk</span>'


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def gauge(probability: float, threshold: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "valueformat": ".2f"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#F8FAFC", "thickness": 0.25},
                "steps": [
                    {"range": [0, 20], "color": "#065f46"},
                    {"range": [20, 60], "color": "#92400e"},
                    {"range": [60, 100], "color": "#9f1239"},
                ],
                "threshold": {"line": {"color": ACCENT, "width": 4}, "value": min(threshold * 100, 100)},
            },
        )
    )
    return style_fig(fig, "Fraud probability (purple line = decision threshold)", 300)


def result_charts(results: pd.DataFrame, threshold: float) -> None:
    left, right = st.columns(2)
    with left:
        counts = results["Predicted Class"].value_counts().rename_axis("Prediction").reset_index(name="Count")
        fig = px.pie(counts, names="Prediction", values="Count", hole=0.6,
                     color="Prediction", color_discrete_map=PALETTE)
        st.plotly_chart(style_fig(fig, "Prediction outcomes"), use_container_width=True)
    with right:
        if "Risk Category" in results:
            risk = results["Risk Category"].value_counts().reindex(["Low", "Medium", "High"]).dropna()
            fig = px.bar(risk.rename_axis("Risk").reset_index(name="Count"), x="Risk", y="Count",
                         color="Risk", color_discrete_map=RISK_COLORS)
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig, "Risk bands"), use_container_width=True)

    fig = px.histogram(results, x="Fraud Probability", nbins=40, color="Predicted Class",
                       color_discrete_map=PALETTE, barmode="overlay", opacity=0.8)
    fig.add_vline(x=threshold, line_dash="dash", line_color=ACCENT,
                  annotation_text="threshold", annotation_position="top")
    fig.update_yaxes(type="log", title="Count (log)")
    st.plotly_chart(style_fig(fig, "Predicted fraud probability distribution"), use_container_width=True)

    if "Amount" in results:
        a, b = st.columns(2)
        with a:
            fig = px.box(results, x="Predicted Class", y="Amount", color="Predicted Class",
                         color_discrete_map=PALETTE, points=False)
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig, "Amount by prediction"), use_container_width=True)
        with b:
            if "Time" in results:
                sample = results.sample(min(len(results), 5000), random_state=0)
                fig = px.scatter(sample, x="Time", y="Amount", color="Predicted Class",
                                 color_discrete_map=PALETTE, opacity=0.7, hover_data=["Fraud Probability"])
                st.plotly_chart(style_fig(fig, "Time vs amount (sampled to 5,000 rows)"), use_container_width=True)


def results_table(results: pd.DataFrame, key: str) -> pd.DataFrame:
    """Filterable results table. Returns the filtered frame."""
    f1, f2, f3 = st.columns([2, 2, 2])
    classes = f1.multiselect("Class", [FRAUD, LEGIT], default=[FRAUD, LEGIT], key=f"{key}_cls")
    min_prob = f2.slider("Min fraud probability (%)", 0.0, 100.0, 0.0, 0.5, key=f"{key}_prob")
    sort_desc = f3.selectbox("Sort", ["Highest risk first", "Original order"], key=f"{key}_sort")
    view = results[results["Predicted Class"].isin(classes) & (results["Fraud Probability"] * 100 >= min_prob)]
    if sort_desc == "Highest risk first":
        view = view.sort_values("Fraud Probability", ascending=False)
    shown = view.copy()
    shown["Fraud Probability"] = shown["Fraud Probability"] * 100
    st.caption(f"Showing {len(view):,} of {len(results):,} rows")
    st.dataframe(
        shown,
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config={
            "Fraud Probability": st.column_config.ProgressColumn(
                "Fraud Probability (%)", min_value=0, max_value=100, format="%.2f"
            )
        },
    )
    return view


def download_row(frame: pd.DataFrame, key: str, label_suffix: str = "") -> None:
    a, b = st.columns(2)
    a.download_button(f"⬇ Download CSV{label_suffix}", frame.to_csv(index=False).encode(),
                      "transactionguard_results.csv", "text/csv", key=f"{key}_csv", use_container_width=True)
    try:
        b.download_button(f"⬇ Download Excel{label_suffix}", export_excel(frame),
                          "transactionguard_results.xlsx", key=f"{key}_xlsx", use_container_width=True)
    except InferenceError as exc:
        b.warning(str(exc))


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def overview(service: InferenceService) -> None:
    hero("🛡️ TransactionGuard", "Local fraud-screening estimates for analyst review — never a confirmation of fraud.")
    results = st.session_state.get("results")
    threshold = service.details.threshold

    if results is None:
        c = st.columns(3)
        c[0].markdown(kpi("Model", service.details.name, f"version {service.details.version}"), unsafe_allow_html=True)
        c[1].markdown(kpi("Decision threshold", f"{threshold:.4%}", "probability cut-off"), unsafe_allow_html=True)
        c[2].markdown(kpi("Input features", len(service.details.feature_names), "numeric fields"), unsafe_allow_html=True)
        st.info("No transactions analyzed yet. Open **Bulk Analysis** to upload a file or **Single Transaction** to test one row.")
        return

    total = len(results)
    flagged = int((results["Predicted Class"] == FRAUD).sum())
    meta = st.session_state.get("results_meta", {})
    cols = st.columns(5)
    cols[0].markdown(kpi("Processed", f"{total:,}", meta.get("source", "")), unsafe_allow_html=True)
    cols[1].markdown(kpi("Flagged", f"{flagged:,}", "for analyst review", "#F43F5E"), unsafe_allow_html=True)
    cols[2].markdown(kpi("Likely legit", f"{total - flagged:,}", "", "#10B981"), unsafe_allow_html=True)
    cols[3].markdown(kpi("Flag rate", f"{flagged / total:.2%}", ""), unsafe_allow_html=True)
    cols[4].markdown(kpi("Max probability", f"{results['Fraud Probability'].max():.1%}", "highest single score", ACCENT), unsafe_allow_html=True)
    st.write("")

    tab_charts, tab_top, tab_whatif = st.tabs(["Charts", "Top risky transactions", "What-if threshold"])
    with tab_charts:
        result_charts(results, threshold)
    with tab_top:
        top = results.nlargest(15, "Fraud Probability").copy()
        top["Fraud Probability"] = top["Fraud Probability"] * 100
        st.dataframe(top, use_container_width=True, hide_index=True,
                     column_config={"Fraud Probability": st.column_config.ProgressColumn(
                         "Fraud Probability (%)", min_value=0, max_value=100, format="%.2f")})
    with tab_whatif:
        st.caption("Exploration only — this does NOT change the saved model threshold or predictions.")
        what_if = st.slider("Hypothetical threshold (%)", 0.0, 100.0, float(min(threshold * 100, 100.0)), 0.01)
        n_flag = int((results["Fraud Probability"] >= what_if / 100).sum())
        a, b, c = st.columns(3)
        a.metric("Flagged at hypothetical threshold", f"{n_flag:,}", delta=f"{n_flag - flagged:+,} vs model")
        b.metric("Flag rate", f"{n_flag / total:.2%}")
        c.metric("Model threshold", f"{threshold:.4%}")


def bulk_analysis(service: InferenceService) -> None:
    hero("📂 Bulk Analysis", "Upload a CSV or Excel file. Files stay in this browser session and are not persisted.")
    top_l, top_r = st.columns([3, 1])
    with top_r:
        st.download_button("⬇ Input template (CSV)", template_bytes(service.details.feature_names),
                           "transactionguard_template.csv", "text/csv", use_container_width=True)
    with top_l:
        uploaded = st.file_uploader("Upload transactions", type=["csv", "xlsx", "xls"], label_visibility="collapsed")

    if uploaded:
        try:
            frame = read_upload(uploaded)
        except InferenceError as exc:
            st.error(str(exc))
            return
        if frame.columns.duplicated().any():
            st.error("Duplicate column names are not supported. Rename them and upload again.")
            return

        _, errors = service.validate(frame)
        extra = [c for c in frame.columns if c not in service.details.feature_names and c != "Class"]

        cols = st.columns(4)
        cols[0].markdown(kpi("Rows", f"{len(frame):,}"), unsafe_allow_html=True)
        cols[1].markdown(kpi("Columns", len(frame.columns)), unsafe_allow_html=True)
        cols[2].markdown(kpi("Missing cells", f"{int(frame.isna().sum().sum()):,}"), unsafe_allow_html=True)
        status, color = ("Valid", "#10B981") if not errors else (f"{len(errors)} issue(s)", "#F43F5E")
        cols[3].markdown(kpi("Validation", status, "", color), unsafe_allow_html=True)
        st.write("")

        with st.expander("Preview (first 10 rows)", expanded=False):
            st.dataframe(frame.head(10), use_container_width=True)
        if extra:
            st.info(f"Extra columns are preserved but not sent to the model: {', '.join(extra[:8])}")
        if "Class" in frame:
            st.info("`Class` is treated as optional evaluation information and excluded from prediction inputs.")
        if errors:
            for error in errors[:8]:
                st.error(error)
            return

        if st.button("🚀 Analyze transactions", type="primary"):
            with st.spinner("Applying the saved model and feature engineering pipeline…"):
                try:
                    predictions = service.predict(frame)
                except InferenceError as exc:
                    st.error(str(exc))
                    return
            st.session_state.results = pd.concat(
                [frame.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
            st.session_state.results_meta = {
                "source": uploaded.name,
                "time": datetime.now().strftime("%d %b %Y, %H:%M"),
            }
            st.toast("Analysis complete", icon="✅")

    results = st.session_state.get("results")
    if results is None:
        return
    meta = st.session_state.get("results_meta", {})
    st.subheader("Results")
    st.caption(f"Source: {meta.get('source', '—')} · analyzed {meta.get('time', '')}")
    tab_table, tab_charts = st.tabs(["Table", "Charts"])
    with tab_table:
        view = results_table(results, "bulk")
        download_row(view, "bulk", " (filtered)")
    with tab_charts:
        result_charts(results, service.details.threshold)


def render_prediction(service: InferenceService, values: dict[str, float]) -> None:
    try:
        prediction = service.predict(pd.DataFrame([values])).iloc[0]
    except InferenceError as exc:
        st.error(str(exc))
        return
    probability = float(prediction["Fraud Probability"])
    label = str(prediction["Predicted Class"])
    risk = str(prediction["Risk Category"])
    color = PALETTE.get(label, "#94A3B8")
    st.markdown(
        f'<div class="tg-verdict" style="background:{color}22;border:1px solid {color};color:{color}">'
        f'{"⚠️" if label == FRAUD else "✅"} {label} &nbsp; {risk_pill(risk)}</div>',
        unsafe_allow_html=True,
    )
    g, m = st.columns([3, 2])
    g.plotly_chart(gauge(probability, service.details.threshold), use_container_width=True)
    with m:
        st.metric("Fraud probability", f"{probability:.2%}")
        st.metric("Risk category", risk)
        st.metric("Decision threshold", f"{service.details.threshold:.4%}")
    st.warning("This is a model estimate for triage, not a guarantee or confirmation of fraud.")

    history = st.session_state.setdefault("single_history", [])
    history.insert(0, {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Amount": values.get("Amount"),
        "Probability %": round(probability * 100, 3),
        "Class": label,
        "Risk": risk,
    })
    del history[25:]


def single_transaction(service: InferenceService) -> None:
    hero("🔍 Single Transaction", "V1–V28 are anonymized PCA features and cannot be derived from customer details.")
    features = service.details.feature_names
    tab_manual, tab_paste = st.tabs(["Manual entry", "Paste a row"])

    with tab_manual:
        with st.form("single_transaction"):
            values: dict[str, float] = {}
            basics, components = st.columns([1, 2])
            with basics:
                st.markdown("**Core fields**")
                for name in [x for x in features if x in {"Time", "Amount"}]:
                    help_text = "Seconds elapsed from the dataset reference point." if name == "Time" else "Transaction amount."
                    values[name] = st.number_input(name, value=0.0, format="%.8f", help=help_text)
            with components:
                st.markdown("**Anonymized components**")
                grid = st.columns(3)
                for i, name in enumerate([x for x in features if x not in {"Time", "Amount"}]):
                    values[name] = grid[i % 3].number_input(name, value=0.0, format="%.8f")
            submitted = st.form_submit_button("Analyze transaction", type="primary")
        if submitted:
            render_prediction(service, {k: values[k] for k in features})

    with tab_paste:
        st.caption(f"Paste {len(features)} comma-separated numbers in this order:")
        st.code(", ".join(features), language=None)
        raw = st.text_area("Row values", height=110, placeholder="0, -1.35, -0.07, ...")
        if st.button("Analyze pasted row", type="primary"):
            try:
                nums = [float(x) for x in raw.replace("\n", ",").split(",") if x.strip()]
            except ValueError:
                st.error("All values must be numeric.")
            else:
                if len(nums) != len(features):
                    st.error(f"Expected {len(features)} values, got {len(nums)}.")
                else:
                    render_prediction(service, dict(zip(features, nums)))

    history = st.session_state.get("single_history")
    if history:
        with st.expander(f"Session history ({len(history)})"):
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


def performance(service: InferenceService) -> None:
    hero("📈 Model Performance", "Evaluation on a held-out test set.")
    st.warning("These are synthetic-data evaluation results, not live application performance or validation on real financial transactions.")
    metrics_path = ROOT / "reports" / "test_metrics.json"
    if not metrics_path.exists():
        st.info("No evaluation report found. Run `python scripts/evaluate.py` after training.")
        return
    metrics = pd.read_json(metrics_path, typ="series")
    labels = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "precision": "Precision", "recall": "Recall",
              "f1": "F1", "fpr": "False-positive rate", "fnr": "False-negative rate"}
    present = [k for k in labels if k in metrics]
    for row in (present[:4], present[4:]):
        cols = st.columns(max(len(row), 1))
        for col, key in zip(cols, row):
            col.markdown(kpi(labels[key], f"{float(metrics[key]):.4f}"), unsafe_allow_html=True)
        st.write("")
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure(go.Bar(x=[float(metrics[k]) for k in present], y=[labels[k] for k in present],
                               orientation="h", marker_color=ACCENT))
        fig.update_xaxes(range=[0, 1])
        st.plotly_chart(style_fig(fig, "Metric overview", 380), use_container_width=True)
    with right:
        image = ROOT / "reports" / "confusion_matrix.png"
        if image.exists():
            st.image(str(image), caption="Confusion matrix (synthetic held-out test set)")
    st.caption(f"Model: {service.details.name} · threshold: {service.details.threshold:.4%}")


def about(service: InferenceService) -> None:
    hero("ℹ️ About / Model Information", "What the model expects and how results are presented.")
    c = st.columns(3)
    c[0].markdown(kpi("Model", service.details.name), unsafe_allow_html=True)
    c[1].markdown(kpi("Version", service.details.version), unsafe_allow_html=True)
    c[2].markdown(kpi("Features", len(service.details.feature_names), "numeric fields"), unsafe_allow_html=True)
    st.write("")
    with st.expander("Required input features", expanded=True):
        st.code(", ".join(service.details.feature_names), language=None)
    st.markdown("**Risk bands** (presentation only — they do not change the saved classification threshold):")
    st.markdown(f"{risk_pill('Low')} &lt;20% &nbsp; {risk_pill('Medium')} 20–&lt;60% &nbsp; {risk_pill('High')} ≥60%",
                unsafe_allow_html=True)
    st.write("")
    if st.button("Clear session results"):
        for key in ("results", "results_meta", "single_history"):
            st.session_state.pop(key, None)
        st.success("Session results cleared.")


def main() -> None:
    inject_style()
    try:
        service = get_service()
    except InferenceError as exc:
        st.error(f"Model unavailable: {exc}")
        st.stop()

    st.sidebar.markdown('<div class="tg-brand">🛡️ Transaction<span>Guard</span></div>', unsafe_allow_html=True)
    st.sidebar.caption(f"{service.details.name} · v{service.details.version}")
    page = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.sidebar.divider()
    results = st.session_state.get("results")
    if results is not None:
        st.sidebar.metric("Session rows analyzed", f"{len(results):,}")
    st.sidebar.caption("Runs locally. Uploaded transaction data is not stored.")

    {
        "overview": overview, "bulk": bulk_analysis, "single": single_transaction,
        "performance": performance, "about": about,
    }[PAGES[page]](service)


if __name__ == "__main__":
    main()