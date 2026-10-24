"""TransactionGuard — local fraud-review workspace (Streamlit)."""

from __future__ import annotations

import sys
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transactionguard.config import config
from transactionguard.inference import InferenceError, InferenceService

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
FRAUD, LEGIT = "Potential Fraud", "Likely Legitimate"

# One accent (navy), one alert colour (muted red), neutrals for everything else.
NAVY = "#1D3557"
RED = "#B42318"
INK = "#101828"
MUTED = "#667085"
LINE = "#E4E7EC"
PALETTE = {FRAUD: RED, LEGIT: "#9FB3C8"}
RISK_COLORS = {"Low": "#9FB3C8", "Medium": "#486581", "High": RED}

PAGES = ["Overview", "Bulk analysis", "Single transaction", "Model performance", "About"]

st.set_page_config(page_title="TransactionGuard", page_icon="🛡️", layout="wide")


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def inject_style() -> None:
    st.markdown(
        f"""<style>
        #MainMenu, footer {{visibility:hidden;}}
        header[data-testid="stHeader"] {{background:transparent;}}
        .stApp {{background:#F7F8FA; color:{INK};
                 font-family:Inter,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;}}
        .block-container {{padding-top:2.2rem; max-width:1240px;}}
        h1,h2,h3,h4 {{color:{INK}; letter-spacing:-.01em;}}

        [data-testid="stSidebar"] {{background:#FFFFFF; border-right:1px solid {LINE};}}
        .tg-brand {{font-size:1.05rem; font-weight:650; color:{INK}; letter-spacing:-.01em;}}
        .tg-brand-sub {{font-size:.78rem; color:{MUTED}; margin-top:2px;}}

        .tg-title {{font-size:1.65rem; font-weight:650; color:{INK}; margin:0; letter-spacing:-.02em;}}
        .tg-subtitle {{color:{MUTED}; font-size:.95rem; margin:.25rem 0 1.1rem 0;}}
        .tg-rule {{border:0; border-top:1px solid {LINE}; margin:0 0 1.4rem 0;}}

        .tg-strip {{display:flex; background:#fff; border:1px solid {LINE}; border-radius:8px;
                    margin-bottom:1.4rem; overflow:hidden;}}
        .tg-cell {{flex:1; padding:16px 20px; border-right:1px solid {LINE}; min-width:0;}}
        .tg-cell:last-child {{border-right:0;}}
        .tg-label {{font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; color:{MUTED}; font-weight:600;}}
        .tg-value {{font-size:1.6rem; font-weight:600; color:{INK}; margin-top:4px;
                    font-variant-numeric:tabular-nums; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}}
        .tg-sub {{font-size:.78rem; color:{MUTED}; margin-top:2px;}}

        .tg-section {{font-size:.72rem; letter-spacing:.08em; text-transform:uppercase;
                      color:{MUTED}; font-weight:600; margin:1.2rem 0 .5rem 0;}}

        .tg-callout {{background:#fff; border:1px solid {LINE}; border-left:4px solid {MUTED};
                      border-radius:6px; padding:14px 18px; margin-bottom:1rem;}}
        .tg-callout-title {{font-size:1.15rem; font-weight:650;}}
        .tg-callout-body {{color:{MUTED}; font-size:.88rem; margin-top:2px;}}

        .tg-chip {{display:inline-block; padding:1px 9px; border-radius:4px; font-size:.75rem;
                   font-weight:600; border:1px solid; vertical-align:middle;}}

        div[data-testid="stTabs"] button[role="tab"] {{font-weight:550; color:{MUTED};}}
        div[data-testid="stTabs"] button[aria-selected="true"] {{color:{INK};}}
        div.stButton > button, div.stDownloadButton > button, div.stFormSubmitButton > button {{
            border-radius:6px; font-weight:550; border:1px solid #D0D5DD;}}
        div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {{
            background:{NAVY}; border-color:{NAVY}; color:#fff;}}
        div.stButton > button[kind="primary"]:hover, div.stFormSubmitButton > button[kind="primary"]:hover {{
            background:#14263F; border-color:#14263F; color:#fff;}}
        [data-testid="stFileUploaderDropzone"] {{background:#fff; border:1px dashed #D0D5DD; border-radius:8px;}}
        [data-testid="stMetric"] {{background:#fff; border:1px solid {LINE}; border-radius:8px; padding:12px 16px;}}
        </style>""",
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="tg-title">{escape(title)}</div>'
        f'<div class="tg-subtitle">{escape(subtitle)}</div><hr class="tg-rule">',
        unsafe_allow_html=True,
    )


def strip(items: list[tuple]) -> None:
    """Single bordered metric strip. items: (label, value, sub[, colour])."""
    cells = ""
    for item in items:
        label, value, sub = item[0], item[1], item[2] if len(item) > 2 else ""
        color = item[3] if len(item) > 3 else INK
        cells += (
            f'<div class="tg-cell"><div class="tg-label">{escape(str(label))}</div>'
            f'<div class="tg-value" style="color:{color}">{escape(str(value))}</div>'
            f'<div class="tg-sub">{escape(str(sub))}</div></div>'
        )
    st.markdown(f'<div class="tg-strip">{cells}</div>', unsafe_allow_html=True)


def section(label: str) -> None:
    st.markdown(f'<div class="tg-section">{escape(label)}</div>', unsafe_allow_html=True)


def chip(text: str, color: str) -> str:
    return f'<span class="tg-chip" style="color:{color};border-color:{color}">{escape(text)}</span>'


def style_fig(fig: go.Figure, title: str = "", height: int = 320) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=13, color=INK), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, Segoe UI, Roboto, sans-serif", size=12, color="#344054"),
        height=height,
        margin=dict(l=8, r=8, t=44, b=8),
        legend=dict(orientation="h", y=-0.22, x=0, title=None),
    )
    fig.update_xaxes(gridcolor="#F2F4F7", linecolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor="#F2F4F7", linecolor=LINE, zeroline=False)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_service() -> InferenceService:
    return InferenceService(config.ARTIFACT_DIR).load()


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


def prob_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy of frame with Fraud Probability expressed as 0-100 for display only."""
    shown = frame.copy()
    shown["Fraud Probability"] = shown["Fraud Probability"] * 100
    return shown


PROB_CONFIG = {
    "Fraud Probability": st.column_config.ProgressColumn(
        "Fraud probability (%)", min_value=0, max_value=100, format="%.3f"
    )
}


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def probability_bar(probability: float, threshold: float) -> go.Figure:
    """Horizontal scale 0-100% with risk bands, the score and the decision threshold."""
    pct, thr = probability * 100, min(threshold * 100, 100)
    fig = go.Figure()
    for lo, hi, color in [(0, 20, "#EEF2F6"), (20, 60, "#DDE5EE"), (60, 100, "#F4D9D6")]:
        fig.add_shape(type="rect", x0=lo, x1=hi, y0=-0.5, y1=0.5, fillcolor=color, line_width=0, layer="below")
    fig.add_trace(go.Bar(x=[pct], y=[""], orientation="h", width=0.28,
                         marker_color=RED if probability >= threshold else NAVY,
                         hovertemplate="%{x:.3f}%<extra></extra>"))
    fig.add_shape(type="line", x0=thr, x1=thr, y0=-0.5, y1=0.5, line=dict(color=INK, width=2, dash="dot"))
    fig.add_annotation(x=thr, y=0.5, text="threshold", showarrow=False, yshift=10, font=dict(size=11, color=MUTED))
    for x, label in [(10, "Low"), (40, "Medium"), (80, "High")]:
        fig.add_annotation(x=x, y=-0.5, text=label, showarrow=False, yshift=-12, font=dict(size=11, color=MUTED))
    fig.update_xaxes(range=[0, 100], ticksuffix="%", showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, range=[-0.5, 0.5])
    return style_fig(fig, "", 150).update_layout(margin=dict(l=8, r=8, t=24, b=34), showlegend=False)


def result_charts(results: pd.DataFrame, threshold: float) -> None:
    left, right = st.columns([3, 2])
    with left:
        fig = px.histogram(results, x="Fraud Probability", nbins=40, color="Predicted Class",
                           color_discrete_map=PALETTE, barmode="overlay", opacity=0.9)
        fig.add_vline(x=threshold, line_dash="dot", line_color=INK,
                      annotation_text="threshold", annotation_position="top")
        fig.update_yaxes(type="log", title="Transactions (log)")
        fig.update_xaxes(title="Fraud probability")
        show(style_fig(fig, "Score distribution"))
    with right:
        if "Risk Category" in results:
            risk = results["Risk Category"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
            fig = px.bar(risk.rename_axis("Risk").reset_index(name="Count"), x="Risk", y="Count",
                         color="Risk", color_discrete_map=RISK_COLORS, text="Count")
            fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False)
            fig.update_yaxes(title=None)
            fig.update_xaxes(title=None)
            show(style_fig(fig, "Risk bands"))
    if "Amount" in results:
        fig = px.box(results, x="Amount", y="Predicted Class", color="Predicted Class",
                     color_discrete_map=PALETTE, points=False, orientation="h")
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title=None)
        show(style_fig(fig, "Transaction amount by prediction", 240))


def threshold_analysis(results: pd.DataFrame, model_threshold: float, flagged_now: int) -> None:
    st.caption("Exploration only. This does not change the saved model, its threshold, or the predictions.")
    probs = results["Fraud Probability"].to_numpy()
    total = len(probs)
    default = float(min(max(model_threshold * 100, 0.01), 100.0))
    chosen = st.slider("Hypothetical threshold (%)", 0.01, 100.0, default, 0.01) / 100
    n_flag = int((probs >= chosen).sum())
    strip([
        ("Flagged at this threshold", f"{n_flag:,}", f"{n_flag - flagged_now:+,} vs saved model"),
        ("Flag rate", f"{n_flag / total:.2%}", "of processed rows"),
        ("Saved model threshold", f"{model_threshold:.4%}", f"{flagged_now:,} flagged"),
    ])
    if "Class" in results and results["Class"].isin([0, 1]).all() and (results["Class"] == 1).any():
        actual = results["Class"].to_numpy() == 1
        pred = probs >= chosen
        tp, fp, fn = int((pred & actual).sum()), int((pred & ~actual).sum()), int((~pred & actual).sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        def review_cost(cut: float) -> float:
            flagged = probs >= cut
            return float((~flagged & actual).sum() * config.FN_COST + (flagged & ~actual).sum() * config.FP_COST)

        cost_now, cost_saved = review_cost(chosen), review_cost(model_threshold)
        section("Against the uploaded Class labels")
        strip([
            ("Precision", "n/a" if np.isnan(precision) else f"{precision:.2%}", f"{tp:,} true / {fp:,} false alerts"),
            ("Recall", "n/a" if np.isnan(recall) else f"{recall:.2%}", f"{fn:,} missed of {tp + fn:,} labelled fraud"),
            ("Expected cost", f"{cost_now:,.0f}", f"{config.FN_COST:,.0f} per miss · {config.FP_COST:,.0f} per false alert"),
            ("Cost vs saved threshold", f"{cost_now - cost_saved:+,.0f}", "negative is cheaper"),
        ])
    grid = np.unique(np.concatenate([np.geomspace(1e-4, 1, 60), [chosen, model_threshold]]))
    curve = pd.DataFrame({"Threshold": grid, "Flagged": [(probs >= g).sum() for g in grid]})
    fig = px.line(curve, x="Threshold", y="Flagged", log_x=True, color_discrete_sequence=[NAVY])
    fig.add_vline(x=chosen, line_dash="dot", line_color=RED)
    fig.update_yaxes(title="Transactions flagged")
    fig.update_xaxes(title="Threshold (log scale)", tickformat=".2%")
    show(style_fig(fig, "Flagged transactions by threshold", 280))


def results_table(results: pd.DataFrame, key: str) -> pd.DataFrame:
    f1, f2, f3 = st.columns([2, 2, 2])
    classes = f1.multiselect("Prediction", [FRAUD, LEGIT], default=[FRAUD, LEGIT], key=f"{key}_cls")
    min_prob = f2.slider("Minimum fraud probability (%)", 0.0, 100.0, 0.0, 0.5, key=f"{key}_prob")
    order = f3.selectbox("Order", ["Highest risk first", "Original order"], key=f"{key}_ord")
    view = results[results["Predicted Class"].isin(classes) & (results["Fraud Probability"] * 100 >= min_prob)]
    if order == "Highest risk first":
        view = view.sort_values("Fraud Probability", ascending=False)
    st.caption(f"{len(view):,} of {len(results):,} rows")
    st.dataframe(prob_column(view), use_container_width=True, hide_index=True, height=420, column_config=PROB_CONFIG)
    return view


def download_row(frame: pd.DataFrame, key: str, suffix: str = "") -> None:
    a, b, _ = st.columns([1, 1, 2])
    a.download_button(f"Download CSV{suffix}", frame.to_csv(index=False).encode(),
                      "transactionguard_results.csv", "text/csv", key=f"{key}_csv", use_container_width=True)
    try:
        b.download_button(f"Download Excel{suffix}", export_excel(frame),
                          "transactionguard_results.xlsx", key=f"{key}_xlsx", use_container_width=True)
    except InferenceError as exc:
        b.warning(str(exc))


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def overview(service: InferenceService) -> None:
    page_header("Overview", "Fraud-screening estimates for analyst review. A flag is a prompt to investigate, not a finding.")
    results = st.session_state.get("results")
    threshold = service.details.threshold

    if results is None:
        strip([
            ("Model", service.details.name, f"version {service.details.version}"),
            ("Decision threshold", f"{threshold:.4%}", "probability cut-off"),
            ("Input features", len(service.details.feature_names), "numeric fields"),
        ])
        st.info("No transactions analysed in this session. Use Bulk analysis to score a file, or Single transaction to score one row.")
        return

    total = len(results)
    flagged = int((results["Predicted Class"] == FRAUD).sum())
    meta = st.session_state.get("results_meta", {})
    strip([
        ("Processed", f"{total:,}", meta.get("source", "")),
        ("Flagged for review", f"{flagged:,}", f"{flagged / total:.2%} of processed", RED),
        ("Likely legitimate", f"{total - flagged:,}", ""),
        ("Highest score", f"{results['Fraud Probability'].max():.2%}", "single transaction"),
    ])
    tab_dist, tab_queue, tab_thr = st.tabs(["Distribution", "Review queue", "Threshold analysis"])
    with tab_dist:
        result_charts(results, threshold)
    with tab_queue:
        st.caption("The 25 highest-scoring transactions in this session.")
        st.dataframe(prob_column(results.nlargest(25, "Fraud Probability")), use_container_width=True,
                     hide_index=True, column_config=PROB_CONFIG)
    with tab_thr:
        threshold_analysis(results, threshold, flagged)


def bulk_analysis(service: InferenceService) -> None:
    page_header("Bulk analysis", "Score a CSV or Excel file. Files stay in this browser session and are not stored.")
    left, right = st.columns([4, 1])
    with right:
        st.download_button("Input template (CSV)", template_bytes(service.details.feature_names),
                           "transactionguard_template.csv", "text/csv", use_container_width=True)
    with left:
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
        strip([
            ("Rows", f"{len(frame):,}", uploaded.name),
            ("Columns", len(frame.columns), f"{len(extra)} extra" if extra else "all expected"),
            ("Missing cells", f"{int(frame.isna().sum().sum()):,}", ""),
            ("Validation", "Passed" if not errors else f"{len(errors)} issue(s)", "", INK if not errors else RED),
        ])
        with st.expander("Preview first 10 rows"):
            st.dataframe(frame.head(10), use_container_width=True)
        if extra:
            st.info(f"Extra columns are preserved but not sent to the model: {', '.join(extra[:8])}")
        if "Class" in frame:
            st.info("`Class` is treated as optional evaluation information and excluded from model inputs.")
        if errors:
            for error in errors[:8]:
                st.error(error)
            return

        if st.button("Analyse transactions", type="primary"):
            with st.spinner("Applying the saved model and feature pipeline…"):
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

    results = st.session_state.get("results")
    if results is None:
        return
    meta = st.session_state.get("results_meta", {})
    section(f"Results · {meta.get('source', '')} · {meta.get('time', '')}")
    tab_table, tab_charts = st.tabs(["Table", "Charts"])
    with tab_table:
        view = results_table(results, "bulk")
        download_row(view, "bulk", " (current view)")
    with tab_charts:
        result_charts(results, service.details.threshold)


def render_prediction(service: InferenceService, values: dict[str, float]) -> None:
    try:
        prediction = service.predict(pd.DataFrame([values])).iloc[0]
    except InferenceError as exc:
        st.error(str(exc))
        return
    probability = float(prediction["Fraud Probability"])
    label, risk = str(prediction["Predicted Class"]), str(prediction["Risk Category"])
    color = RED if label == FRAUD else NAVY
    st.markdown(
        f'<div class="tg-callout" style="border-left-color:{color}">'
        f'<div class="tg-callout-title" style="color:{color}">{escape(label)} &nbsp;{chip(risk + " risk", RISK_COLORS.get(risk, MUTED))}</div>'
        f'<div class="tg-callout-body">Model estimate for triage. Not a confirmation of fraud.</div></div>',
        unsafe_allow_html=True,
    )
    strip([
        ("Fraud probability", f"{probability:.3%}", "model score"),
        ("Decision threshold", f"{service.details.threshold:.4%}", "flag at or above"),
        ("Risk band", risk, "presentation only"),
    ])
    show(probability_bar(probability, service.details.threshold))

    history = st.session_state.setdefault("single_history", [])
    history.insert(0, {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Amount": values.get("Amount"),
        "Probability (%)": round(probability * 100, 4),
        "Prediction": label,
        "Risk": risk,
    })
    del history[25:]


def single_transaction(service: InferenceService) -> None:
    page_header("Single transaction", "V1–V28 are anonymised PCA components and cannot be derived from customer details.")
    features = service.details.feature_names
    tab_manual, tab_paste = st.tabs(["Enter values", "Paste a row"])

    with tab_manual:
        with st.form("single_transaction"):
            values: dict[str, float] = {}
            basics, components = st.columns([1, 3])
            with basics:
                section("Core fields")
                for name in [x for x in features if x in {"Time", "Amount"}]:
                    help_text = "Seconds elapsed from the dataset reference point." if name == "Time" else "Transaction amount."
                    values[name] = st.number_input(name, value=0.0, format="%.8f", help=help_text)
            with components:
                section("Anonymised components")
                grid = st.columns(4)
                for i, name in enumerate([x for x in features if x not in {"Time", "Amount"}]):
                    values[name] = grid[i % 4].number_input(name, value=0.0, format="%.8f")
            submitted = st.form_submit_button("Analyse transaction", type="primary")
        if submitted:
            render_prediction(service, {k: values[k] for k in features})

    with tab_paste:
        st.caption(f"Paste {len(features)} comma-separated numbers in exactly this order:")
        st.code(", ".join(features), language=None)
        raw = st.text_area("Row values", height=110, label_visibility="collapsed",
                           placeholder="0, -1.359807, -0.072781, ...")
        if st.button("Analyse pasted row", type="primary"):
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
        section(f"Session history ({len(history)})")
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


def performance(service: InferenceService) -> None:
    page_header("Model performance", "Evaluation on a held-out test set.")
    st.warning("Synthetic-data evaluation. These figures are not live performance or validation on real financial transactions.")
    metrics_path = config.REPORT_DIR / "test_metrics.json"
    if not metrics_path.exists():
        st.info("No evaluation report found. Run `python scripts/evaluate.py` after training.")
        return
    metrics = pd.read_json(metrics_path, typ="series")
    labels = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "precision": "Precision", "recall": "Recall",
              "f1": "F1", "fpr": "False-positive rate", "fnr": "False-negative rate"}
    present = [k for k in labels if k in metrics]
    for row in (present[:4], present[4:]):
        if row:
            strip([(labels[k], f"{float(metrics[k]):.4f}", "") for k in row])
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure(go.Bar(x=[float(metrics[k]) for k in present][::-1], y=[labels[k] for k in present][::-1],
                               orientation="h", marker_color=NAVY,
                               text=[f"{float(metrics[k]):.3f}" for k in present][::-1], textposition="outside"))
        fig.update_xaxes(range=[0, 1.08])
        show(style_fig(fig, "Metric overview", 360))
    with right:
        image = config.REPORT_DIR / "confusion_matrix.png"
        if image.exists():
            st.image(str(image), caption="Confusion matrix, synthetic held-out test set")
    st.caption(f"Model: {service.details.name} · threshold: {service.details.threshold:.4%}")


def about(service: InferenceService) -> None:
    page_header("About", "What the model expects and how results are presented.")
    strip([
        ("Model", service.details.name, ""),
        ("Version", service.details.version, ""),
        ("Input features", len(service.details.feature_names), "numeric fields"),
    ])
    section("Required input features")
    st.code(", ".join(service.details.feature_names), language=None)
    section("Risk bands")
    st.markdown(
        f"{chip('Low', RISK_COLORS['Low'])} &nbsp;below 20% &nbsp;&nbsp; "
        f"{chip('Medium', RISK_COLORS['Medium'])} &nbsp;20% to below 60% &nbsp;&nbsp; "
        f"{chip('High', RISK_COLORS['High'])} &nbsp;60% and above",
        unsafe_allow_html=True,
    )
    st.caption("Bands are for presentation only. They do not change the saved classification threshold.")
    section("Session")
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

    st.sidebar.markdown(
        f'<div class="tg-brand">TransactionGuard</div>'
        f'<div class="tg-brand-sub">{escape(str(service.details.name))} · v{escape(str(service.details.version))}</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.write("")
    page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    results = st.session_state.get("results")
    if results is not None:
        st.sidebar.caption(f"{len(results):,} rows analysed this session")
    st.sidebar.caption("Runs locally. Uploaded transaction data is not stored.")

    {
        "Overview": overview, "Bulk analysis": bulk_analysis, "Single transaction": single_transaction,
        "Model performance": performance, "About": about,
    }[page](service)


if __name__ == "__main__":
    main()