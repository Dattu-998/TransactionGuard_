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

# Premium dark fintech palette. The inference layer deliberately does not depend on these values.
NAVY = "#8B5CF6"
CYAN = "#67E8F9"
RED = "#F87171"
GREEN = "#4ADE80"
INK = "#F4F5FA"
MUTED = "#9298AC"
LINE = "rgba(196,181,253,.16)"
PALETTE = {FRAUD: RED, LEGIT: GREEN}
RISK_COLORS = {"Low": GREEN, "Medium": "#FBBF24", "High": RED}

PAGES = ["Overview", "Bulk analysis", "Single transaction", "Model performance", "About"]

st.set_page_config(page_title="TransactionGuard", page_icon="🛡️", layout="wide")


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def inject_style() -> None:
    st.markdown(
        f"""<style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        #MainMenu, footer {{visibility:hidden;}}
        header[data-testid="stHeader"] {{background:rgba(8,10,18,.72);backdrop-filter:blur(14px);}}
        .stApp {{background:radial-gradient(circle at 72% -10%,#2a1d5d 0,transparent 28%),radial-gradient(circle at 10% 38%,#10243c 0,transparent 24%),#080A12;color:{INK};font-family:Inter,-apple-system,"Segoe UI",sans-serif;}}
        .block-container {{padding-top:2.2rem;padding-bottom:3rem;max-width:1320px;}}
        h1,h2,h3,h4 {{color:{INK};font-family:"Space Grotesk",Inter,sans-serif;letter-spacing:-.035em;}}
        [data-testid="stSidebar"] {{background:linear-gradient(180deg,#101326,#090b14);border-right:1px solid {LINE};}}
        [data-testid="stSidebar"] > div:first-child {{padding-top:1.5rem;}}
        .tg-brand {{font:700 1.12rem "Space Grotesk",sans-serif;color:#fff;letter-spacing:-.04em;}}
        .tg-brand:before {{content:"✦";display:inline-grid;place-items:center;width:28px;height:28px;margin-right:9px;border-radius:9px;background:linear-gradient(135deg,{NAVY},{CYAN});color:#080A12;font-size:.85rem;}}
        .tg-brand-sub {{font-size:.7rem;color:{MUTED};margin:8px 0 18px 38px;text-transform:uppercase;letter-spacing:.1em;}}
        [data-testid="stSidebar"] [role="radiogroup"] label {{border:1px solid transparent;border-radius:10px;padding:7px 8px;margin:2px 0;transition:.2s;}}
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {{background:rgba(139,92,246,.12);border-color:rgba(196,181,253,.18);}}
        .tg-title {{font:700 clamp(1.8rem,3vw,2.45rem) "Space Grotesk",sans-serif;color:{INK};margin:0;letter-spacing:-.05em;}}
        .tg-subtitle {{color:{MUTED};font-size:.95rem;margin:.38rem 0 1.1rem;max-width:720px;line-height:1.6;}}
        .tg-rule {{border:0;border-top:1px solid {LINE};margin:0 0 1.4rem;}}
        .tg-strip {{display:flex;background:linear-gradient(135deg,rgba(25,29,51,.92),rgba(14,17,33,.8));border:1px solid {LINE};border-radius:14px;margin-bottom:1.4rem;overflow:hidden;box-shadow:0 18px 50px rgba(0,0,0,.16);}}
        .tg-cell {{flex:1;padding:17px 20px;border-right:1px solid {LINE};min-width:0;}}
        .tg-cell:last-child {{border-right:0;}}
        .tg-label {{font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; color:{MUTED}; font-weight:600;}}
        .tg-value {{font:600 1.6rem "Space Grotesk",sans-serif;color:{INK};margin-top:4px;
                    font-variant-numeric:tabular-nums; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}}
        .tg-sub {{font-size:.78rem; color:{MUTED}; margin-top:2px;}}

        .tg-section {{font-size:.72rem; letter-spacing:.08em; text-transform:uppercase;
                      color:{MUTED}; font-weight:600; margin:1.2rem 0 .5rem 0;}}

        .tg-callout {{background:linear-gradient(135deg,rgba(28,33,57,.96),rgba(15,18,34,.9));border:1px solid {LINE};border-left:4px solid {MUTED};border-radius:14px;padding:17px 20px;margin-bottom:1rem;box-shadow:0 15px 35px rgba(0,0,0,.16);}}
        .tg-callout-title {{font-size:1.15rem; font-weight:650;}}
        .tg-callout-body {{color:{MUTED}; font-size:.88rem; margin-top:2px;}}

        .tg-chip {{display:inline-block; padding:1px 9px; border-radius:4px; font-size:.75rem;
                   font-weight:600; border:1px solid; vertical-align:middle;}}

        div[data-testid="stTabs"] button[role="tab"] {{font-weight:550; color:{MUTED};}}
        div[data-testid="stTabs"] button[aria-selected="true"] {{color:{INK};}}
        div.stButton > button, div.stDownloadButton > button, div.stFormSubmitButton > button {{
            border-radius:9px;font-weight:600;border:1px solid {LINE};background:#1a1f36;color:{INK};transition:transform .2s,box-shadow .2s;}}
        div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {{
            background:linear-gradient(135deg,{NAVY},#6D4EDC);border-color:{NAVY};color:#fff;box-shadow:0 8px 22px rgba(139,92,246,.25);}}
        div.stButton > button[kind="primary"]:hover, div.stFormSubmitButton > button[kind="primary"]:hover {{
            background:#7552E3;border-color:#A78BFA;color:#fff;transform:translateY(-1px);}}
        [data-testid="stFileUploaderDropzone"] {{background:linear-gradient(135deg,rgba(28,33,57,.8),rgba(12,15,28,.7));border:1px dashed rgba(196,181,253,.38);border-radius:14px;padding:1.5rem;}}
        [data-testid="stMetric"] {{background:#171b30;border:1px solid {LINE};border-radius:12px;padding:12px 16px;}}
        [data-testid="stDataFrame"] {{border:1px solid {LINE};border-radius:12px;overflow:hidden;}}
        .tg-hero {{position:relative;overflow:hidden;min-height:390px;padding:clamp(25px,5vw,64px);border:1px solid rgba(196,181,253,.2);border-radius:24px;background:linear-gradient(120deg,rgba(24,26,52,.98),rgba(12,15,30,.88));box-shadow:0 30px 90px rgba(0,0,0,.35);margin-bottom:1.6rem;}}
        .tg-hero:before {{content:"";position:absolute;inset:0;background-image:linear-gradient(rgba(196,181,253,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(196,181,253,.045) 1px,transparent 1px);background-size:36px 36px;mask-image:linear-gradient(90deg,#000,transparent);}}
        .tg-eyebrow {{position:relative;color:{CYAN};font-size:.7rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;}}
        .tg-hero h1 {{position:relative;max-width:600px;font-size:clamp(2.8rem,5.2vw,5rem);line-height:.98;margin:.8rem 0 1rem;}}
        .tg-hero p {{position:relative;max-width:530px;color:#B8BED2;font-size:1rem;line-height:1.7;}}
        .tg-orbit {{position:absolute;right:8%;top:50%;width:250px;height:250px;transform:translateY(-50%) rotateX(62deg) rotateZ(-18deg);border:1px solid rgba(103,232,249,.5);border-radius:50%;box-shadow:0 0 45px rgba(103,232,249,.16),inset 0 0 34px rgba(139,92,246,.25);animation:tgspin 15s linear infinite;}}
        .tg-orbit:before,.tg-orbit:after {{content:"";position:absolute;inset:30px;border:1px solid rgba(196,181,253,.52);border-radius:50%;}} .tg-orbit:after {{inset:70px;background:radial-gradient(circle at 34% 30%,#fff 0 2%,{CYAN} 7%,{NAVY} 34%,rgba(15,18,44,.8) 68%);box-shadow:0 0 42px rgba(139,92,246,.9);}}
        .tg-node {{position:absolute;right:25%;top:28%;width:10px;height:10px;border-radius:50%;background:{CYAN};box-shadow:0 0 20px {CYAN};animation:tgfloat 4s ease-in-out infinite;}}
        @keyframes tgspin {{to {{transform:translateY(-50%) rotateX(62deg) rotateZ(342deg);}}}} @keyframes tgfloat {{50% {{transform:translateY(22px);}}}}
        @media (prefers-reduced-motion:reduce) {{*,*:before,*:after {{animation-duration:.01ms!important;transition-duration:.01ms!important;}}}} @media(max-width:760px) {{.tg-orbit,.tg-node {{opacity:.28;right:-15%;}}.tg-hero {{min-height:360px;}}.block-container {{padding-left:1rem;padding-right:1rem;}}}}
        </style>""",
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="tg-title">{escape(title)}</div>'
        f'<div class="tg-subtitle">{escape(subtitle)}</div><hr class="tg-rule">',
        unsafe_allow_html=True,
    )


def navigate(page: str) -> None:
    """Callback-safe navigation from overview actions."""
    st.session_state.nav = page


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
        template="plotly_dark",
        title=dict(text=title, font=dict(size=13, color=INK), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, Segoe UI, Roboto, sans-serif", size=12, color="#CBD5E1"),
        height=height,
        margin=dict(l=8, r=8, t=44, b=8),
        legend=dict(orientation="h", y=-0.22, x=0, title=None),
    )
    fig.update_xaxes(gridcolor="rgba(196,181,253,.08)", linecolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(196,181,253,.08)", linecolor=LINE, zeroline=False)
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
    results = st.session_state.get("results")
    threshold = service.details.threshold

    if results is None:
        st.markdown(
            '''<section class="tg-hero"><div class="tg-eyebrow">AI-assisted fraud screening</div>
            <h1>Financial intelligence.<br>Built for trust.</h1>
            <p>Screen transaction data with a transparent, local review workflow. Every flag is an estimate for an analyst to investigate—not a finding of fraud.</p>
            <div class="tg-orbit"></div><div class="tg-node"></div></section>''',
            unsafe_allow_html=True,
        )
        a, b, _ = st.columns([1.35, 1.55, 3])
        a.button("Analyse transactions", type="primary", use_container_width=True,
                 on_click=navigate, args=("Bulk analysis",))
        b.button("Explore model insights", use_container_width=True,
                 on_click=navigate, args=("Model performance",))
        section("Model at a glance")
        strip([
            ("Model", service.details.name, f"version {service.details.version}"),
            ("Decision threshold", f"{threshold:.4%}", "probability cut-off"),
            ("Input features", len(service.details.feature_names), "numeric fields"),
        ])
        st.info("No transactions have been analysed in this session. Start with a CSV/Excel file or examine one transaction at a time.")
        return

    page_header("Command centre", "Session-level fraud-screening estimates for analyst review. A flag is a prompt to investigate, not a finding.")

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
    page_header("Bulk analysis", "A private, high-signal workspace for CSV and Excel transaction review. Files remain in this browser session.")
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
    page_header("Model intelligence", "A transparent view of the saved model's held-out evaluation.")
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
    page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed", key="nav")
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
