"""TransactionGuard's local Streamlit fraud-review interface."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from transactionguard.inference import InferenceError, InferenceService

ROOT = Path(__file__).parent
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
PALETTE = {"Potential Fraud": "#EF4444", "Likely Legitimate": "#22C55E"}


st.set_page_config(page_title="TransactionGuard", page_icon="🛡️", layout="wide")


def inject_style() -> None:
    st.markdown(
        """<style>
        .stApp { background: #0B1020; color: #F8FAFC; }
        [data-testid="stSidebar"] { background: #11182a; }
        .tg-card {background:#141B2D;border:1px solid #283552;border-radius:14px;padding:18px;}
        .tg-label {color:#94A3B8;font-size:.85rem;margin-bottom:.3rem;}
        .tg-value {font-size:1.6rem;font-weight:700;}
        .tg-muted {color:#94A3B8;}
        </style>""",
        unsafe_allow_html=True,
    )


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


def result_charts(results: pd.DataFrame) -> None:
    left, right = st.columns(2)
    with left:
        counts = results["Predicted Class"].value_counts().rename_axis("Prediction").reset_index(name="Count")
        chart = px.bar(counts, x="Prediction", y="Count", color="Prediction", color_discrete_map=PALETTE)
        chart.update_layout(template="plotly_dark", title="Prediction outcomes", showlegend=False)
        st.plotly_chart(chart, use_container_width=True)
    with right:
        chart = px.histogram(results, x="Fraud Probability", nbins=30, color_discrete_sequence=["#8B5CF6"])
        chart.update_layout(template="plotly_dark", title="Predicted fraud probability")
        st.plotly_chart(chart, use_container_width=True)
    if "Amount" in results:
        chart = px.histogram(results, x="Amount", color="Predicted Class", barmode="overlay", color_discrete_map=PALETTE)
        chart.update_layout(template="plotly_dark", title="Transaction amount by prediction")
        st.plotly_chart(chart, use_container_width=True)


def overview(service: InferenceService) -> None:
    st.title("TransactionGuard")
    st.caption("Local fraud-screening estimates for review — never a confirmation of fraud.")
    results = st.session_state.get("results")
    if results is None:
        st.info("No transactions have been analyzed in this session. Upload a file or use Single Transaction to begin.")
        st.markdown("### Model ready")
        st.write(f"**{service.details.name}** · decision threshold: `{service.details.threshold:.4%}`")
        return
    total = len(results)
    flagged = int((results["Predicted Class"] == "Potential Fraud").sum())
    cards = [("Processed", total), ("Flagged for review", flagged), ("Likely legitimate", total - flagged), ("Flag rate", f"{flagged / total:.1%}")]
    columns = st.columns(4)
    for column, (label, value) in zip(columns, cards):
        column.markdown(f'<div class="tg-card"><div class="tg-label">{label}</div><div class="tg-value">{value}</div></div>', unsafe_allow_html=True)
    st.caption(f"Model: {service.details.name} · selected threshold: {service.details.threshold:.4%}")
    result_charts(results)


def bulk_analysis(service: InferenceService) -> None:
    st.title("Bulk Analysis")
    st.caption("Files stay in this browser session and are not persisted by the application.")
    st.download_button("Download CSV input template", template_bytes(service.details.feature_names), "transactionguard_template.csv", "text/csv")
    uploaded = st.file_uploader("Upload transactions", type=["csv", "xlsx", "xls"])
    if not uploaded:
        return
    try:
        frame = read_upload(uploaded)
    except InferenceError as exc:
        st.error(str(exc))
        return
    if frame.columns.duplicated().any():
        st.error("Duplicate column names are not supported. Rename them and upload again.")
        return
    st.write(f"{len(frame):,} rows · {len(frame.columns)} columns")
    st.dataframe(frame.head(10), use_container_width=True)
    _, errors = service.validate(frame)
    extra = [column for column in frame.columns if column not in service.details.feature_names and column != "Class"]
    if extra:
        st.info(f"Extra columns will be preserved but not sent to the model: {', '.join(extra[:8])}")
    if "Class" in frame:
        st.info("Class is treated as optional evaluation information and is excluded from prediction inputs.")
    if errors:
        for error in errors[:8]:
            st.error(error)
        return
    if st.button("Analyze transactions", type="primary"):
        with st.spinner("Applying the saved model and feature engineering pipeline…"):
            predictions = service.predict(frame)
        results = pd.concat([frame.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
        st.session_state.results = results
        st.success(f"Analyzed {len(results):,} transaction(s).")
    results = st.session_state.get("results")
    if results is not None:
        st.subheader("Most recent results")
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.download_button("Download results (CSV)", results.to_csv(index=False).encode(), "transactionguard_results.csv", "text/csv")
        try:
            st.download_button("Download results (Excel)", export_excel(results), "transactionguard_results.xlsx")
        except InferenceError as exc:
            st.warning(str(exc))


def single_transaction(service: InferenceService) -> None:
    st.title("Single Transaction")
    st.caption("Enter the exact model features. V1–V28 are anonymized PCA features and cannot be derived from customer details.")
    with st.form("single_transaction"):
        values: dict[str, float] = {}
        basics, components = st.columns(2)
        with basics:
            for name in [x for x in service.details.feature_names if x in {"Time", "Amount"}]:
                help_text = "Seconds elapsed from the dataset reference point." if name == "Time" else "Transaction amount."
                values[name] = st.number_input(name, value=0.0, format="%.8f", help=help_text)
        with components:
            st.caption("Anonymized components")
            for name in [x for x in service.details.feature_names if x not in {"Time", "Amount"}]:
                values[name] = st.number_input(name, value=0.0, format="%.8f")
        submitted = st.form_submit_button("Analyze transaction", type="primary")
    if submitted:
        try:
            prediction = service.predict(pd.DataFrame([values])).iloc[0]
        except InferenceError as exc:
            st.error(str(exc))
            return
        probability = float(prediction["Fraud Probability"])
        st.subheader(str(prediction["Predicted Class"]))
        a, b, c = st.columns(3)
        a.metric("Fraud probability", f"{probability:.2%}")
        b.metric("Risk category", prediction["Risk Category"])
        c.metric("Decision threshold", f"{service.details.threshold:.4%}")
        st.warning("This is a model estimate for triage, not a guarantee or confirmation of fraud.")


def performance(service: InferenceService) -> None:
    st.title("Model Performance")
    st.warning("These are synthetic-data evaluation results, not live application performance or validation on real financial transactions.")
    metrics_path = ROOT / "reports" / "test_metrics.json"
    if not metrics_path.exists():
        st.info("No evaluation report found. Run `python scripts/evaluate.py` after training.")
        return
    metrics = pd.read_json(metrics_path, typ="series")
    labels = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "precision": "Precision", "recall": "Recall", "f1": "F1", "fpr": "False-positive rate", "fnr": "False-negative rate"}
    display = pd.DataFrame({"Metric": [labels[k] for k in labels if k in metrics], "Value": [metrics[k] for k in labels if k in metrics]})
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption(f"Model: {service.details.name} · threshold: {service.details.threshold:.4%}")
    image = ROOT / "reports" / "confusion_matrix.png"
    if image.exists():
        st.image(str(image), caption="Confusion matrix from the synthetic held-out test set")


def about(service: InferenceService) -> None:
    st.title("About / Model Information")
    st.write(f"**Model:** {service.details.name} (version {service.details.version})")
    st.write(f"**Required input features:** {len(service.details.feature_names)} numeric fields")
    st.code(", ".join(service.details.feature_names), language=None)
    st.markdown("Risk labels are presentation bands: Low <20%, Medium 20–<60%, High ≥60%. They do not change the saved classification threshold.")
    if st.button("Clear session results"):
        st.session_state.pop("results", None)
        st.success("Session results cleared.")


def main() -> None:
    inject_style()
    try:
        service = get_service()
    except InferenceError as exc:
        st.error(f"Model unavailable: {exc}")
        st.stop()
    page = st.sidebar.radio("Navigate", ["Overview", "Bulk Analysis", "Single Transaction", "Model Performance", "About / Model Information"])
    st.sidebar.caption("TransactionGuard runs locally. Uploaded transaction data is not stored.")
    {"Overview": overview, "Bulk Analysis": bulk_analysis, "Single Transaction": single_transaction, "Model Performance": performance, "About / Model Information": about}[page](service)


if __name__ == "__main__":
    main()
