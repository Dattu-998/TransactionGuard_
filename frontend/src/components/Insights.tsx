import { Activity, ArrowUpRight, SlidersHorizontal } from "lucide-react";
import { SectionHeading } from "./ui";
import type { ModelInfo } from "../types";

const percentageMetrics = new Set([
  "pr_auc",
  "roc_auc",
  "precision",
  "recall",
  "f1",
  "fpr",
  "fnr",
]);

const metricLabels: Record<string, string> = {
  pr_auc: "PR AUC",
  roc_auc: "ROC AUC",
  precision: "Precision",
  recall: "Recall",
  f1: "F1 score",
  fpr: "False positive rate",
  fnr: "False negative rate",
  tn: "True negatives",
  fp: "False positives",
  fn: "False negatives",
  tp: "True positives",
};

function formatMetric(name: string, value: number) {
  if (percentageMetrics.has(name)) {
    return `${(value * 100).toFixed(2)}%`;
  }
  return Number.isInteger(value) ? value.toLocaleString() : value.toString();
}

export function Insights({ info }: { info: ModelInfo | null }) {
  const metrics = Object.entries(info?.metrics ?? {}).filter(
    ([, value]) => typeof value === "number" && Number.isFinite(value),
  );

  return (
    <section className="page-content content-width insights-page">
      <SectionHeading
        eyebrow="MODEL TRANSPARENCY"
        title="Model insights"
        description="Understand which model is active, how its threshold is set, and what evaluation data is available."
      />

      <div className="model-overview-card">
        <div className="model-card-main">
          <span className="model-card-icon"><Activity size={20} /></span>
          <div>
            <p className="eyebrow">ACTIVE CLASSIFIER</p>
            <h2>{info?.name ?? "Model information unavailable"}</h2>
            <span className="version-label">{info ? `Version ${info.version}` : "Check API connection"}</span>
          </div>
        </div>
        <div className="threshold-panel">
          <div className="threshold-label">
            <SlidersHorizontal size={15} />
            <span>Saved decision threshold</span>
          </div>
          <strong>
            {info ? `${(info.threshold * 100).toPrecision(3)}%` : "—"}
          </strong>
          <p>
            Predictions at or above this model threshold are flagged for
            review.
          </p>
        </div>
      </div>

      <div className="metrics-section">
        <div className="metrics-heading">
          <div>
            <p className="eyebrow"><span className="eyebrow-line" /> AVAILABLE EVALUATION</p>
            <h2>Metrics from the API</h2>
          </div>
          <span className="evaluation-tag">
            <i /> {info?.evaluation_data ?? "Evaluation source not specified"}
          </span>
        </div>
        {!info ? (
          <div className="metrics-empty" role="status">
            Model metrics are unavailable until the API responds.
          </div>
        ) : metrics.length === 0 ? (
          <div className="metrics-empty">
            No evaluation metrics were provided by the API.
          </div>
        ) : (
          <div className="metric-grid">
            {metrics.map(([key, value]) => (
              <article className="metric-card" key={key}>
                <span>{metricLabels[key] ?? key.replace(/_/g, " ")}</span>
                <strong>{formatMetric(key, value)}</strong>
                <ArrowUpRight size={15} aria-hidden="true" />
              </article>
            ))}
          </div>
        )}
      </div>

      <div className="limitations-card">
        <span className="limitations-mark"><span>!</span></span>
        <div>
          <p className="eyebrow">READ THE LIMITATIONS</p>
          <h2>Evaluation is not a real-world guarantee.</h2>
          <p>
            The available evaluation metrics are derived from synthetic data.
            They do not establish performance on real transactions. A flagged
            prediction is a signal to investigate, not a conclusion about
            intent or fraud.
          </p>
        </div>
      </div>

      <div className="feature-schema">
        <div>
          <p className="eyebrow">INPUT SCHEMA</p>
          <h2>Model feature set</h2>
          <p>Feature names provided by the connected API.</p>
        </div>
        <div className="feature-chips">
          {(info?.features ?? []).map((feature) => (
            <span key={feature}>{feature}</span>
          ))}
          {info && info.features.length === 0 && (
            <span className="schema-empty">No feature names were provided.</span>
          )}
          {!info && <span className="schema-empty">Waiting for model metadata.</span>}
        </div>
      </div>
    </section>
  );
}
