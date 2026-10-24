import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ArrowDownToLine, BarChart3 } from "lucide-react";
import type { AnalysisResult } from "../types";

const formatPercent = (value: number) =>
  `${(value * 100).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 3,
  })}%`;

function downloadCsv(result: AnalysisResult) {
  const rows = result.rows;
  const keys = Object.keys(rows[0] || {});
  const escapeCsv = (value: unknown) => {
    const text = String(value ?? "");
    return `"${text.replace(/"/g, '""')}"`;
  };
  const csv = [
    keys.map(escapeCsv).join(","),
    ...rows.map((row) => keys.map((key) => escapeCsv(row[key])).join(",")),
  ].join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "transactionguard-results.csv";
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function ProbabilityTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ value: number; payload: { index: number } }>;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <span>Transaction {payload[0].payload.index}</span>
      <strong>{formatPercent(payload[0].value / 100)}</strong>
    </div>
  );
}

export function ResultsDashboard({ result }: { result: AnalysisResult | null }) {
  if (!result) {
    return (
      <section className="results-empty" aria-labelledby="results-title">
        <div className="results-empty-icon"><BarChart3 size={19} /></div>
        <div>
          <p className="eyebrow">YOUR RESULTS</p>
          <h2 id="results-title">Analysis results will appear here.</h2>
          <p>Submit a transaction or batch file to see the model’s actual output.</p>
        </div>
      </section>
    );
  }

  const chartData = result.rows.slice(0, 100).map((row, index) => ({
    index: index + 1,
    score: Number(row["Fraud Probability"]) * 100,
  }));
  const notFlagged = result.summary.processed - result.summary.flagged;

  return (
    <section className="results-section" aria-labelledby="results-heading">
      <div className="results-heading">
        <div>
          <p className="eyebrow"><span className="eyebrow-line" /> MODEL OUTPUT</p>
          <h2 id="results-heading">Analysis results</h2>
          <p>Based on the prediction response for this submission.</p>
        </div>
        <button
          className="button button-secondary export-button"
          type="button"
          onClick={() => downloadCsv(result)}
          disabled={result.rows.length === 0}
        >
          <ArrowDownToLine size={16} /> Export CSV
        </button>
      </div>

      {result.rows.length === 0 ? (
        <div className="empty-results-message">
          The API returned no transaction rows to display.
        </div>
      ) : (
        <>
          <div className="kpi-grid">
            <article className="kpi-card">
              <span>TRANSACTIONS PROCESSED</span>
              <strong>{result.summary.processed.toLocaleString()}</strong>
              <small>In this analysis</small>
            </article>
            <article className="kpi-card kpi-flagged">
              <span>FLAGGED FOR REVIEW</span>
              <strong>{result.summary.flagged.toLocaleString()}</strong>
              <small>Model classification</small>
            </article>
            <article className="kpi-card">
              <span>NOT FLAGGED</span>
              <strong>{notFlagged.toLocaleString()}</strong>
              <small>Processed minus flagged</small>
            </article>
            <article className="kpi-card">
              <span>HIGHEST PROBABILITY</span>
              <strong>{formatPercent(result.summary.highest_probability)}</strong>
              <small>From API summary</small>
            </article>
          </div>

          <div className="results-chart-card">
            <div className="card-heading">
              <div>
                <h3>Prediction probability</h3>
                <p>First {chartData.length} transactions returned</p>
              </div>
              <span className="chart-legend"><i /> Probability</span>
            </div>
            <div className="chart-wrap" role="img" aria-label="Bar chart showing prediction probability per transaction">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="rgba(185, 196, 218, 0.09)" vertical={false} />
                  <XAxis
                    dataKey="index"
                    stroke="#788298"
                    tickLine={false}
                    axisLine={false}
                    tick={{ fontSize: 11 }}
                    tickMargin={11}
                  />
                  <YAxis
                    stroke="#788298"
                    tickLine={false}
                    axisLine={false}
                    tick={{ fontSize: 11 }}
                    tickFormatter={(value: number) => `${value}%`}
                    width={42}
                  />
                  <Tooltip content={<ProbabilityTooltip />} cursor={{ fill: "rgba(255,255,255,.035)" }} />
                  <Bar dataKey="score" fill="#9b83ed" radius={[3, 3, 0, 0]} maxBarSize={22} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="table-card">
            <div className="card-heading table-heading">
              <div>
                <h3>Transaction detail</h3>
                <p>
                  Showing {Math.min(result.rows.length, 20)} of {result.rows.length} rows
                </p>
              </div>
              <span className="table-context">Rows are displayed in API response order</span>
            </div>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th scope="col">ROW</th>
                    <th scope="col">PREDICTION</th>
                    <th scope="col">PROBABILITY</th>
                    <th scope="col">RISK LEVEL</th>
                    <th scope="col">STATUS</th>
                  </tr>
                </thead>
                <tbody>
                  {result.rows.slice(0, 20).map((row, index) => {
                    const flagged = row["Predicted Class"] === "Potential Fraud";
                    return (
                      <tr key={index}>
                        <td className="row-number">{String(index + 1).padStart(2, "0")}</td>
                        <td>
                          <span className={`prediction-tag${flagged ? " is-flagged" : ""}`}>
                            <i />
                            {row["Predicted Class"]}
                          </span>
                        </td>
                        <td className="probability-cell">
                          {formatPercent(Number(row["Fraud Probability"]))}
                        </td>
                        <td>{row["Risk Category"]}</td>
                        <td className="status-cell">{row["Prediction Status"]}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
          <p className="results-disclaimer">
            Model predictions are intended to support review and should not be
            treated as a standalone fraud determination.
          </p>
        </>
      )}
    </section>
  );
}
