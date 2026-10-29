import { Activity, AlertTriangle, CheckCircle2, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { Legend, axisProps, tooltipProps } from "../components/charts";
import { Button, Card, EmptyState, Hint, PageHeader, Segmented, Stat } from "../components/ui";
import { CHART, fmt } from "../lib/format";
import { useCard, useChampion, useStore } from "../store";
import type { DriftReport, ServiceMetrics } from "../types";

const STATUS_META = {
  stable: { label: "Stable", color: CHART.approve, icon: CheckCircle2 },
  moderate: { label: "Moderate drift", color: CHART.review, icon: AlertTriangle },
  significant: { label: "Significant drift", color: CHART.block, icon: AlertTriangle },
} as const;

const SCENARIO_LABELS: Record<string, string> = {
  baseline: "Baseline",
  holiday_surge: "Holiday surge",
  fraud_ring: "Fraud ring",
  schema_shift: "Pipeline bug",
  last_batch: "Last batch",
};

export function Monitoring() {
  const { lastBatch, status } = useStore();
  const champion = useChampion();
  const card = useCard(champion?.version);
  const [scenarios, setScenarios] = useState<Record<string, string>>({});
  const [scenario, setScenario] = useState("baseline");
  const [report, setReport] = useState<DriftReport | null>(null);
  const [feature, setFeature] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [metrics, setMetrics] = useState<ServiceMetrics | null>(null);

  useEffect(() => {
    if (status === "online") api.driftScenarios().then(setScenarios).catch(() => undefined);
  }, [status]);

  const run = useCallback(async (name: string) => {
    setBusy(true);
    setError("");
    try {
      const result = name === "last_batch" && lastBatch
        ? await api.drift(lastBatch.inputs)
        : await api.driftSimulate(name);
      setReport(result);
      setFeature(result.features[0]?.feature ?? null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Drift check failed.");
    } finally {
      setBusy(false);
    }
  }, [lastBatch]);

  useEffect(() => {
    if (status === "online") run(scenario);
  }, [scenario, status, run]);

  useEffect(() => {
    if (status !== "online") return;
    const load = () => api.metrics().then(setMetrics).catch(() => undefined);
    load();
    const timer = window.setInterval(load, 5000);
    return () => window.clearInterval(timer);
  }, [status]);

  const options = [...Object.keys(scenarios), ...(lastBatch ? ["last_batch"] : [])]
    .map((value) => ({ value, label: SCENARIO_LABELS[value] ?? value }));
  const detail = report?.features.find((f) => f.feature === feature);
  const expectedFlag = card ? card.decision_mix.review + card.decision_mix.block : null;
  const meta = report ? STATUS_META[report.status] : null;

  return (
    <div className="page">
      <PageHeader
        title="Monitoring"
        description="Population Stability Index (PSI) of every input and of the model's own score, against the training distribution — plus live service health."
      />

      <div className="control-bar">
        <div className="control grow">
          <span className="control-label">Production window</span>
          {options.length ? (
            <Segmented label="Drift scenario" value={scenario} onChange={setScenario} options={options} />
          ) : <span className="skeleton" style={{ height: 32, width: 360 }} />}
        </div>
        <Button icon={<RefreshCw size={15} />} onClick={() => run(scenario)} loading={busy}>Re-run check</Button>
      </div>
      {(report?.description || scenarios[scenario]) && (
        <p className="scenario-note">{scenario === "last_batch" ? `Comparing “${lastBatch?.name}” against training data.` : scenarios[scenario]}</p>
      )}
      {error && <p className="error-text">{error}</p>}

      {report && meta ? (
        <>
          <div className="status-banner" style={{ borderColor: meta.color }}>
            <meta.icon size={22} style={{ color: meta.color }} />
            <div>
              <strong style={{ color: meta.color }}>{meta.label}</strong>
              <span className="muted">
                {report.counts.significant} significant · {report.counts.moderate} moderate · {report.counts.stable} stable across {report.features_checked} signals ({fmt.int(report.rows)} transactions)
              </span>
            </div>
            <div className="banner-metric">
              <span className="muted small">Alert rate</span>
              <strong>{fmt.pct(report.flag_rate, 2)}</strong>
              {expectedFlag !== null && <span className="muted small">expected {fmt.pct(expectedFlag, 2)}</span>}
            </div>
          </div>

          <div className="grid grid-2">
            <Card title="Drift by signal" subtitle="PSI · 0.10 moderate, 0.25 significant · click a bar">
              <div className="chart-box" style={{ height: 330 }}>
                <ResponsiveContainer>
                  <BarChart data={report.features.slice(0, 14)} layout="vertical" margin={{ top: 0, right: 16, left: 6, bottom: 0 }}>
                    <CartesianGrid stroke={CHART.grid} horizontal={false} />
                    <XAxis type="number" {...axisProps} scale="sqrt" domain={[0, (max: number) => Math.max(0.3, max)]} tickFormatter={(v) => v.toFixed(2)} />
                    <YAxis type="category" dataKey="feature" {...axisProps} width={70} interval={0} />
                    <Tooltip {...tooltipProps} formatter={(v: number) => v.toFixed(4)} />
                    <ReferenceLine x={0.1} stroke={CHART.review} strokeDasharray="4 4" />
                    <ReferenceLine x={0.25} stroke={CHART.block} strokeDasharray="4 4" />
                    <Bar dataKey="psi" name="PSI" radius={[0, 4, 4, 0]} isAnimationActive={false}
                         onClick={(d: { feature?: string }) => d.feature && setFeature(d.feature)} cursor="pointer">
                      {report.features.slice(0, 14).map((f) => (
                        <Cell key={f.feature} fill={STATUS_META[f.status].color} fillOpacity={f.feature === feature ? 1 : 0.65} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>

            <Card title={detail ? `${detail.feature === "risk_score" ? "Model risk score" : detail.feature} distribution` : "Distribution"}
                  subtitle={detail ? `PSI ${detail.psi.toFixed(3)} · mean shift ${fmt.signed(detail.mean_shift)}σ · training deciles` : ""}>
              {detail ? (
                <>
                  <div className="chart-box" style={{ height: 290 }}>
                    <ResponsiveContainer>
                      <BarChart data={detail.expected.map((e, i) => ({ bin: `D${i + 1}`, expected: e, actual: detail.actual[i] }))}
                                margin={{ top: 8, right: 4, left: -12, bottom: 0 }}>
                        <CartesianGrid stroke={CHART.grid} vertical={false} />
                        <XAxis dataKey="bin" {...axisProps} />
                        <YAxis {...axisProps} tickFormatter={(v) => fmt.pct(v, 0)} />
                        <Tooltip {...tooltipProps} formatter={(v: number) => fmt.pct(v, 1)} />
                        <Bar dataKey="expected" name="Training" fill="rgba(148,163,184,0.45)" radius={[3, 3, 0, 0]} isAnimationActive={false} />
                        <Bar dataKey="actual" name="Current" fill={STATUS_META[detail.status].color} radius={[3, 3, 0, 0]} isAnimationActive={false} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <Legend items={[{ label: "Training reference", color: "rgba(148,163,184,0.6)" }, { label: "Current window", color: STATUS_META[detail.status].color }]} />
                </>
              ) : <EmptyState icon={<Activity size={20} />} title="Select a signal" />}
            </Card>
          </div>

          <Card title="All signals" pad={false}>
            <div className="table-scroll" style={{ maxHeight: 360 }}>
              <table className="table compact">
                <thead><tr><th>Signal</th><th className="num">PSI</th><th className="num">Mean shift (σ)</th><th>Status</th></tr></thead>
                <tbody>
                  {report.features.map((f) => (
                    <tr key={f.feature} className={f.feature === feature ? "row-highlight clickable" : "clickable"} onClick={() => setFeature(f.feature)}>
                      <td className="mono">{f.feature === "risk_score" ? "risk_score (model output)" : f.feature}</td>
                      <td className="num mono">{f.psi.toFixed(4)}</td>
                      <td className="num mono">{fmt.signed(f.mean_shift)}</td>
                      <td><span className="status-text" style={{ color: STATUS_META[f.status].color }}>● {STATUS_META[f.status].label}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      ) : (
        !error && <div className="grid grid-2"><span className="skeleton" style={{ height: 330 }} /><span className="skeleton" style={{ height: 330 }} /></div>
      )}

      <Card title="Service health" subtitle="Live from /api/metrics · refreshes every 5 seconds">
        {metrics ? (
          <>
            <div className="stat-grid">
              <Stat label="Uptime" value={fmt.duration(metrics.uptime_seconds)} />
              <Stat label="Requests" value={fmt.int(metrics.requests_total)} />
              <Stat label="Transactions scored" value={fmt.int(metrics.transactions_scored)} />
              <Stat label="Latency p50" value={fmt.ms(metrics.latency_ms.p50)} hint={`p95 ${fmt.ms(metrics.latency_ms.p95)} · p99 ${fmt.ms(metrics.latency_ms.p99)}`} />
              <Stat label="Server errors" tone={metrics.errors_total ? "block" : undefined} value={fmt.int(metrics.errors_total)} />
              <Stat label="Decisions" value={`${fmt.int(metrics.decisions.block)} / ${fmt.int(metrics.decisions.review)}`} hint="blocked / reviewed" />
            </div>
            <Hint>Counters are in-process and reset when the API restarts. Prediction latency includes reason-code computation.</Hint>
          </>
        ) : <span className="skeleton" style={{ height: 90 }} />}
      </Card>
    </div>
  );
}
