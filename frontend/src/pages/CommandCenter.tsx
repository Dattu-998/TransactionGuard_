import { Gauge, Pause, Play, Radar } from "lucide-react";
import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Legend, axisProps, tooltipProps } from "../components/charts";
import {
  Button,
  Card,
  DecisionBadge,
  EmptyState,
  PageHeader,
  ScoreBar,
  StageBadge,
  Stat,
  TruthBadge,
  VersionTag,
} from "../components/ui";
import { CHART, DECISION_META, fmt, riskColor } from "../lib/format";
import { sessionStats, useCard, useChampion, useStore } from "../store";
import type { Route } from "../types";

const BUCKET_SECONDS = 10;
const BUCKETS = 30;

export function CommandCenter({ go }: { go: (route: Route) => void }) {
  const { events, live, inspect, registry } = useStore();
  const champion = useChampion();
  const card = useCard(champion?.version);
  const stats = useMemo(() => sessionStats(events), [events]);

  const timeline = useMemo(() => {
    const now = Math.floor(Date.now() / 1000 / BUCKET_SECONDS);
    const buckets = Array.from({ length: BUCKETS }, (_, i) => ({
      t: (now - BUCKETS + 1 + i) * BUCKET_SECONDS * 1000, approve: 0, review: 0, block: 0,
    }));
    for (const event of events) {
      const index = Math.floor(event.at / 1000 / BUCKET_SECONDS) - (now - BUCKETS + 1);
      if (index >= 0 && index < BUCKETS) buckets[index][event.prediction.decision] += 1;
    }
    return buckets;
  }, [events]);

  const distribution = useMemo(() => {
    const bins = Array.from({ length: 10 }, (_, i) => ({ bucket: `${i * 10}`, count: 0, start: i * 10 }));
    for (const event of events) bins[Math.min(9, Math.floor(event.prediction.risk_score / 10))].count += 1;
    return bins;
  }, [events]);

  const mix = [
    { name: "Approve", value: stats.approve, color: CHART.approve },
    { name: "Review", value: stats.review, color: CHART.review },
    { name: "Block", value: stats.block, color: CHART.block },
  ];
  const alerts = events.filter((e) => e.prediction.decision !== "approve").slice(0, 7);

  return (
    <div className="page">
      <PageHeader
        title="Command Center"
        description="Real-time view of screening decisions, alert pressure and the model in production."
        actions={
          live.running ? (
            <Button icon={<Pause size={15} />} onClick={live.stop}>Pause live feed</Button>
          ) : (
            <Button variant="primary" icon={<Play size={15} />} onClick={live.start}>Start live feed</Button>
          )
        }
      />

      <div className="stat-grid">
        <Stat label="Transactions screened" value={fmt.int(stats.total)}
              hint={`${fmt.moneyCompact(stats.amount)} processed this session`} />
        <Stat label="Auto-blocked" tone="block" value={fmt.int(stats.block)}
              hint={stats.total ? `${fmt.pct(stats.block / stats.total)} of traffic` : "High-confidence fraud"} />
        <Stat label="In review queue" tone="review" value={fmt.int(stats.review)}
              hint={stats.total ? `${fmt.pct(stats.review / stats.total)} of traffic` : "Analyst workload"} />
        <Stat label="Value held" tone="accent" value={fmt.moneyCompact(stats.amountStopped)}
              hint="Blocked or pending review" />
        <Stat label="Detection rate" tone="approve"
              value={stats.detectionRate === null ? "—" : fmt.pct(stats.detectionRate)}
              hint={stats.actualFraud ? `${stats.caught} of ${stats.actualFraud} known frauds stopped` : "Needs labelled traffic"} />
        <Stat label="Model latency" value={stats.total ? fmt.ms(stats.avgLatency) : card ? fmt.ms(card.latency.p50_ms) : "—"}
              hint={stats.total ? "Mean server time incl. reason codes" : "p50 benchmark at training"} />
      </div>

      <div className="grid grid-2-1">
        <Card
          title="Decision volume"
          subtitle={`Last ${(BUCKETS * BUCKET_SECONDS) / 60} minutes · ${BUCKET_SECONDS}s buckets`}
          actions={<Legend items={[
            { label: "Approve", color: CHART.approve },
            { label: "Review", color: CHART.review },
            { label: "Block", color: CHART.block },
          ]} />}
        >
          {events.length ? (
            <div className="chart-box" style={{ height: 240 }}>
              <ResponsiveContainer>
                <BarChart data={timeline} margin={{ top: 8, right: 8, left: -18, bottom: 0 }} barCategoryGap="22%">
                  <CartesianGrid stroke={CHART.grid} vertical={false} />
                  <XAxis dataKey="t" {...axisProps} tickFormatter={(t) => fmt.clock(t).slice(0, 5)} minTickGap={40} />
                  <YAxis {...axisProps} allowDecimals={false} />
                  <Tooltip {...tooltipProps} labelFormatter={(t) => fmt.clock(Number(t))} />
                  {(["approve", "review", "block"] as const).map((key) => (
                    <Bar key={key} dataKey={key} stackId="1" fill={CHART[key]} fillOpacity={0.85}
                         name={DECISION_META[key].label} isAnimationActive={false} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState icon={<Radar size={22} />} title="No traffic yet"
                        action={<Button variant="primary" icon={<Play size={15} />} onClick={live.start}>Start live feed</Button>}>
              Stream held-out transactions through the champion model to watch decisions arrive in real time.
            </EmptyState>
          )}
        </Card>

        <Card title="Decision mix" subtitle={stats.total ? `${fmt.int(stats.total)} decisions` : "Expected mix at the policy thresholds"}>
          <div className="donut-wrap">
            <div className="chart-box" style={{ height: 180 }}>
              <ResponsiveContainer>
                <PieChart>
                  <Pie
                    data={stats.total ? mix : card ? [
                      { name: "Approve", value: card.decision_mix.approve, color: CHART.approve },
                      { name: "Review", value: card.decision_mix.review, color: CHART.review },
                      { name: "Block", value: card.decision_mix.block, color: CHART.block },
                    ] : mix}
                    dataKey="value" innerRadius={56} outerRadius={78} paddingAngle={2} stroke="none"
                    isAnimationActive={false}
                  >
                    {mix.map((entry) => <Cell key={entry.name} fill={entry.color} />)}
                  </Pie>
                  <Tooltip {...tooltipProps} formatter={(v: number) => (stats.total ? fmt.int(v) : fmt.pct(v, 2))} />
                </PieChart>
              </ResponsiveContainer>
              <div className="donut-center">
                <strong>{stats.total ? fmt.pct((stats.review + stats.block) / stats.total) : card ? fmt.pct(card.decision_mix.review + card.decision_mix.block) : "—"}</strong>
                <span>flag rate</span>
              </div>
            </div>
            <ul className="mix-list">
              {(["approve", "review", "block"] as const).map((key) => (
                <li key={key}>
                  <DecisionBadge decision={key} size="sm" />
                  <span className="mono">
                    {stats.total ? fmt.int(stats[key]) : card ? fmt.pct(card.decision_mix[key], 2) : "—"}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </Card>
      </div>

      <div className="grid grid-3">
        <Card title="Latest alerts" subtitle="Review and block decisions" pad={false}
              actions={<Button variant="ghost" onClick={() => go("live")}>Open monitor</Button>}>
          {alerts.length ? (
            <ul className="alert-list">
              {alerts.map((event) => (
                <li key={event.key}>
                  <button type="button" onClick={() => inspect(event)}>
                    <span className="alert-dot" style={{ background: riskColor(event.prediction.risk_score) }} />
                    <span className="alert-main">
                      <strong className="mono">{event.input.transaction_id ?? "manual"}</strong>
                      <span className="muted small">{fmt.money(Number(event.input.Amount))} · {event.prediction.reasons[0]?.label.replace("Behavioral component", "Component") ?? "—"}</span>
                    </span>
                    <span className="alert-side">
                      <DecisionBadge decision={event.prediction.decision} size="sm" />
                      <TruthBadge actual={event.actual} />
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted small pad">Alerts appear here as risky transactions are scored.</p>
          )}
        </Card>

        <Card title="Production model" subtitle="Champion serving all traffic"
              actions={<Button variant="ghost" onClick={() => go("registry")}>Registry</Button>}>
          {champion && card ? (
            <div className="champion">
              <div className="champion-head">
                <VersionTag version={champion.version} />
                <div>
                  <strong>{champion.name}</strong>
                  <span className="muted small">{champion.family}</span>
                </div>
                <StageBadge stage="champion" />
              </div>
              <div className="metric-rows">
                <MetricRow label="PR-AUC (test)" value={card.test.pr_auc} display={card.test.pr_auc.toFixed(3)} />
                <MetricRow label="Recall at review threshold" value={card.test.recall} />
                <MetricRow label="Fraud $ caught" value={card.business.amount_recall} />
                <MetricRow label="Cost reduction vs no model" value={card.business.cost_reduction} />
                <MetricRow label="Block precision" value={card.test_block.precision} />
              </div>
              <p className="muted small">
                Trained {fmt.date(card.created_at)} on {fmt.compact(card.training.rows)} rows ·{" "}
                {registry?.versions.length ?? 0} versions in registry
              </p>
            </div>
          ) : (
            <div className="stack"><span className="skeleton" style={{ height: 120 }} /></div>
          )}
        </Card>

        <Card title="Session risk distribution" subtitle="Risk scores of everything screened">
          <div className="chart-box" style={{ height: 210 }}>
            <ResponsiveContainer>
              <BarChart data={distribution} margin={{ top: 8, right: 4, left: -22, bottom: 0 }}>
                <CartesianGrid stroke={CHART.grid} vertical={false} />
                <XAxis dataKey="bucket" {...axisProps} />
                <YAxis {...axisProps} allowDecimals={false} />
                <Tooltip {...tooltipProps} labelFormatter={(b) => `Score ${b}–${Number(b) + 10}`} />
                <Bar dataKey="count" name="Transactions" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                  {distribution.map((bin) => <Cell key={bin.bucket} fill={riskColor(bin.start + 5)} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="score-legend">
            <span><Gauge size={13} /> 50 = review threshold</span>
            <span>80 = auto-block</span>
          </div>
        </Card>
      </div>

      {events.length > 0 && (
        <Card title="Recent decisions" pad={false}>
          <table className="table">
            <thead>
              <tr><th>Time</th><th>Transaction</th><th className="num">Amount</th><th>Risk</th><th>Decision</th><th>Truth</th><th>Top driver</th></tr>
            </thead>
            <tbody>
              {events.slice(0, 6).map((event) => (
                <tr key={event.key} className="clickable" onClick={() => inspect(event)}>
                  <td className="mono muted">{fmt.clock(event.at)}</td>
                  <td className="mono">{event.input.transaction_id ?? "manual"}</td>
                  <td className="num mono">{fmt.money(Number(event.input.Amount))}</td>
                  <td><ScoreBar score={event.prediction.risk_score} /></td>
                  <td><DecisionBadge decision={event.prediction.decision} size="sm" /></td>
                  <td><TruthBadge actual={event.actual} /></td>
                  <td className="muted small">{event.prediction.reasons[0]?.label ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  );
}

function MetricRow({ label, value, display }: { label: string; value: number; display?: string }) {
  return (
    <div className="metric-row">
      <span>{label}</span>
      <span className="metric-bar"><i style={{ width: `${value * 100}%` }} /></span>
      <strong className="mono">{display ?? fmt.pct(value)}</strong>
    </div>
  );
}
