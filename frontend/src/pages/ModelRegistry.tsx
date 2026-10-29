import { ArrowRight, Boxes, GitBranch, SlidersHorizontal } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { Legend, axisProps, tooltipProps } from "../components/charts";
import { Card, Hint, KeyValue, PageHeader, Segmented, StageBadge, Stat, VersionTag } from "../components/ui";
import { CHART, fmt, versionColor } from "../lib/format";
import { useStore } from "../store";
import type { ModelCard } from "../types";

export function ModelRegistry() {
  const { registry, cards, loadCard, setScoringModel, scoringModel } = useStore();
  const [selected, setSelected] = useState<string | null>(null);
  const [curve, setCurve] = useState<"pr" | "roc" | "calibration">("pr");

  useEffect(() => {
    registry?.versions.forEach((v) => loadCard(v.version));
    if (registry && !selected) setSelected(registry.champion);
  }, [registry, loadCard, selected]);

  const versions = registry?.versions ?? [];
  const loaded = versions.map((v) => cards[v.version]).filter(Boolean) as ModelCard[];
  const card = selected ? cards[selected] : null;
  const baseline = versions[0];

  if (!registry) {
    return (
      <div className="page">
        <PageHeader title="Model Registry" description="Loading registry…" />
        <div className="grid grid-3">{[0, 1, 2].map((i) => <span key={i} className="skeleton" style={{ height: 220 }} />)}</div>
      </div>
    );
  }

  return (
    <div className="page">
      <PageHeader
        title="Model Registry"
        description={<>Every version is trained on the same split and evaluated on the same untouched test set. The champion is promoted by validation PR-AUC — not by hand.</>}
      />

      <div className="lineage">
        {versions.map((v, index) => {
          const gain = baseline && index > 0 ? (v.metrics.pr_auc / baseline.metrics.pr_auc - 1) * 100 : null;
          return (
            <div key={v.version} className="lineage-item">
              <button
                type="button"
                className={`version-card ${selected === v.version ? "is-selected" : ""} ${v.stage === "champion" ? "is-champion" : ""}`}
                onClick={() => setSelected(v.version)}
              >
                <div className="version-card-top">
                  <VersionTag version={v.version} />
                  <StageBadge stage={v.stage} />
                </div>
                <h3>{v.name}</h3>
                <p className="muted small">{v.summary}</p>
                <div className="version-metrics">
                  <div><span>PR-AUC</span><strong>{v.metrics.pr_auc.toFixed(3)}</strong></div>
                  <div><span>Recall</span><strong>{fmt.pct(v.metrics.recall, 0)}</strong></div>
                  <div><span>Cost ↓</span><strong>{fmt.pct(v.metrics.cost_reduction, 0)}</strong></div>
                </div>
                <div className="chips">{v.highlights.map((h) => <span key={h} className="chip">{h}</span>)}</div>
                {gain !== null && <span className="gain">+{gain.toFixed(0)}% PR-AUC vs {baseline.version}</span>}
              </button>
              {index < versions.length - 1 && <ArrowRight className="lineage-arrow" size={18} />}
            </div>
          );
        })}
      </div>

      <Card title="Leaderboard" subtitle="Held-out test set · metrics at each version's own review threshold" pad={false}>
        <div className="table-scroll">
          <table className="table">
            <thead>
              <tr>
                <th>Version</th><th className="num">PR-AUC</th><th className="num">ROC-AUC</th><th className="num">Recall</th>
                <th className="num">Precision</th><th className="num">Brier</th><th className="num">Fraud $ caught</th>
                <th className="num">Cost reduction</th><th className="num">Latency p50</th><th className="num">Size</th><th />
              </tr>
            </thead>
            <tbody>
              {versions.map((v) => {
                const best = (key: keyof typeof v.metrics, lower = false) => {
                  const values = versions.map((x) => x.metrics[key]);
                  return v.metrics[key] === (lower ? Math.min(...values) : Math.max(...values)) ? "best" : "";
                };
                const serving = (scoringModel ?? registry.champion) === v.version;
                return (
                  <tr key={v.version} className={selected === v.version ? "row-highlight clickable" : "clickable"} onClick={() => setSelected(v.version)}>
                    <td><VersionTag version={v.version} /> <span className="small">{v.name}</span></td>
                    <td className={`num mono ${best("pr_auc")}`}>{v.metrics.pr_auc.toFixed(4)}</td>
                    <td className={`num mono ${best("roc_auc")}`}>{v.metrics.roc_auc.toFixed(4)}</td>
                    <td className={`num mono ${best("recall")}`}>{fmt.pct(v.metrics.recall)}</td>
                    <td className={`num mono ${best("precision")}`}>{fmt.pct(v.metrics.precision)}</td>
                    <td className={`num mono ${best("brier", true)}`}>{v.metrics.brier.toFixed(5)}</td>
                    <td className={`num mono ${best("amount_recall")}`}>{fmt.pct(v.metrics.amount_recall)}</td>
                    <td className={`num mono ${best("cost_reduction")}`}>{fmt.pct(v.metrics.cost_reduction)}</td>
                    <td className={`num mono ${best("latency_p50_ms", true)}`}>{fmt.ms(v.metrics.latency_p50_ms)}</td>
                    <td className="num mono muted">{fmt.bytes(v.artifact_bytes)}</td>
                    <td>
                      <button type="button" className={serving ? "link-btn active" : "link-btn"}
                              onClick={(e) => { e.stopPropagation(); setScoringModel(v.version === registry.champion ? undefined : v.version); }}>
                        {serving ? "Scoring" : "Route traffic"}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid grid-2">
        <Card
          title={curve === "pr" ? "Precision–recall" : curve === "roc" ? "ROC (log FPR)" : "Calibration"}
          subtitle={curve === "pr" ? "The metric that matters at 0.2% prevalence" : curve === "roc" ? "True-positive rate vs false-positive rate" : "Predicted probability vs observed fraud rate"}
          actions={<Segmented label="Curve" value={curve} onChange={setCurve} options={[
            { value: "pr", label: "PR" }, { value: "roc", label: "ROC" }, { value: "calibration", label: "Calibration" },
          ]} />}
        >
          <CurveChart cards={loaded} curve={curve} />
          <Legend items={loaded.map((c) => ({ label: `${c.version} · ${c.name}`, color: versionColor(c.version) }))} />
        </Card>
        {card && <ThresholdExplorer card={card} />}
      </div>

      {card && (
        <div className="grid grid-2">
          <Card title="What the model looks at" subtitle={`${card.version} · permutation importance (drop in PR-AUC when shuffled)`}>
            <div className="chart-box" style={{ height: 320 }}>
              <ResponsiveContainer>
                <BarChart data={card.importance.slice(0, 12)} layout="vertical" margin={{ top: 0, right: 16, left: 4, bottom: 0 }}>
                  <CartesianGrid stroke={CHART.grid} horizontal={false} />
                  <XAxis type="number" {...axisProps} tickFormatter={(v) => v.toFixed(2)} />
                  <YAxis type="category" dataKey="feature" {...axisProps} width={58} interval={0} />
                  <Tooltip {...tooltipProps} formatter={(v: number) => v.toFixed(4)} />
                  <Bar dataKey="importance" name="Importance" fill={versionColor(card.version)} radius={[0, 4, 4, 0]} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
          <Card title="Score separation" subtitle="Share of legitimate vs fraudulent test transactions per risk band">
            <div className="chart-box" style={{ height: 290 }}>
              <ResponsiveContainer>
                <BarChart data={card.curves.score_histogram} margin={{ top: 8, right: 4, left: -12, bottom: 0 }} barGap={0}>
                  <CartesianGrid stroke={CHART.grid} vertical={false} />
                  <XAxis dataKey="start" {...axisProps} />
                  <YAxis {...axisProps} tickFormatter={(v) => fmt.pct(v, 0)} />
                  <Tooltip {...tooltipProps} formatter={(v: number) => fmt.pct(v, 2)} labelFormatter={(s) => `Score ${s}–${Number(s) + 5}`} />
                  <ReferenceLine x={50} stroke={CHART.review} strokeDasharray="4 4" />
                  <ReferenceLine x={80} stroke={CHART.block} strokeDasharray="4 4" />
                  <Bar dataKey="legit_share" name="Legitimate" fill={CHART.legit} isAnimationActive={false} />
                  <Bar dataKey="fraud_share" name="Fraud" fill={CHART.fraud} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <Legend items={[{ label: "Legitimate", color: CHART.legit }, { label: "Fraud", color: CHART.fraud }]} />
          </Card>
        </div>
      )}

      {card && <ModelCardDetails card={card} />}
    </div>
  );
}

function CurveChart({ cards, curve }: { cards: ModelCard[]; curve: "pr" | "roc" | "calibration" }) {
  if (!cards.length) return <span className="skeleton" style={{ height: 300 }} />;
  if (curve === "calibration") {
    return (
      <div className="chart-box" style={{ height: 300 }}>
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 8, right: 12, left: -6, bottom: 0 }}>
            <CartesianGrid stroke={CHART.grid} />
            <XAxis type="number" dataKey="predicted" name="Predicted" scale="log" domain={[1e-5, 1]} {...axisProps}
                   tickFormatter={(v) => fmt.prob(v)} ticks={[1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]} allowDataOverflow />
            <YAxis type="number" dataKey="observed" name="Observed" scale="log" domain={[1e-5, 1]} {...axisProps}
                   tickFormatter={(v) => fmt.prob(v)} ticks={[1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]} allowDataOverflow />
            <ZAxis type="number" dataKey="count" range={[30, 260]} />
            <Tooltip {...tooltipProps} formatter={(v: number, name: string) => (name === "count" ? fmt.int(v) : fmt.prob(v))} />
            <ReferenceLine segment={[{ x: 1e-5, y: 1e-5 }, { x: 1, y: 1 }]} stroke="rgba(148,163,184,0.4)" strokeDasharray="4 4" />
            {cards.map((c) => (
              <Scatter key={c.version} name={c.version} data={c.curves.calibration.map((p) => ({ ...p, observed: Math.max(p.observed, 1e-5) }))}
                       fill={versionColor(c.version)} line={{ stroke: versionColor(c.version), strokeWidth: 1.5 }} isAnimationActive={false} />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    );
  }
  const x = curve === "pr" ? "recall" : "fpr";
  const y = curve === "pr" ? "precision" : "tpr";
  return (
    <div className="chart-box" style={{ height: 300 }}>
      <ResponsiveContainer>
        <LineChart margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
          <CartesianGrid stroke={CHART.grid} />
          <XAxis type="number" dataKey={x} domain={curve === "pr" ? [0, 1] : [1e-5, 1]} scale={curve === "pr" ? "linear" : "log"}
                 ticks={curve === "pr" ? [0, 0.2, 0.4, 0.6, 0.8, 1] : [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]} allowDataOverflow
                 {...axisProps} tickFormatter={(v) => (curve === "pr" ? v.toFixed(1) : fmt.prob(v))} />
          <YAxis type="number" domain={[0, 1]} {...axisProps} tickFormatter={(v) => v.toFixed(1)} />
          <Tooltip {...tooltipProps} formatter={(v: number) => v.toFixed(3)} labelFormatter={(v) => `${x} ${Number(v).toPrecision(2)}`} />
          {cards.map((c) => (
            <Line key={c.version} data={(curve === "pr" ? c.curves.pr : c.curves.roc.filter((p) => p.fpr > 0)) as Array<Record<string, number>>}
                  dataKey={y} name={c.version} stroke={versionColor(c.version)} strokeWidth={c.stage === "champion" ? 2.6 : 1.6}
                  dot={false} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function ThresholdExplorer({ card }: { card: ModelCard }) {
  const sweep = card.curves.threshold_sweep;
  const policyIndex = useMemo(() => {
    let best = 0;
    sweep.forEach((p, i) => {
      if (Math.abs(Math.log(p.threshold) - Math.log(card.thresholds.review)) < Math.abs(Math.log(sweep[best].threshold) - Math.log(card.thresholds.review))) best = i;
    });
    return best;
  }, [sweep, card.thresholds.review]);
  const [index, setIndex] = useState(policyIndex);
  useEffect(() => setIndex(policyIndex), [policyIndex]);
  const data = useMemo(() => sweep.map((p) => ({ ...p, logT: Math.log10(p.threshold) })), [sweep]);
  const point = data[Math.min(index, data.length - 1)];

  return (
    <Card title="Threshold explorer" subtitle={`${card.version} · validation set · FN = ${fmt.money(card.costs.false_negative)}, FP = ${fmt.money(card.costs.false_positive)}`}
          actions={<SlidersHorizontal size={16} className="muted" />}>
      <div className="explorer-stats">
        <Stat label="Threshold" value={fmt.prob(point.threshold)} />
        <Stat label="Recall" tone="approve" value={fmt.pct(point.recall)} />
        <Stat label="Precision" tone="review" value={fmt.pct(point.precision)} />
        <Stat label="Alert rate" value={fmt.pct(point.flag_rate, 2)} />
        <Stat label="Expected cost" tone="block" value={fmt.moneyCompact(point.cost)} />
      </div>
      <input type="range" min={0} max={sweep.length - 1} value={index} onChange={(e) => setIndex(Number(e.target.value))}
             aria-label="Decision threshold" className="explorer-range" />
      <div className="chart-box" style={{ height: 190 }}>
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ top: 6, right: 8, left: -14, bottom: 0 }}>
            <CartesianGrid stroke={CHART.grid} />
            <XAxis dataKey="logT" type="number" domain={["dataMin", "dataMax"]} {...axisProps}
                   tickFormatter={(v) => fmt.prob(10 ** v)} />
            <YAxis yAxisId="rate" domain={[0, 1]} {...axisProps} tickFormatter={(v) => fmt.pct(v, 0)} />
            <YAxis yAxisId="cost" orientation="right" {...axisProps} tickFormatter={(v) => fmt.compact(v)} />
            <Tooltip {...tooltipProps} labelFormatter={(v) => `threshold ${fmt.prob(10 ** Number(v))}`}
                     formatter={(v: number, name: string) => (name === "Cost" ? fmt.money(v) : fmt.pct(v))} />
            <Line yAxisId="rate" dataKey="recall" name="Recall" stroke={CHART.approve} dot={false} strokeWidth={1.8} isAnimationActive={false} />
            <Line yAxisId="rate" dataKey="precision" name="Precision" stroke={CHART.review} dot={false} strokeWidth={1.8} isAnimationActive={false} />
            <Line yAxisId="cost" dataKey="cost" name="Cost" stroke={CHART.block} dot={false} strokeWidth={1.8} strokeDasharray="5 3" isAnimationActive={false} />
            <ReferenceLine yAxisId="rate" x={Math.log10(card.thresholds.review)} stroke="rgba(226,232,240,0.5)" strokeDasharray="3 3" label={{ value: "policy", fill: CHART.text, fontSize: 10, position: "insideTopLeft" }} />
            <ReferenceLine yAxisId="rate" x={point.logT} stroke={CHART.cyan} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <Legend items={[{ label: "Recall", color: CHART.approve }, { label: "Precision", color: CHART.review }, { label: "Expected cost", color: CHART.block, dashed: true }]} />
      <Hint>The policy line is the cost-optimal review threshold chosen on validation data; scores above {fmt.prob(card.thresholds.block)} are auto-blocked ({fmt.pct(card.test_block.precision, 0)} precision on test).</Hint>
    </Card>
  );
}

function ModelCardDetails({ card }: { card: ModelCard }) {
  return (
    <div className="grid grid-3">
      <Card title="Model card" subtitle={`${card.version} · ${card.family}`} actions={<Boxes size={16} className="muted" />}>
        <KeyValue items={[
          ["Trained", fmt.date(card.created_at)],
          ["Training time", fmt.duration(card.training.seconds)],
          ["Training rows", `${fmt.int(card.training.rows)} (${fmt.int(card.training.fraud_rows)} fraud)`],
          ["Review threshold", fmt.prob(card.thresholds.review)],
          ["Block threshold", `${fmt.prob(card.thresholds.block)} (≥${fmt.pct(card.thresholds.block_precision_target ?? 0.9, 0)} precision)`],
          ["Artifact", fmt.bytes(card.artifact_bytes)],
          ["Batch throughput", `${fmt.compact(card.latency.batch_throughput_per_sec)} tx/s`],
        ]} />
      </Card>
      <Card title="Hyperparameters" actions={<GitBranch size={16} className="muted" />}>
        <KeyValue items={Object.entries(card.params).map(([k, v]) => [k, <span key={k} className="mono">{String(v)}</span>])} />
      </Card>
      <Card title="Data & lineage">
        <KeyValue items={[
          ["Source", card.dataset.source === "synthetic" ? "Synthetic benchmark" : card.dataset.source],
          ["Rows", `${fmt.int(card.dataset.rows)} · ${fmt.pct(card.dataset.fraud_rate, 3)} fraud`],
          ...Object.entries(card.dataset.splits).map(([k, s]) => [`${k[0].toUpperCase()}${k.slice(1)} split`, `${fmt.int(s.rows)} (${s.fraud_rows} fraud)`] as [string, string]),
          ["Data fingerprint", <span key="fp" className="mono">{card.dataset.fingerprint}</span>],
          ["Code commit", <span key="c" className="mono">{card.environment.git_commit ?? "—"}</span>],
          ["scikit-learn", card.environment.scikit_learn],
        ]} />
        <p className="muted small">{card.dataset.description}</p>
      </Card>
    </div>
  );
}
