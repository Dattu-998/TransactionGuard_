import { Braces, Dices, FlaskConical, RotateCcw, ScanSearch, ShieldAlert, Sparkles } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { ReasonBars, RiskGauge } from "../components/charts";
import {
  Button,
  Card,
  DecisionBadge,
  EmptyState,
  Hint,
  KeyValue,
  PageHeader,
  RiskBadge,
  StageBadge,
  TruthBadge,
  VersionTag,
} from "../components/ui";
import { DECISION_META, fmt, narrative, riskColor } from "../lib/format";
import { useStore } from "../store";
import type { Comparison, Prediction, Transaction } from "../types";

const COMPONENTS = Array.from({ length: 28 }, (_, i) => `V${i + 1}`);

export function Investigate() {
  const { schema, scoringModel, addEvent, status } = useStore();
  const [draft, setDraft] = useState<Transaction | null>(null);
  const [actual, setActual] = useState<0 | 1 | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [jsonMode, setJsonMode] = useState(false);
  const [jsonText, setJsonText] = useState("");

  const typical = useMemo<Transaction | null>(() => {
    if (!schema) return null;
    const row: Transaction = { transaction_id: "MANUAL-001", Time: 0, Amount: 0 };
    for (const name of schema.features) row[name] = Number(schema.statistics[name].median.toFixed(4));
    return row;
  }, [schema]);

  useEffect(() => {
    if (typical && !draft) setDraft(typical);
  }, [typical, draft]);

  const score = useCallback(async (transaction: Transaction, label: 0 | 1 | null) => {
    setBusy(true);
    setError("");
    try {
      const started = performance.now();
      const [result, compared] = await Promise.all([
        api.predict(transaction, scoringModel, true),
        api.compare(transaction),
      ]);
      setPrediction(result);
      setComparison(compared);
      addEvent({ source: "manual", input: transaction, actual: label, prediction: result,
                 roundTripMs: performance.now() - started });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Scoring failed.");
    } finally {
      setBusy(false);
    }
  }, [scoringModel, addEvent]);

  const loadSample = async (fraud: boolean) => {
    setBusy(true);
    setError("");
    try {
      const [sample] = await api.samples(1, fraud ? 1 : 0, Math.floor(Math.random() * 1e9));
      const label = sample.Class === 1 ? 1 : 0;
      const { Class: _ignored, ...rest } = sample;
      setDraft(rest as Transaction);
      setActual(label);
      await score(rest as Transaction, label);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load a sample.");
      setBusy(false);
    }
  };

  const update = (name: string, raw: string) => {
    if (!draft) return;
    setDraft({ ...draft, [name]: raw === "" ? NaN : Number(raw) });
    setActual(null);
  };

  const applyJson = () => {
    try {
      const parsed = JSON.parse(jsonText) as Transaction;
      setDraft({ ...(draft ?? {}), ...parsed } as Transaction);
      setActual(typeof parsed.Class === "number" ? (parsed.Class ? 1 : 0) : null);
      setJsonMode(false);
      setError("");
    } catch {
      setError("That is not valid JSON.");
    }
  };

  const hour = draft ? (Number(draft.Time) / 3600) % 24 : 12;
  const day = draft ? Math.floor(Number(draft.Time) / 86400) : 0;
  const invalid = draft ? Object.entries(draft).some(([k, v]) => k !== "transaction_id" && typeof v === "number" && !Number.isFinite(v)) : true;

  return (
    <div className="page">
      <PageHeader
        eyebrow="Operations"
        title="Investigate"
        description="Build or load a transaction, score it with reason codes, and see how every model version in the registry would decide."
        actions={
          <>
            <Button icon={<Dices size={15} />} onClick={() => loadSample(false)} disabled={status !== "online" || busy}>Random legitimate</Button>
            <Button icon={<ShieldAlert size={15} />} onClick={() => loadSample(true)} disabled={status !== "online" || busy}>Random fraud</Button>
          </>
        }
      />

      <div className="grid grid-investigate">
        <Card
          title="Transaction builder"
          subtitle={actual === null ? "Edited or manual input" : <>Loaded from the held-out test set · truth: <TruthBadge actual={actual} /></>}
          actions={
            <>
              <Button variant="ghost" icon={<Braces size={15} />} onClick={() => {
                setJsonText(JSON.stringify(draft, null, 2));
                setJsonMode((m) => !m);
              }}>JSON</Button>
              <Button variant="ghost" icon={<RotateCcw size={15} />} onClick={() => { setDraft(typical); setActual(null); }}>Reset</Button>
            </>
          }
        >
          {!draft || !schema ? (
            <div className="stack">{Array.from({ length: 6 }, (_, i) => <span key={i} className="skeleton" style={{ height: 34 }} />)}</div>
          ) : jsonMode ? (
            <div className="stack">
              <textarea className="code-input" value={jsonText} onChange={(e) => setJsonText(e.target.value)} rows={18} spellCheck={false} aria-label="Transaction JSON" />
              <Button variant="primary" onClick={applyJson}>Apply JSON</Button>
            </div>
          ) : (
            <form className="builder" onSubmit={(e) => { e.preventDefault(); score(draft, actual); }}>
              <div className="builder-top">
                <label className="field">
                  <span>Transaction ID</span>
                  <input value={String(draft.transaction_id ?? "")} onChange={(e) => setDraft({ ...draft, transaction_id: e.target.value })} />
                </label>
                <label className="field">
                  <span>Amount (USD)</span>
                  <input type="number" step="0.01" min={0} value={Number.isFinite(draft.Amount) ? draft.Amount : ""}
                         onChange={(e) => update("Amount", e.target.value)} />
                </label>
              </div>
              <label className="field">
                <span>Time of day · {fmt.hour(Number(draft.Time))} on day {day + 1}</span>
                <input type="range" min={0} max={23.98} step={0.25} value={hour}
                       onChange={(e) => { setDraft({ ...draft, Time: Math.round(day * 86400 + Number(e.target.value) * 3600) }); setActual(null); }} />
              </label>

              <div className="builder-heading">
                <span>Behavioral components (PCA)</span>
                <span className="muted small">bar = value within training p1–p99</span>
              </div>
              <div className="component-inputs">
                {COMPONENTS.map((name) => {
                  const s = schema.statistics[name];
                  const value = Number(draft[name]);
                  const position = Number.isFinite(value) ? Math.min(1, Math.max(0, (value - s.p01) / (s.p99 - s.p01))) : 0.5;
                  const extreme = value < s.p01 || value > s.p99;
                  return (
                    <label key={name} className={extreme ? "component-input extreme" : "component-input"}>
                      <span>{name}</span>
                      <input type="number" step="0.1" value={Number.isFinite(value) ? value : ""}
                             onChange={(e) => update(name, e.target.value)} />
                      <i className="range"><b style={{ left: `${position * 100}%` }} /></i>
                    </label>
                  );
                })}
              </div>
              <Button type="submit" variant="primary" icon={<ScanSearch size={15} />} loading={busy} disabled={invalid}>
                Score transaction
              </Button>
              {invalid && <p className="error-text">Every field needs a numeric value.</p>}
            </form>
          )}
          {error && <p className="error-text">{error}</p>}
        </Card>

        <div className="stack-lg">
          <Card title="Verdict" subtitle={prediction ? <>Scored by <VersionTag version={prediction.model_version} /> in {fmt.ms(prediction.latency_ms)}</> : "Run a score to see the decision"}>
            {prediction ? (
              <div className="verdict" style={{ borderColor: DECISION_META[prediction.decision].color }}>
                <RiskGauge score={prediction.risk_score} />
                <div className="verdict-side">
                  <div className="verdict-decision" style={{ color: DECISION_META[prediction.decision].color }}>
                    {DECISION_META[prediction.decision].verb}
                  </div>
                  <KeyValue items={[
                    ["Risk level", <RiskBadge key="r" level={prediction.risk_level} />],
                    ["Fraud probability", <span key="p" className="mono">{fmt.prob(prediction.fraud_probability)}</span>],
                    ["Review threshold", <span key="t" className="mono">{fmt.prob(prediction.thresholds.review)}</span>],
                    ["Block threshold", <span key="b" className="mono">{fmt.prob(prediction.thresholds.block)}</span>],
                  ]} />
                </div>
              </div>
            ) : (
              <EmptyState icon={<FlaskConical size={22} />} title="Nothing scored yet">
                Load a random legitimate or fraudulent transaction, or edit the inputs and score them.
              </EmptyState>
            )}
          </Card>

          {prediction && (
            <Card title="Reason codes" subtitle="What moved the score, relative to a typical transaction">
              <p className="narrative"><Sparkles size={14} /> {narrative(prediction.reasons)}</p>
              <ReasonBars reasons={prediction.reasons} />
              <Hint>Attribution replaces one input at a time with its training median and measures the change in fraud log-odds. It works identically for every model version.</Hint>
            </Card>
          )}

          {comparison && (
            <Card title="Champion vs challengers" subtitle={comparison.agreement ? "All versions agree" : "Versions disagree on this transaction"} pad={false}>
              <table className="table">
                <thead><tr><th>Version</th><th>Stage</th><th>Probability</th><th>Score</th><th>Decision</th></tr></thead>
                <tbody>
                  {comparison.results.map((r) => (
                    <tr key={r.version} className={r.version === comparison.champion ? "row-highlight" : ""}>
                      <td><VersionTag version={r.version} /> <span className="small">{r.name}</span></td>
                      <td><StageBadge stage={r.stage} /></td>
                      <td className="mono">{fmt.prob(r.fraud_probability)}</td>
                      <td className="mono" style={{ color: riskColor(r.risk_score) }}>{r.risk_score.toFixed(1)}</td>
                      <td><DecisionBadge decision={r.decision} size="sm" /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
