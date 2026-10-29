import { AnimatePresence, motion } from "framer-motion";
import { GitCompareArrows, X } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api";
import { DECISION_META, fmt, narrative } from "../lib/format";
import { useStore } from "../store";
import type { Comparison } from "../types";
import { ReasonBars, RiskGauge } from "./charts";
import { Button, DecisionBadge, KeyValue, RiskBadge, StageBadge, TruthBadge, VersionTag } from "./ui";

export function TransactionDrawer() {
  const { selected, inspect } = useStore();
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [comparing, setComparing] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setComparison(null);
    setError("");
    if (!selected) return;
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && inspect(null);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected, inspect]);

  const compare = async () => {
    if (!selected) return;
    setComparing(true);
    setError("");
    try {
      setComparison(await api.compare(selected.input));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Comparison failed.");
    } finally {
      setComparing(false);
    }
  };

  const p = selected?.prediction;
  const components = selected
    ? Object.entries(selected.input).filter(([key]) => /^V\d+$/.test(key))
    : [];

  return (
    <AnimatePresence>
      {selected && p && (
        <>
          <motion.div className="drawer-backdrop" onClick={() => inspect(null)}
                      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} />
          <motion.aside
            className="drawer"
            role="dialog"
            aria-modal="true"
            aria-label="Transaction details"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 32, stiffness: 320 }}
          >
            <div className="drawer-head">
              <div>
                <p className="eyebrow">Transaction investigation</p>
                <h2 className="mono">{selected.input.transaction_id ?? p.transaction_id ?? "Manual entry"}</h2>
              </div>
              <button className="icon-btn" type="button" onClick={() => inspect(null)} aria-label="Close">
                <X size={18} />
              </button>
            </div>

            <div className="drawer-body">
              <div className="drawer-verdict" style={{ borderColor: DECISION_META[p.decision].color }}>
                <RiskGauge score={p.risk_score} size={200} />
                <div className="verdict-side">
                  <DecisionBadge decision={p.decision} />
                  <div className="verdict-line"><span>Risk level</span><RiskBadge level={p.risk_level} /></div>
                  <div className="verdict-line"><span>Fraud probability</span><strong className="mono">{fmt.prob(p.fraud_probability)}</strong></div>
                  <div className="verdict-line"><span>Ground truth</span><TruthBadge actual={selected.actual} /></div>
                  <div className="verdict-line"><span>Model</span><VersionTag version={p.model_version} /></div>
                </div>
              </div>

              <section className="drawer-section">
                <h3>Why this decision</h3>
                <p className="narrative">{narrative(p.reasons)}</p>
                <ReasonBars reasons={p.reasons} />
                <p className="muted small">
                  Bars show how far each input moves fraud log-odds compared with a typical transaction.
                </p>
              </section>

              <section className="drawer-section">
                <div className="section-row">
                  <h3>Champion vs challengers</h3>
                  <Button icon={<GitCompareArrows size={15} />} onClick={compare} loading={comparing}>
                    {comparison ? "Re-run" : "Compare versions"}
                  </Button>
                </div>
                {error && <p className="error-text">{error}</p>}
                {comparison && (
                  <>
                    <table className="table compact">
                      <thead>
                        <tr><th>Version</th><th>Stage</th><th className="num">Score</th><th>Decision</th></tr>
                      </thead>
                      <tbody>
                        {comparison.results.map((r) => (
                          <tr key={r.version}>
                            <td><VersionTag version={r.version} /> <span className="muted small">{r.name}</span></td>
                            <td><StageBadge stage={r.stage} /></td>
                            <td className="num mono">{r.risk_score.toFixed(1)}</td>
                            <td><DecisionBadge decision={r.decision} size="sm" /></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p className="muted small">
                      {comparison.agreement ? "All versions agree on this decision." : "Versions disagree — a typical case for champion/challenger review."}
                    </p>
                  </>
                )}
              </section>

              <section className="drawer-section">
                <h3>Transaction</h3>
                <KeyValue items={[
                  ["Amount", fmt.money(Number(selected.input.Amount))],
                  ["Time of day", fmt.hour(Number(selected.input.Time))],
                  ["Day", `Day ${Math.floor(Number(selected.input.Time) / 86400) + 1}`],
                  ["Thresholds", `review ≥ ${fmt.prob(p.thresholds.review)} · block ≥ ${fmt.prob(p.thresholds.block)}`],
                  ["Server latency", fmt.ms(p.latency_ms)],
                  ["Round trip", fmt.ms(selected.roundTripMs)],
                ]} />
                <div className="component-grid">
                  {components.map(([key, value]) => (
                    <div key={key}><span>{key}</span><strong className="mono">{Number(value).toFixed(2)}</strong></div>
                  ))}
                </div>
              </section>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
