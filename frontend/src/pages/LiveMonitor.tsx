import { AnimatePresence, motion } from "framer-motion";
import { Pause, Play, Radio, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { MiniBars } from "../components/charts";
import {
  Button,
  Card,
  DecisionBadge,
  EmptyState,
  PageHeader,
  ScoreBar,
  Segmented,
  Stat,
  TruthBadge,
  VersionTag,
} from "../components/ui";
import { CHART, fmt } from "../lib/format";
import { sessionStats, useStore } from "../store";
import type { Decision } from "../types";

const SPEEDS = [
  { value: 0.75, label: "0.75/s" },
  { value: 1.5, label: "1.5/s" },
  { value: 3, label: "3/s" },
  { value: 5, label: "5/s" },
];

export function LiveMonitor() {
  const { events, live, inspect, clearEvents, registry, scoringModel } = useStore();
  const [filter, setFilter] = useState<"all" | Decision | "mistakes">("all");
  const liveEvents = useMemo(() => events.filter((e) => e.source === "live"), [events]);
  const stats = useMemo(() => sessionStats(liveEvents), [liveEvents]);

  const throughput = useMemo(() => {
    const now = Date.now();
    return Array.from({ length: 30 }, (_, i) => {
      const end = now - (29 - i) * 2000;
      return liveEvents.filter((e) => e.at > end - 2000 && e.at <= end).length;
    });
  }, [liveEvents]);

  const visible = liveEvents.filter((event) => {
    if (filter === "all") return true;
    if (filter === "mistakes") {
      const flagged = event.prediction.decision !== "approve";
      return event.actual !== null && flagged !== (event.actual === 1);
    }
    return event.prediction.decision === filter;
  }).slice(0, 80);

  const model = scoringModel ?? registry?.champion ?? "—";

  return (
    <div className="page">
      <PageHeader
        eyebrow="Operations"
        title="Live Monitor"
        description="A simulated production stream of held-out transactions, scored one by one with reason codes. Ground truth is shown so you can watch the model earn its keep."
        actions={
          <>
            <Button variant="ghost" icon={<Trash2 size={15} />} onClick={clearEvents} disabled={!events.length}>Clear</Button>
            {live.running ? (
              <Button icon={<Pause size={15} />} onClick={live.stop}>Pause</Button>
            ) : (
              <Button variant="primary" icon={<Play size={15} />} onClick={live.start}>Start stream</Button>
            )}
          </>
        }
      />

      <div className="control-bar">
        <div className="control">
          <span className="control-label">Stream status</span>
          <span className={live.running ? "live-pill on" : "live-pill"}>
            <i /> {live.running ? "Streaming" : "Paused"}
          </span>
        </div>
        <div className="control">
          <span className="control-label">Throughput</span>
          <Segmented label="Transactions per second" value={live.settings.speed}
                     options={SPEEDS} onChange={(speed) => live.setSettings({ speed })} />
        </div>
        <div className="control grow">
          <span className="control-label">Fraud pressure · {fmt.pct(live.settings.fraudShare, 0)} of stream</span>
          <input type="range" min={0} max={0.4} step={0.02} value={live.settings.fraudShare}
                 aria-label="Share of fraudulent transactions in the stream"
                 onChange={(e) => live.setSettings({ fraudShare: Number(e.target.value) })} />
        </div>
        <div className="control">
          <span className="control-label">Scoring with</span>
          <span><VersionTag version={model} /></span>
        </div>
      </div>
      {live.error && <p className="error-text">{live.error}</p>}

      <div className="stat-grid">
        <Stat label="Scored" value={fmt.int(stats.total)} hint={<MiniBars values={throughput} color={CHART.accent} height={22} />} />
        <Stat label="Blocked" tone="block" value={fmt.int(stats.block)} hint={stats.total ? fmt.pct(stats.block / stats.total) : "—"} />
        <Stat label="Reviewed" tone="review" value={fmt.int(stats.review)} hint={stats.total ? fmt.pct(stats.review / stats.total) : "—"} />
        <Stat label="Fraud caught" tone="approve" value={`${stats.caught}/${stats.actualFraud}`}
              hint={stats.detectionRate === null ? "No fraud seen yet" : `${fmt.pct(stats.detectionRate)} detection`} />
        <Stat label="Missed fraud" value={fmt.int(stats.missed)} hint="Approved but actually fraud" />
        <Stat label="False alarms" value={fmt.int(stats.falseAlarms)}
              hint={stats.review + stats.block ? `${fmt.pct(stats.falseAlarms / (stats.review + stats.block))} of alerts` : "—"} />
      </div>

      <Card
        title="Transaction stream"
        subtitle="Newest first · click a row to investigate"
        pad={false}
        actions={
          <Segmented label="Filter stream" value={filter} onChange={setFilter} options={[
            { value: "all", label: "All" },
            { value: "block", label: "Blocked" },
            { value: "review", label: "Review" },
            { value: "approve", label: "Approved" },
            { value: "mistakes", label: "Errors" },
          ]} />
        }
      >
        {visible.length ? (
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>Time</th><th>Transaction</th><th className="num">Amount</th><th>Local hour</th>
                  <th>Risk score</th><th>Decision</th><th>Truth</th><th>Primary driver</th><th className="num">Latency</th>
                </tr>
              </thead>
              <tbody>
                <AnimatePresence initial={false}>
                  {visible.map((event) => {
                    const wrong = event.actual !== null && (event.prediction.decision !== "approve") !== (event.actual === 1);
                    return (
                      <motion.tr
                        key={event.key}
                        initial={{ opacity: 0, backgroundColor: "rgba(109,140,255,0.16)" }}
                        animate={{ opacity: 1, backgroundColor: "rgba(0,0,0,0)" }}
                        exit={{ opacity: 0, transition: { duration: 0 } }}
                        transition={{ duration: 0.6 }}
                        className={`clickable ${wrong ? "row-wrong" : ""}`}
                        onClick={() => inspect(event)}
                      >
                        <td className="mono muted">{fmt.clock(event.at)}</td>
                        <td className="mono">{event.input.transaction_id}</td>
                        <td className="num mono">{fmt.money(Number(event.input.Amount))}</td>
                        <td className="mono muted">{fmt.hour(Number(event.input.Time))}</td>
                        <td><ScoreBar score={event.prediction.risk_score} /></td>
                        <td><DecisionBadge decision={event.prediction.decision} size="sm" /></td>
                        <td><TruthBadge actual={event.actual} /></td>
                        <td className="small">{event.prediction.reasons[0]?.label.replace("Behavioral component", "Component") ?? <span className="muted">—</span>}</td>
                        <td className="num mono muted">{fmt.ms(event.prediction.latency_ms)}</td>
                      </motion.tr>
                    );
                  })}
                </AnimatePresence>
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState icon={<Radio size={22} />} title={liveEvents.length ? "Nothing matches this filter" : "The stream is idle"}
                      action={!live.running ? <Button variant="primary" icon={<Play size={15} />} onClick={live.start}>Start stream</Button> : undefined}>
            {liveEvents.length ? "Try another filter." : "Start the stream to score held-out transactions through the API in real time."}
          </EmptyState>
        )}
      </Card>
    </div>
  );
}
