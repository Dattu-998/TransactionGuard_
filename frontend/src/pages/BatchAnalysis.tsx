import { Database, Download, FileSpreadsheet, FileUp, Search, Table2, UploadCloud } from "lucide-react";
import { useMemo, useRef, useState, type DragEvent } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { axisProps, tooltipProps } from "../components/charts";
import {
  Button,
  Card,
  DecisionBadge,
  EmptyState,
  Hint,
  PageHeader,
  ScoreBar,
  Segmented,
  Stat,
  TruthBadge,
  VersionTag,
} from "../components/ui";
import { CHART, downloadCsv, fmt, riskColor } from "../lib/format";
import { useStore } from "../store";
import type { BatchRow, Decision, Prediction, Transaction } from "../types";

const PAGE_SIZE = 25;
const FEATURES = ["Time", ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`), "Amount"];

function labelOf(row: BatchRow): 0 | 1 | null {
  const value = row.Class ?? (row.is_fraud as number | undefined) ?? (row.label as number | undefined);
  return value === 1 ? 1 : value === 0 ? 0 : null;
}

export function BatchAnalysis() {
  const { lastBatch, setLastBatch, scoringModel, inspect, status, registry } = useStore();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [filter, setFilter] = useState<"all" | Decision>("all");
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(0);
  const input = useRef<HTMLInputElement>(null);

  const run = async (name: string, work: () => Promise<{ result: Awaited<ReturnType<typeof api.batch>>; inputs: Transaction[] }>) => {
    setBusy(true);
    setError("");
    try {
      const { result, inputs } = await work();
      setLastBatch({ name, result, inputs });
      setPage(0);
      setFilter("all");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The batch could not be scored.");
    } finally {
      setBusy(false);
    }
  };

  const runDemo = () =>
    run("Held-out demo batch (500 rows)", async () => {
      const inputs = await api.samples(500, 0.06, Math.floor(Math.random() * 1e9));
      return { result: await api.batch(inputs, scoringModel), inputs };
    });

  const runFile = (file: File) => {
    const ext = file.name.split(".").pop()?.toLowerCase();
    if (!ext || !["csv", "xlsx", "xls"].includes(ext)) {
      setError("Choose a CSV, XLSX or XLS file.");
      return;
    }
    if (file.size > 25 * 1024 * 1024) {
      setError("Files are limited to 25 MB.");
      return;
    }
    run(file.name, async () => {
      const result = await api.upload(file, scoringModel);
      return { result, inputs: result.rows as Transaction[] };
    });
  };

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    if (event.dataTransfer.files[0]) runFile(event.dataTransfer.files[0]);
  };

  const result = lastBatch?.result;
  const rows = useMemo(() => {
    if (!result) return [];
    const q = query.trim().toLowerCase();
    return result.rows
      .map((row, index) => ({ row, index }))
      .filter(({ row }) => filter === "all" || row.decision === filter)
      .filter(({ row, index }) => !q || String(row.transaction_id ?? index + 1).toLowerCase().includes(q))
      .sort((a, b) => b.row.risk_score - a.row.risk_score);
  }, [result, filter, query]);
  const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const shown = rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const evaluation = result?.evaluation;

  const open = (row: BatchRow, index: number) => {
    const { reasons: _r, fraud_probability: _p, risk_score: _s, risk_level: _l, decision: _d, ...features } = row;
    const prediction: Prediction = {
      transaction_id: (row.transaction_id as string) ?? null,
      model_version: result!.model_version,
      fraud_probability: row.fraud_probability,
      risk_score: row.risk_score,
      risk_level: row.risk_level,
      decision: row.decision,
      thresholds: registry?.versions.find((v) => v.version === result!.model_version)?.thresholds
        ?? { review: NaN, block: NaN },
      reasons: row.reasons ?? [],
      latency_ms: result!.summary.latency_ms / result!.summary.processed,
    };
    inspect({
      key: `batch-${index}`, at: Date.now(), source: "batch", roundTripMs: 0,
      input: { ...features, transaction_id: (row.transaction_id as string) ?? `row ${index + 1}` } as Transaction,
      actual: labelOf(row), prediction,
    });
  };

  const template = () => {
    const row: Record<string, unknown> = { transaction_id: "TX-0001" };
    FEATURES.forEach((f) => (row[f] = f === "Amount" ? 42.5 : 0));
    downloadCsv("transactionguard_template.csv", [row]);
  };

  return (
    <div className="page">
      <PageHeader
        eyebrow="Operations"
        title="Batch Analysis"
        description="Score a file of transactions in one call. Include a Class column (0/1) and the batch is evaluated against ground truth automatically."
        actions={
          <>
            <Button variant="ghost" icon={<FileSpreadsheet size={15} />} onClick={template}>Template</Button>
            {result && (
              <Button icon={<Download size={15} />} onClick={() =>
                downloadCsv(`scored_${lastBatch!.name.replace(/\.[^.]+$/, "")}.csv`,
                  result.rows.map(({ reasons, ...row }) => ({
                    ...row, top_reasons: (reasons ?? []).slice(0, 3).map((r) => r.feature).join("|"),
                  })))
              }>Export results</Button>
            )}
          </>
        }
      />

      <div className="grid grid-2">
        <div
          className={dragging ? "dropzone is-dragging" : "dropzone"}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          role="button"
          tabIndex={0}
          onClick={() => input.current?.click()}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
        >
          <UploadCloud size={28} />
          <strong>Drop a CSV or Excel file</strong>
          <span className="muted small">Columns: Time, V1–V28, Amount · optional transaction_id, Class · up to 20,000 rows / 25 MB</span>
          <Button variant="primary" icon={<FileUp size={15} />} loading={busy} disabled={status !== "online"}>Choose file</Button>
          <input ref={input} type="file" accept=".csv,.xlsx,.xls" hidden
                 onChange={(e) => { if (e.target.files?.[0]) runFile(e.target.files[0]); e.target.value = ""; }} />
        </div>
        <div className="dropzone alt">
          <Database size={28} />
          <strong>No file handy?</strong>
          <span className="muted small">Score 500 held-out transactions the models never saw during training, with labels, to see the full evaluation.</span>
          <Button icon={<Table2 size={15} />} onClick={runDemo} loading={busy} disabled={status !== "online"}>Run demo batch</Button>
        </div>
      </div>
      {error && <p className="error-text">{error}</p>}

      {!result ? (
        <Card><EmptyState icon={<FileSpreadsheet size={22} />} title="No batch scored yet">Results, charts and an evaluation appear here.</EmptyState></Card>
      ) : (
        <>
          <div className="batch-title">
            <span className="muted">Results for</span> <strong>{lastBatch!.name}</strong> <VersionTag version={result.model_version} />
          </div>
          <div className="stat-grid">
            <Stat label="Processed" value={fmt.int(result.summary.processed)} hint={`${fmt.ms(result.summary.latency_ms)} server time`} />
            <Stat label="Approved" tone="approve" value={fmt.int(result.summary.approve)} hint={fmt.pct(result.summary.approve / result.summary.processed)} />
            <Stat label="Review" tone="review" value={fmt.int(result.summary.review)} hint={fmt.pct(result.summary.review / result.summary.processed)} />
            <Stat label="Blocked" tone="block" value={fmt.int(result.summary.block)} hint={fmt.pct(result.summary.block / result.summary.processed)} />
            <Stat label="Value flagged" tone="accent" value={fmt.moneyCompact(result.summary.amount_flagged)} hint={`of ${fmt.moneyCompact(result.summary.amount_total)}`} />
            <Stat label="Highest risk" value={result.summary.highest_risk.toFixed(1)} hint={`mean ${result.summary.mean_risk.toFixed(1)}`} />
          </div>

          <div className="grid grid-2-1">
            <Card title="Risk score distribution" subtitle="Transactions per score band">
              <div className="chart-box" style={{ height: 230 }}>
                <ResponsiveContainer>
                  <BarChart data={result.score_histogram} margin={{ top: 8, right: 4, left: -16, bottom: 0 }}>
                    <CartesianGrid stroke={CHART.grid} vertical={false} />
                    <XAxis dataKey="start" {...axisProps} />
                    <YAxis {...axisProps} allowDecimals={false} scale="sqrt" />
                    <Tooltip {...tooltipProps} labelFormatter={(s) => `Score ${s}–${Number(s) + 5}`} />
                    <Bar dataKey="legit" name="Transactions" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                      {result.score_histogram.map((b) => <Cell key={b.bucket} fill={riskColor(b.start + 2.5)} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <Hint>Square-root scale so the small high-risk tail stays visible.</Hint>
            </Card>

            <Card title="Ground-truth evaluation" subtitle={evaluation ? `Using the “${evaluation.label_column}” column · review threshold` : "Add a Class column to evaluate"}>
              {evaluation ? (
                <div className="stack">
                  <div className="confusion">
                    <span />
                    <span className="cm-head">Flagged</span>
                    <span className="cm-head">Approved</span>
                    <span className="cm-head side">Fraud</span>
                    <span className="cm-cell good">{evaluation.tp}<small>caught</small></span>
                    <span className="cm-cell bad">{evaluation.fn}<small>missed</small></span>
                    <span className="cm-head side">Legit</span>
                    <span className="cm-cell warn">{evaluation.fp}<small>false alarms</small></span>
                    <span className="cm-cell neutral">{evaluation.tn}<small>cleared</small></span>
                  </div>
                  {evaluation.recall !== undefined && (
                    <div className="mini-metrics">
                      <div><span>Recall</span><strong>{fmt.pct(evaluation.recall)}</strong></div>
                      <div><span>Precision</span><strong>{fmt.pct(evaluation.precision ?? 0)}</strong></div>
                      <div><span>PR-AUC</span><strong>{(evaluation.pr_auc ?? 0).toFixed(3)}</strong></div>
                    </div>
                  )}
                </div>
              ) : (
                <EmptyState icon={<Table2 size={20} />} title="No labels in this batch">Upload with a Class column (1 = fraud) to see recall and precision.</EmptyState>
              )}
            </Card>
          </div>

          <Card
            title="Scored transactions"
            subtitle={`${fmt.int(rows.length)} rows · sorted by risk`}
            pad={false}
            actions={
              <>
                <label className="search">
                  <Search size={14} />
                  <input placeholder="Search ID" value={query} onChange={(e) => { setQuery(e.target.value); setPage(0); }} />
                </label>
                <Segmented label="Filter decisions" value={filter} onChange={(v) => { setFilter(v); setPage(0); }} options={[
                  { value: "all", label: "All" },
                  { value: "block", label: `Block ${result.summary.block}` },
                  { value: "review", label: `Review ${result.summary.review}` },
                  { value: "approve", label: "Approve" },
                ]} />
              </>
            }
          >
            <div className="table-scroll">
              <table className="table">
                <thead>
                  <tr><th>#</th><th>Transaction</th><th className="num">Amount</th><th>Risk score</th><th>Probability</th><th>Decision</th><th>Truth</th><th>Top reasons</th></tr>
                </thead>
                <tbody>
                  {shown.map(({ row, index }) => (
                    <tr key={index} className="clickable" onClick={() => open(row, index)}>
                      <td className="muted mono">{index + 1}</td>
                      <td className="mono">{String(row.transaction_id ?? "—")}</td>
                      <td className="num mono">{fmt.money(Number(row.Amount))}</td>
                      <td><ScoreBar score={row.risk_score} /></td>
                      <td className="mono muted">{fmt.prob(row.fraud_probability)}</td>
                      <td><DecisionBadge decision={row.decision} size="sm" /></td>
                      <td><TruthBadge actual={labelOf(row)} /></td>
                      <td className="small">
                        {(row.reasons ?? []).filter((r) => r.contribution > 0).slice(0, 2).map((r) => r.feature).join(", ") || <span className="muted">—</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="pager">
              <span className="muted small">Page {page + 1} of {pages}</span>
              <Button variant="ghost" onClick={() => setPage((p) => Math.max(0, p - 1))} disabled={page === 0}>Previous</Button>
              <Button variant="ghost" onClick={() => setPage((p) => Math.min(pages - 1, p + 1))} disabled={page >= pages - 1}>Next</Button>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
