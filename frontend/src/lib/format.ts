import type { Decision, RiskLevel, Stage } from "../types";

export const DECISION_META: Record<Decision, { label: string; color: string; verb: string }> = {
  approve: { label: "Approve", color: "#34d399", verb: "Approved" },
  review: { label: "Review", color: "#fbbf24", verb: "Sent to review" },
  block: { label: "Block", color: "#f87171", verb: "Blocked" },
};

export const RISK_META: Record<RiskLevel, { label: string; color: string }> = {
  low: { label: "Low", color: "#34d399" },
  elevated: { label: "Elevated", color: "#a3e635" },
  high: { label: "High", color: "#fbbf24" },
  critical: { label: "Critical", color: "#f87171" },
};

export const STAGE_META: Record<Stage, { label: string; color: string }> = {
  champion: { label: "Champion", color: "#22d3ee" },
  challenger: { label: "Challenger", color: "#a78bfa" },
  archived: { label: "Baseline", color: "#94a3b8" },
};

export const VERSION_COLORS: Record<string, string> = {
  v1: "#94a3b8",
  v2: "#a78bfa",
  v3: "#22d3ee",
};

export const versionColor = (version: string) => VERSION_COLORS[version] ?? "#60a5fa";

export const CHART = {
  grid: "rgba(148, 163, 184, 0.09)",
  axis: "#6b7686",
  text: "#9aa4b2",
  tooltipBg: "#121821",
  approve: DECISION_META.approve.color,
  review: DECISION_META.review.color,
  block: DECISION_META.block.color,
  accent: "#6d8cff",
  cyan: "#22d3ee",
  legit: "#5b7cfa",
  fraud: "#f87171",
};

export function riskColor(score: number): string {
  if (score >= 80) return RISK_META.critical.color;
  if (score >= 50) return RISK_META.high.color;
  if (score >= 30) return RISK_META.elevated.color;
  return RISK_META.low.color;
}

const integer = new Intl.NumberFormat("en-US");
const currency = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const currencyCents = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 2 });
const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });

export const fmt = {
  int: (value: number) => integer.format(Math.round(value)),
  compact: (value: number) => compact.format(value),
  money: (value: number) => (Math.abs(value) >= 1000 ? currency.format(value) : currencyCents.format(value)),
  moneyCompact: (value: number) => (Math.abs(value) >= 10_000 ? `$${compact.format(value)}` : currency.format(value)),
  pct: (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`,
  num: (value: number, digits = 3) => value.toFixed(digits),
  ms: (value: number) => (value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${value.toFixed(value < 10 ? 1 : 0)} ms`),
  prob: (value: number) => {
    if (value === 0) return "0";
    if (value < 0.0001) return value.toExponential(1);
    if (value < 0.01) return `${(value * 100).toFixed(3)}%`;
    return `${(value * 100).toFixed(1)}%`;
  },
  bytes: (value: number) =>
    value >= 1_048_576 ? `${(value / 1_048_576).toFixed(1)} MB` : `${(value / 1024).toFixed(1)} KB`,
  duration: (seconds: number) => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  },
  hour: (timeSeconds: number) => {
    const hours = (timeSeconds / 3600) % 24;
    const h = Math.floor(hours);
    const m = Math.floor((hours - h) * 60);
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
  },
  clock: (timestamp: number) =>
    new Date(timestamp).toLocaleTimeString("en-US", { hour12: false }),
  date: (iso: string) =>
    new Date(iso).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" }),
  signed: (value: number, digits = 2) => `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`,
};

export function narrative(reasons: Array<{ label: string; direction: string; value: number; baseline: number }>): string {
  const drivers = reasons.filter((r) => r.direction === "increases_risk").slice(0, 3);
  if (!drivers.length) return "No input pushes this transaction meaningfully towards fraud; it resembles typical traffic.";
  const parts = drivers.map((r) => {
    const relation = r.value > r.baseline ? "unusually high" : "unusually low";
    return `${r.label.replace("Behavioral component ", "component ")} is ${relation}`;
  });
  const last = parts.pop();
  return `Risk is driven mainly by ${parts.length ? `${parts.join(", ")} and ${last}` : last}.`;
}

export function downloadCsv(filename: string, rows: Array<Record<string, unknown>>) {
  if (!rows.length) return;
  const headers = Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
  const escape = (value: unknown) => {
    const text = value === null || value === undefined ? "" : typeof value === "object" ? JSON.stringify(value) : String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  const csv = [headers.join(","), ...rows.map((row) => headers.map((h) => escape(row[h])).join(","))].join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
