import type { ReactNode } from "react";
import { CHART, fmt, riskColor } from "../lib/format";
import type { Reason } from "../types";

export const axisProps = {
  stroke: CHART.axis,
  tick: { fill: CHART.text, fontSize: 11 },
  tickLine: false,
  axisLine: { stroke: CHART.grid },
};

export const tooltipProps = {
  contentStyle: {
    background: CHART.tooltipBg,
    border: "1px solid rgba(148,163,184,0.18)",
    borderRadius: 10,
    fontSize: 12,
    boxShadow: "0 12px 32px rgba(0,0,0,0.45)",
  },
  labelStyle: { color: "#e6edf3", fontWeight: 600, marginBottom: 4 },
  itemStyle: { padding: 0 },
  cursor: { fill: "rgba(148,163,184,0.06)", stroke: "rgba(148,163,184,0.25)" },
};

export function Legend({ items }: { items: Array<{ label: ReactNode; color: string; dashed?: boolean }> }) {
  return (
    <div className="chart-legend">
      {items.map((item, index) => (
        <span key={index}>
          <i style={{ background: item.dashed ? "transparent" : item.color, borderColor: item.color }}
             className={item.dashed ? "dashed" : ""} />
          {item.label}
        </span>
      ))}
    </div>
  );
}

/** Semi-circular risk gauge with review (50) and block (80) zones. */
export function RiskGauge({ score, size = 220 }: { score: number; size?: number }) {
  const stroke = 16;
  const radius = (size - stroke) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const polar = (value: number) => {
    const angle = Math.PI * (1 - value / 100);
    return [cx + radius * Math.cos(angle), cy - radius * Math.sin(angle)];
  };
  const arc = (from: number, to: number) => {
    const [x1, y1] = polar(from);
    const [x2, y2] = polar(to);
    return `M ${x1} ${y1} A ${radius} ${radius} 0 0 1 ${x2} ${y2}`;
  };
  const clamped = Math.min(100, Math.max(0, score));
  const [nx, ny] = polar(clamped);
  return (
    <svg viewBox={`0 0 ${size} ${size / 2 + 24}`} className="gauge" role="img"
         aria-label={`Risk score ${clamped.toFixed(0)} out of 100`}>
      <path d={arc(0, 49.6)} stroke="rgba(52,211,153,0.22)" strokeWidth={stroke} fill="none" />
      <path d={arc(50.4, 79.6)} stroke="rgba(251,191,36,0.22)" strokeWidth={stroke} fill="none" />
      <path d={arc(80.4, 100)} stroke="rgba(248,113,113,0.24)" strokeWidth={stroke} fill="none" />
      {clamped > 0.5 && (
        <path d={arc(0, clamped)} stroke={riskColor(clamped)} strokeWidth={stroke} fill="none"
              strokeLinecap="round" className="gauge-value" />
      )}
      <circle cx={nx} cy={ny} r={stroke / 2 + 3} fill="#0b0f15" stroke={riskColor(clamped)} strokeWidth={3} />
      <text x={cx} y={cy - 8} textAnchor="middle" className="gauge-score" fill="#f3f6fa">
        {clamped.toFixed(0)}
      </text>
      <text x={cx} y={cy + 14} textAnchor="middle" className="gauge-caption" fill={CHART.text}>
        RISK SCORE
      </text>
      <text x={stroke / 2} y={cy + 20} textAnchor="middle" className="gauge-tick" fill={CHART.axis}>0</text>
      <text x={size - stroke / 2} y={cy + 20} textAnchor="middle" className="gauge-tick" fill={CHART.axis}>100</text>
    </svg>
  );
}

/** Diverging bar chart of reason-code contributions (log-odds). */
export function ReasonBars({ reasons, compact = false }: { reasons: Reason[]; compact?: boolean }) {
  if (!reasons.length) {
    return <p className="muted small">No single input moves this score meaningfully.</p>;
  }
  const max = Math.max(...reasons.map((r) => Math.abs(r.contribution)), 0.5);
  return (
    <ul className={compact ? "reasons compact" : "reasons"}>
      {reasons.map((reason) => {
        const width = (Math.abs(reason.contribution) / max) * 50;
        const up = reason.contribution > 0;
        return (
          <li key={reason.feature}>
            <div className="reason-label">
              <span className="reason-name">{reason.label.replace("Behavioral component", "Component")}</span>
              {!compact && (
                <span className="reason-value mono">
                  {reason.feature === "Amount" ? fmt.money(reason.value)
                    : reason.feature === "Time" ? fmt.hour(reason.value)
                    : reason.value.toFixed(2)}
                  <span className="muted"> vs {reason.feature === "Amount" ? fmt.money(reason.baseline)
                    : reason.feature === "Time" ? fmt.hour(reason.baseline) : reason.baseline.toFixed(2)}</span>
                </span>
              )}
            </div>
            <div className="reason-track">
              <span className="reason-axis" />
              <span
                className={up ? "reason-bar up" : "reason-bar down"}
                style={up ? { left: "50%", width: `${width}%` } : { right: "50%", width: `${width}%` }}
              />
            </div>
            <span className={up ? "reason-amount up" : "reason-amount down"}>
              {fmt.signed(reason.contribution, 2)}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

export function MiniBars({ values, color, height = 34 }: { values: number[]; color: string; height?: number }) {
  const max = Math.max(...values, 1);
  return (
    <div className="mini-bars" style={{ height }}>
      {values.map((v, i) => (
        <span key={i} style={{ height: `${Math.max(4, (v / max) * 100)}%`, background: color }} />
      ))}
    </div>
  );
}
