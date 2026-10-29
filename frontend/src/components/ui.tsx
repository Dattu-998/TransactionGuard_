import { Info, LoaderCircle } from "lucide-react";
import type { CSSProperties, ReactNode } from "react";
import { DECISION_META, RISK_META, STAGE_META, riskColor, versionColor } from "../lib/format";
import type { Decision, RiskLevel, Stage } from "../types";

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow: string;
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        {description && <p className="page-description">{description}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  );
}

export function Card({
  title,
  subtitle,
  actions,
  children,
  className = "",
  pad = true,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  pad?: boolean;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <div className="card-head">
          <div>
            {title && <h2 className="card-title">{title}</h2>}
            {subtitle && <p className="card-subtitle">{subtitle}</p>}
          </div>
          {actions && <div className="card-actions">{actions}</div>}
        </div>
      )}
      <div className={pad ? "card-body" : "card-body flush"}>{children}</div>
    </section>
  );
}

export function Stat({
  label,
  value,
  hint,
  tone,
  icon,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "approve" | "review" | "block" | "accent";
  icon?: ReactNode;
}) {
  return (
    <div className={`stat ${tone ? `tone-${tone}` : ""}`}>
      <div className="stat-top">
        <span className="stat-label">{label}</span>
        {icon && <span className="stat-icon">{icon}</span>}
      </div>
      <div className="stat-value">{value}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}

export function DecisionBadge({ decision, size = "md" }: { decision: Decision; size?: "sm" | "md" }) {
  const meta = DECISION_META[decision];
  return (
    <span className={`badge badge-${size}`} style={{ "--badge": meta.color } as CSSProperties}>
      <i className="badge-dot" />
      {meta.label}
    </span>
  );
}

export function RiskBadge({ level }: { level: RiskLevel }) {
  const meta = RISK_META[level];
  return (
    <span className="badge badge-sm badge-outline" style={{ "--badge": meta.color } as CSSProperties}>
      {meta.label}
    </span>
  );
}

export function StageBadge({ stage }: { stage: Stage }) {
  const meta = STAGE_META[stage];
  return (
    <span className="badge badge-sm badge-outline" style={{ "--badge": meta.color } as CSSProperties}>
      {meta.label}
    </span>
  );
}

export function VersionTag({ version }: { version: string }) {
  return (
    <span className="version-tag" style={{ "--badge": versionColor(version) } as CSSProperties}>
      {version}
    </span>
  );
}

export function TruthBadge({ actual }: { actual: 0 | 1 | null }) {
  if (actual === null) return <span className="muted">—</span>;
  return (
    <span className={`truth ${actual ? "truth-fraud" : "truth-legit"}`}>{actual ? "Fraud" : "Legit"}</span>
  );
}

export function ScoreBar({ score }: { score: number }) {
  return (
    <span className="score-bar">
      <span className="score-track">
        <span className="score-fill" style={{ width: `${Math.max(2, score)}%`, background: riskColor(score) }} />
        <span className="score-mark" style={{ left: "50%" }} />
        <span className="score-mark" style={{ left: "80%" }} />
      </span>
      <span className="score-num" style={{ color: riskColor(score) }}>{score.toFixed(0)}</span>
    </span>
  );
}

export function Segmented<T extends string | number>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: Array<{ value: T; label: ReactNode }>;
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map((option) => (
        <button
          key={String(option.value)}
          type="button"
          role="radio"
          aria-checked={option.value === value}
          className={option.value === value ? "is-active" : ""}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function Button({
  children,
  onClick,
  variant = "secondary",
  icon,
  disabled,
  loading,
  type = "button",
  title,
}: {
  children?: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "ghost" | "danger";
  icon?: ReactNode;
  disabled?: boolean;
  loading?: boolean;
  type?: "button" | "submit";
  title?: string;
}) {
  return (
    <button
      type={type}
      className={`btn btn-${variant}`}
      onClick={onClick}
      disabled={disabled || loading}
      title={title}
    >
      {loading ? <LoaderCircle size={15} className="spin" /> : icon}
      {children && <span>{children}</span>}
    </button>
  );
}

export function EmptyState({
  icon,
  title,
  children,
  action,
}: {
  icon: ReactNode;
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <div className="empty-icon">{icon}</div>
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

export function Skeleton({ height = 16, width = "100%" }: { height?: number; width?: number | string }) {
  return <span className="skeleton" style={{ height, width }} />;
}

export function Hint({ children }: { children: ReactNode }) {
  return (
    <p className="hint">
      <Info size={13} />
      <span>{children}</span>
    </p>
  );
}

export function Delta({ value, suffix = "" }: { value: number; suffix?: string }) {
  const positive = value >= 0;
  return (
    <span className={positive ? "delta up" : "delta down"}>
      {positive ? "▲" : "▼"} {Math.abs(value).toFixed(1)}
      {suffix}
    </span>
  );
}

export function KeyValue({ items }: { items: Array<[string, ReactNode]> }) {
  return (
    <dl className="kv">
      {items.map(([key, value]) => (
        <div key={key}>
          <dt>{key}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}
