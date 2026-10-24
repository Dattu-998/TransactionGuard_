import { ArrowRight } from "lucide-react";
import type { ReactNode } from "react";

export function SectionHeading({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="section-heading">
      <p className="eyebrow"><span className="eyebrow-line" /> {eyebrow}</p>
      <h1>{title}</h1>
      <p className="section-description">{description}</p>
    </div>
  );
}

export function PageLink({
  children,
  onClick,
  secondary = false,
}: {
  children: ReactNode;
  onClick: () => void;
  secondary?: boolean;
}) {
  return (
    <button
      type="button"
      className={secondary ? "button button-secondary" : "button button-primary"}
      onClick={onClick}
    >
      {children}
      <ArrowRight size={16} aria-hidden="true" />
    </button>
  );
}
