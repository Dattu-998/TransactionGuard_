import { Eye, FileCheck2, Shield, UserRoundCheck } from "lucide-react";
import { SectionHeading } from "./ui";
import type { ModelInfo } from "../types";

const principles = [
  {
    icon: UserRoundCheck,
    number: "01",
    title: "Human review comes first",
    text: "A prediction is a model output. It should be considered alongside other evidence and reviewed by a person.",
  },
  {
    icon: Eye,
    number: "02",
    title: "Inputs are explicit",
    text: "The active feature names and saved decision threshold are displayed from the backend model metadata.",
  },
  {
    icon: FileCheck2,
    number: "03",
    title: "Evaluation is contextual",
    text: "Reported evaluation uses synthetic data and should not be interpreted as a real-world performance claim.",
  },
];

export function About({ info }: { info: ModelInfo | null }) {
  return (
    <section className="page-content content-width about-page">
      <SectionHeading
        eyebrow="ABOUT TRANSACTIONGUARD"
        title="Built for responsible review."
        description="An open demonstration of how machine-learning scores can be presented with clear context, visible limits, and an emphasis on human judgment."
      />

      <div className="about-intro-card">
        <span className="about-intro-icon"><Shield size={22} /></span>
        <div>
          <p className="eyebrow">THE DEMONSTRATION</p>
          <h2>Surface a signal. Keep the decision human.</h2>
          <p>
            TransactionGuard presents predictions from a saved transaction
            classification model. Its interface makes the active model,
            decision threshold, and returned risk labels visible instead of
            hiding them behind a score.
          </p>
        </div>
        <div className="about-active-model">
          <span>MODEL REPORTED BY API</span>
          <strong>{info?.name ?? "Unavailable"}</strong>
          <small>{info ? `Version ${info.version}` : "Connect to load model details"}</small>
        </div>
      </div>

      <div className="principles-grid">
        {principles.map(({ icon: Icon, number, title, text }) => (
          <article className="principle-card" key={number}>
            <div className="principle-top">
              <span>{number}</span>
              <Icon size={19} strokeWidth={1.7} />
            </div>
            <h3>{title}</h3>
            <p>{text}</p>
          </article>
        ))}
      </div>

      <div className="about-notice">
        <span className="about-notice-icon">i</span>
        <div>
          <strong>Demonstration, not a fraud-decision service</strong>
          <p>
            This project does not claim production-ready fraud detection,
            guaranteed accuracy, or business impact. Synthetic evaluation
            cannot establish how a model will perform in a different
            population or environment.
          </p>
        </div>
      </div>
    </section>
  );
}
