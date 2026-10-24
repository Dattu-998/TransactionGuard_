import { Canvas, useFrame } from "@react-three/fiber";
import { Float, Sparkles } from "@react-three/drei";
import { motion, useReducedMotion } from "framer-motion";
import {
  ArrowDownToLine,
  ArrowRight,
  BarChart3,
  Check,
  FileSpreadsheet,
  Fingerprint,
  ScanSearch,
  ShieldCheck,
} from "lucide-react";
import { useRef } from "react";
import * as THREE from "three";
import type { Health, ModelInfo, Page } from "../types";
import { PageLink } from "./ui";

function ModelCore() {
  const ref = useRef<THREE.Group>(null);
  const reduceMotion = useReducedMotion();
  useFrame((_, delta) => {
    if (!reduceMotion && ref.current) ref.current.rotation.y += delta * 0.12;
  });
  return (
    <group ref={ref}>
      <Float
        speed={reduceMotion ? 0 : 0.7}
        rotationIntensity={reduceMotion ? 0 : 0.15}
        floatIntensity={reduceMotion ? 0 : 0.2}
      >
        <mesh>
          <icosahedronGeometry args={[0.9, 1]} />
          <meshPhysicalMaterial
            color="#9a7bff"
            transmission={0.22}
            roughness={0.26}
            metalness={0.3}
            transparent
            opacity={0.76}
          />
        </mesh>
        <mesh scale={1.1}>
          <icosahedronGeometry args={[0.9, 1]} />
          <meshBasicMaterial
            color="#8ee8ea"
            wireframe
            transparent
            opacity={0.28}
          />
        </mesh>
        <mesh rotation={[0.7, 0.4, 0.2]} scale={[1.35, 0.2, 1.35]}>
          <torusGeometry args={[1, 0.012, 8, 72]} />
          <meshBasicMaterial color="#91dfe1" transparent opacity={0.72} />
        </mesh>
      </Float>
    </group>
  );
}

function ProductPreview({
  info,
  health,
}: {
  info: ModelInfo | null;
  health: Health;
}) {
  const reduceMotion = useReducedMotion();
  return (
    <div className="preview-frame" aria-label="Live model information preview">
      <div className="preview-topbar">
        <div className="preview-window-dots" aria-hidden="true">
          <i />
          <i />
          <i />
        </div>
        <span>TRANSACTIONGUARD / MODEL OVERVIEW</span>
        <span className={`preview-live live-${health}`}>
          <i /> {health === "ready" ? "CONNECTED" : health.toUpperCase()}
        </span>
      </div>
      <div className="preview-content">
        <div className="preview-copy">
          <span className="preview-label">ACTIVE MODEL</span>
          <strong>{info?.name ?? "Model information unavailable"}</strong>
          <small>{info ? `Version ${info.version}` : "Waiting for API metadata"}</small>
        </div>
        <div className="preview-art" aria-hidden="true">
          <Canvas camera={{ position: [0, 0, 3.5], fov: 42 }} dpr={[1, 1.5]}>
            <ambientLight intensity={0.9} />
            <pointLight position={[2, 2, 3]} intensity={7} color="#a18bff" />
            <pointLight position={[-2, -1, 2]} intensity={5} color="#73dfe2" />
            <ModelCore />
            <Sparkles
              count={18}
              scale={3.6}
              size={1.4}
              speed={reduceMotion ? 0 : 0.2}
              color="#b8c2e5"
            />
          </Canvas>
        </div>
        <div className="preview-rule" />
        <div className="preview-data">
          <div>
            <span>DECISION THRESHOLD</span>
            <strong>
              {info ? `${(info.threshold * 100).toPrecision(3)}%` : "—"}
            </strong>
          </div>
          <div>
            <span>MODEL INPUTS</span>
            <strong>{info ? info.features.length : "—"}</strong>
          </div>
          <div>
            <span>RESULTS</span>
            <strong className="preview-results-label">Run an analysis</strong>
          </div>
        </div>
        <div className="preview-footnote">
          <span className="preview-footnote-mark"><Check size={12} /></span>
          Live model metadata · No sample transactions shown
        </div>
      </div>
    </div>
  );
}

const featureCards = [
  {
    icon: ScanSearch,
    eyebrow: "01 / SCREEN",
    title: "Review one transaction",
    text: "Submit numeric model inputs and inspect the returned probability and risk classification.",
  },
  {
    icon: FileSpreadsheet,
    eyebrow: "02 / ANALYZE",
    title: "Process a batch",
    text: "Upload a CSV or spreadsheet and review predictions across the submitted transactions.",
  },
  {
    icon: BarChart3,
    eyebrow: "03 / UNDERSTAND",
    title: "Explore model insights",
    text: "See the active model, saved decision threshold, and evaluation metrics when available.",
  },
  {
    icon: ArrowDownToLine,
    eyebrow: "04 / EXPORT",
    title: "Take results with you",
    text: "Download the transaction results as a CSV for further inspection.",
  },
];

export function Overview({
  info,
  health,
  navigate,
}: {
  info: ModelInfo | null;
  health: Health;
  navigate: (page: Page) => void;
}) {
  const reduceMotion = useReducedMotion();
  return (
    <motion.div
      className="overview-page"
      initial={reduceMotion ? false : { opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: reduceMotion ? 0 : 0.35 }}
    >
      <section className="hero content-width">
        <div className="hero-copy">
          <p className="eyebrow"><span className="eyebrow-line" /> TRANSACTION INTELLIGENCE</p>
          <h1>See the signal.<br /><span>Review with clarity.</span></h1>
          <p className="hero-description">
            A transparent machine-learning demonstration for screening
            transaction data and bringing unusual patterns into focus.
          </p>
          <div className="hero-actions">
            <PageLink onClick={() => navigate("analyse")}>Analyze transactions</PageLink>
            <button
              className="text-link"
              type="button"
              onClick={() => navigate("intelligence")}
            >
              Explore model insights <ArrowRight size={15} />
            </button>
          </div>
          <div className="hero-note">
            <ShieldCheck size={15} />
            Predictions are signals for human review, not financial decisions.
          </div>
        </div>
        <div className="hero-visual">
          <div className="visual-index"><span>01</span> / LIVE SYSTEM PREVIEW</div>
          <ProductPreview info={info} health={health} />
          <span className="visual-caption">Model metadata supplied by the connected API</span>
        </div>
        <span className="hero-orbit orbit-one" aria-hidden="true" />
        <span className="hero-orbit orbit-two" aria-hidden="true" />
      </section>

      <section className="feature-section content-width">
        <div className="section-intro">
          <div>
            <p className="eyebrow"><span className="eyebrow-line" /> A CLEARER WORKFLOW</p>
            <h2>From raw data<br />to considered review.</h2>
          </div>
          <p>
            The tools you need to explore the model’s output, with the context
            and limitations kept in view.
          </p>
        </div>
        <div className="feature-grid">
          {featureCards.map(({ icon: Icon, eyebrow, title, text }, index) => (
            <article className="feature-card" key={title}>
              <div className="feature-card-top">
                <span>{eyebrow}</span>
                <Icon size={19} strokeWidth={1.7} />
              </div>
              <h3>{title}</h3>
              <p>{text}</p>
              {index === 0 && <Fingerprint className="feature-watermark" aria-hidden="true" />}
            </article>
          ))}
        </div>
      </section>

      <section className="trust-section">
        <div className="content-width trust-inner">
          <div className="trust-icon"><ShieldCheck size={21} /></div>
          <div>
            <p className="eyebrow">IMPORTANT CONTEXT</p>
            <h2>A model output is a starting point.</h2>
            <p>
              TransactionGuard is a demonstration. Its evaluation uses
              synthetic data and does not establish real-world fraud-detection
              performance. Review every result with appropriate context.
            </p>
          </div>
          <button
            className="trust-link"
            type="button"
            onClick={() => navigate("about")}
          >
            Read about the model <ArrowRight size={15} />
          </button>
        </div>
      </section>
      <div className="overview-bottom-spacer" />
    </motion.div>
  );
}
