import { useEffect, useState } from "react";
import { Activity, Menu, ShieldCheck, X } from "lucide-react";
import { About } from "./components/About";
import { Analysis } from "./components/Analysis";
import { Insights } from "./components/Insights";
import { Overview } from "./components/Overview";
import { PageLink } from "./components/ui";
import type { AnalysisResult, Health, ModelInfo, Page, Prediction } from "./types";

const API = (
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api"
).replace(/\/+$/, "");

async function readResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail || `The API request failed (${response.status}).`);
  }
  return response.json() as Promise<T>;
}

const navigation: Array<{ id: Page; label: string }> = [
  { id: "overview", label: "Overview" },
  { id: "analyse", label: "Analyze transactions" },
  { id: "intelligence", label: "Model insights" },
  { id: "about", label: "About" },
];

export default function App() {
  const [page, setPage] = useState<Page>("overview");
  const [menuOpen, setMenuOpen] = useState(false);
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [health, setHealth] = useState<Health>("checking");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const controller = new AbortController();

    fetch(`${API}/health`, { signal: controller.signal })
      .then(readResponse<{ status: string; model_loaded: boolean }>)
      .then((data) => setHealth(data.model_loaded ? "ready" : "unavailable"))
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setHealth("unavailable");
          setError(
            reason instanceof Error
              ? reason.message
              : "The TransactionGuard API could not be reached.",
          );
        }
      });

    fetch(`${API}/model-info`, { signal: controller.signal })
      .then(readResponse<ModelInfo>)
      .then(setInfo)
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(
            reason instanceof Error
              ? reason.message
              : "Model information could not be loaded.",
          );
        }
      });

    return () => controller.abort();
  }, []);

  const navigate = (target: Page) => {
    setPage(target);
    setMenuOpen(false);
    setError("");
    window.scrollTo({
      top: 0,
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "auto"
        : "smooth",
    });
  };

  const analyzeFile = async (file: File) => {
    setLoading(true);
    setError("");
    const data = new FormData();
    data.append("file", file);

    try {
      const response = await fetch(`${API}/predict/upload`, {
        method: "POST",
        body: data,
      });
      setResult(await readResponse<AnalysisResult>(response));
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "The transaction file could not be analyzed.",
      );
    } finally {
      setLoading(false);
    }
  };

  const predict = async (payload: Record<string, number>) => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const prediction = await readResponse<Prediction>(response);
      setResult({
        rows: [prediction],
        summary: {
          processed: 1,
          flagged: prediction["Predicted Class"] === "Potential Fraud" ? 1 : 0,
          highest_probability: prediction["Fraud Probability"],
        },
      });
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "The transaction could not be analyzed.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="header-inner">
          <button
            className="brand"
            type="button"
            onClick={() => navigate("overview")}
            aria-label="TransactionGuard home"
          >
            <span className="brand-mark" aria-hidden="true">
              <ShieldCheck size={19} strokeWidth={1.8} />
            </span>
            <span>Transaction<span className="brand-light">Guard</span></span>
          </button>

          <nav className="desktop-nav" aria-label="Main navigation">
            {navigation.map((item) => (
              <button
                key={item.id}
                type="button"
                className={page === item.id ? "nav-link is-active" : "nav-link"}
                aria-current={page === item.id ? "page" : undefined}
                onClick={() => navigate(item.id)}
              >
                {item.label}
              </button>
            ))}
          </nav>

          <div className={`service-status status-${health}`} role="status">
            <span className="status-dot" aria-hidden="true" />
            <span>
              {health === "checking"
                ? "Connecting"
                : health === "ready"
                  ? "Model ready"
                  : "API unavailable"}
            </span>
          </div>

          <button
            className="menu-toggle"
            type="button"
            aria-label={menuOpen ? "Close navigation" : "Open navigation"}
            aria-expanded={menuOpen}
            aria-controls="mobile-navigation"
            onClick={() => setMenuOpen((current) => !current)}
          >
            {menuOpen ? <X size={21} /> : <Menu size={21} />}
          </button>
        </div>
        {menuOpen && (
          <nav
            className="mobile-nav"
            id="mobile-navigation"
            aria-label="Mobile navigation"
          >
            {navigation.map((item) => (
              <button
                key={item.id}
                type="button"
                className={page === item.id ? "mobile-nav-link is-active" : "mobile-nav-link"}
                aria-current={page === item.id ? "page" : undefined}
                onClick={() => navigate(item.id)}
              >
                {item.label}
              </button>
            ))}
          </nav>
        )}
      </header>

      {error && (
        <div className="global-alert" role="alert">
          <div className="content-width alert-inner">
            <Activity size={17} aria-hidden="true" />
            <span>{error}</span>
            <button
              type="button"
              className="alert-dismiss"
              aria-label="Dismiss message"
              onClick={() => setError("")}
            >
              <X size={17} />
            </button>
          </div>
        </div>
      )}

      <main id="main-content">
        {page === "overview" && (
          <Overview info={info} health={health} navigate={navigate} />
        )}
        {page === "analyse" && (
          <Analysis
            info={info}
            result={result}
            loading={loading}
            onAnalyzeFile={analyzeFile}
            onPredict={predict}
          />
        )}
        {page === "intelligence" && <Insights info={info} />}
        {page === "about" && <About info={info} />}
      </main>

      <footer className="site-footer">
        <div className="content-width footer-inner">
          <button
            className="brand footer-brand"
            type="button"
            onClick={() => navigate("overview")}
          >
            <span className="brand-mark" aria-hidden="true">
              <ShieldCheck size={17} strokeWidth={1.8} />
            </span>
            <span>Transaction<span className="brand-light">Guard</span></span>
          </button>
          <p>Machine-learning support for careful human review.</p>
          <span className="footer-note">A demonstration project</span>
        </div>
      </footer>
    </div>
  );
}
