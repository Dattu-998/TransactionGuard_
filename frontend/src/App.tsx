import {
  Boxes,
  Code2,
  FileSpreadsheet,
  LayoutDashboard,
  Menu,
  Radio,
  ScanSearch,
  ShieldCheck,
  Waves,
  X,
} from "lucide-react";
import { useCallback, useEffect, useState, type ReactNode } from "react";
import { API_BASE } from "./api";
import { TransactionDrawer } from "./components/TransactionDrawer";
import { VersionTag } from "./components/ui";
import { fmt } from "./lib/format";
import { BatchAnalysis } from "./pages/BatchAnalysis";
import { CommandCenter } from "./pages/CommandCenter";
import { Developers } from "./pages/Developers";
import { Investigate } from "./pages/Investigate";
import { LiveMonitor } from "./pages/LiveMonitor";
import { ModelRegistry } from "./pages/ModelRegistry";
import { Monitoring } from "./pages/Monitoring";
import { useChampion, useStore } from "./store";
import type { Route } from "./types";

const NAV: Array<{ group: string; items: Array<{ id: Route; label: string; icon: ReactNode }> }> = [
  {
    group: "Operate",
    items: [
      { id: "command", label: "Command Center", icon: <LayoutDashboard size={17} /> },
      { id: "live", label: "Live Monitor", icon: <Radio size={17} /> },
      { id: "investigate", label: "Investigate", icon: <ScanSearch size={17} /> },
      { id: "batch", label: "Batch Analysis", icon: <FileSpreadsheet size={17} /> },
    ],
  },
  {
    group: "Models",
    items: [
      { id: "registry", label: "Model Registry", icon: <Boxes size={17} /> },
      { id: "monitoring", label: "Monitoring", icon: <Waves size={17} /> },
    ],
  },
  { group: "Build", items: [{ id: "developers", label: "API & Docs", icon: <Code2 size={17} /> }] },
];

const ROUTES = NAV.flatMap((group) => group.items.map((item) => item.id));

function readRoute(): Route {
  const hash = window.location.hash.replace(/^#\/?/, "") as Route;
  return ROUTES.includes(hash) ? hash : "command";
}

export default function App() {
  const [route, setRoute] = useState<Route>(readRoute);
  const [menuOpen, setMenuOpen] = useState(false);
  const { status, health, registry, scoringModel, setScoringModel, live, events } = useStore();
  const champion = useChampion();

  useEffect(() => {
    const onHash = () => setRoute(readRoute());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const go = useCallback((target: Route) => {
    window.location.hash = `/${target}`;
    setMenuOpen(false);
    window.scrollTo({ top: 0 });
  }, []);

  const title = NAV.flatMap((g) => g.items).find((item) => item.id === route)?.label ?? "";
  const alerts = events.filter((e) => e.prediction.decision !== "approve").length;

  return (
    <div className="shell">
      <aside className={menuOpen ? "sidebar is-open" : "sidebar"}>
        <div className="brand">
          <span className="brand-mark"><ShieldCheck size={18} strokeWidth={2} /></span>
          <div>
            <strong>TransactionGuard</strong>
            <span>Fraud Intelligence</span>
          </div>
          <button type="button" className="icon-btn menu-close" onClick={() => setMenuOpen(false)} aria-label="Close menu">
            <X size={18} />
          </button>
        </div>

        <nav aria-label="Main">
          {NAV.map((group) => (
            <div key={group.group} className="nav-group">
              <span className="nav-group-label">{group.group}</span>
              {group.items.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={route === item.id ? "nav-item is-active" : "nav-item"}
                  aria-current={route === item.id ? "page" : undefined}
                  onClick={() => go(item.id)}
                >
                  {item.icon}
                  <span>{item.label}</span>
                  {item.id === "live" && live.running && <i className="nav-live" aria-label="streaming" />}
                  {item.id === "command" && alerts > 0 && <span className="nav-count">{alerts > 99 ? "99+" : alerts}</span>}
                </button>
              ))}
            </div>
          ))}
        </nav>

        <div className="sidebar-foot">
          {champion ? (
            <button type="button" className="champion-pill" onClick={() => go("registry")}>
              <span className="muted small">Champion in production</span>
              <span className="champion-pill-row">
                <VersionTag version={champion.version} />
                <strong>{champion.name}</strong>
              </span>
              <span className="muted small">PR-AUC {champion.metrics.pr_auc.toFixed(3)} · {registry?.versions.length} versions</span>
            </button>
          ) : (
            <span className="skeleton" style={{ height: 70 }} />
          )}
          <p className="disclaimer">Demo on a synthetic benchmark. Decisions support analysts; they are not financial advice.</p>
        </div>
      </aside>
      {menuOpen && <div className="sidebar-scrim" onClick={() => setMenuOpen(false)} />}

      <div className="main">
        <header className="topbar">
          <button type="button" className="icon-btn menu-open" onClick={() => setMenuOpen(true)} aria-label="Open menu">
            <Menu size={19} />
          </button>
          <div className="crumbs">
            <span className="muted">TransactionGuard</span>
            <span className="muted">/</span>
            <strong>{title}</strong>
          </div>
          <div className="topbar-right">
            {registry && (
              <label className="model-select">
                <span>Scoring model</span>
                <select value={scoringModel ?? ""} onChange={(e) => setScoringModel(e.target.value || undefined)}>
                  <option value="">Champion ({registry.champion})</option>
                  {registry.versions.filter((v) => v.version !== registry.champion).map((v) => (
                    <option key={v.version} value={v.version}>{v.version} · {v.name}</option>
                  ))}
                </select>
              </label>
            )}
            <span className={`api-status ${status}`} title={API_BASE}>
              <i />
              {status === "online" ? `API online${health ? ` · v${health.api_version}` : ""}`
                : status === "connecting" ? "Connecting to API…" : "API offline"}
              {status === "online" && health && <span className="muted"> · up {fmt.duration(health.uptime_seconds)}</span>}
            </span>
          </div>
        </header>

        {status === "connecting" && (
          <div className="banner info">Waking the scoring API — free hosting can take up to a minute on the first request.</div>
        )}
        {status === "offline" && (
          <div className="banner danger">
            The API at <code>{API_BASE}</code> is unreachable. Start it with <code>uvicorn app.main:app --port 8000</code>; retrying automatically.
          </div>
        )}

        <main className="content">
          {route === "command" && <CommandCenter go={go} />}
          {route === "live" && <LiveMonitor />}
          {route === "investigate" && <Investigate />}
          {route === "batch" && <BatchAnalysis />}
          {route === "registry" && <ModelRegistry />}
          {route === "monitoring" && <Monitoring />}
          {route === "developers" && <Developers />}
        </main>
      </div>
      <TransactionDrawer />
    </div>
  );
}
