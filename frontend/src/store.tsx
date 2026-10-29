import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { api } from "./api";
import type {
  ApiStatus,
  BatchResult,
  Health,
  ModelCard,
  Prediction,
  RegistryIndex,
  SchemaInfo,
  ScoredEvent,
  Transaction,
} from "./types";

const MAX_EVENTS = 600;

export type LiveSettings = { speed: number; fraudShare: number };

type Store = {
  status: ApiStatus;
  health: Health | null;
  registry: RegistryIndex | null;
  schema: SchemaInfo | null;
  cards: Record<string, ModelCard>;
  loadCard: (version: string) => Promise<ModelCard | null>;
  scoringModel: string | undefined;
  setScoringModel: (version: string | undefined) => void;
  events: ScoredEvent[];
  addEvent: (event: Omit<ScoredEvent, "key" | "at">) => ScoredEvent;
  clearEvents: () => void;
  live: {
    running: boolean;
    settings: LiveSettings;
    setSettings: (settings: Partial<LiveSettings>) => void;
    start: () => void;
    stop: () => void;
    error: string;
  };
  selected: ScoredEvent | null;
  inspect: (event: ScoredEvent | null) => void;
  lastBatch: { name: string; result: BatchResult; inputs: Transaction[] } | null;
  setLastBatch: (batch: Store["lastBatch"]) => void;
  notice: string;
  setNotice: (message: string) => void;
};

const StoreContext = createContext<Store | null>(null);

let sequence = 0;

export function StoreProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<ApiStatus>("connecting");
  const [health, setHealth] = useState<Health | null>(null);
  const [registry, setRegistry] = useState<RegistryIndex | null>(null);
  const [schema, setSchema] = useState<SchemaInfo | null>(null);
  const [cards, setCards] = useState<Record<string, ModelCard>>({});
  const [scoringModel, setScoringModel] = useState<string | undefined>(undefined);
  const [events, setEvents] = useState<ScoredEvent[]>([]);
  const [selected, setSelected] = useState<ScoredEvent | null>(null);
  const [lastBatch, setLastBatch] = useState<Store["lastBatch"]>(null);
  const [notice, setNotice] = useState("");
  const [running, setRunning] = useState(false);
  const [liveError, setLiveError] = useState("");
  const [settings, setSettingsState] = useState<LiveSettings>({ speed: 1.5, fraudShare: 0.08 });
  const pendingCards = useRef<Record<string, Promise<ModelCard | null>>>({});

  // ---- connection + metadata --------------------------------------------
  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const controller = new AbortController();
    const connect = async () => {
      try {
        const result = await api.health(controller.signal);
        if (cancelled) return;
        setHealth(result);
        setStatus(result.model_loaded ? "online" : "offline");
        if (result.model_loaded) {
          const [index, info] = await Promise.all([api.registry(), api.schema()]);
          if (cancelled) return;
          setRegistry(index);
          setSchema(info);
          return;
        }
      } catch {
        if (cancelled) return;
        setStatus("offline");
      }
      timer = window.setTimeout(connect, 8000);
    };
    connect();
    return () => {
      cancelled = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, []);

  const loadCard = useCallback(async (version: string) => {
    if (cards[version]) return cards[version];
    pendingCards.current[version] ??= api
      .card(version)
      .then((card) => {
        setCards((current) => ({ ...current, [version]: card }));
        return card;
      })
      .catch(() => {
        delete pendingCards.current[version];
        return null;
      });
    return pendingCards.current[version];
  }, [cards]);

  // ---- session events ---------------------------------------------------
  const addEvent = useCallback((partial: Omit<ScoredEvent, "key" | "at">) => {
    sequence += 1;
    const event: ScoredEvent = { ...partial, key: `${Date.now()}-${sequence}`, at: Date.now() };
    setEvents((current) => [event, ...current].slice(0, MAX_EVENTS));
    return event;
  }, []);

  const clearEvents = useCallback(() => setEvents([]), []);

  // ---- live feed ----------------------------------------------------------
  const queue = useRef<Transaction[]>([]);
  const busy = useRef(false);
  const settingsRef = useRef(settings);
  const modelRef = useRef(scoringModel);
  settingsRef.current = settings;
  modelRef.current = scoringModel;

  const setSettings = useCallback((next: Partial<LiveSettings>) => {
    setSettingsState((current) => {
      if (next.fraudShare !== undefined && next.fraudShare !== current.fraudShare) queue.current = [];
      return { ...current, ...next };
    });
  }, []);

  useEffect(() => {
    if (!running) return;
    let cancelled = false;
    const tick = async () => {
      if (busy.current || cancelled) return;
      busy.current = true;
      try {
        if (queue.current.length < 5) {
          const batch = await api.samples(60, settingsRef.current.fraudShare);
          queue.current.push(...batch);
        }
        const next = queue.current.shift();
        if (!next || cancelled) return;
        const started = performance.now();
        const prediction: Prediction = await api.predict(next, modelRef.current, true);
        if (cancelled) return;
        addEvent({
          source: "live",
          input: next,
          actual: next.Class === 1 ? 1 : next.Class === 0 ? 0 : null,
          prediction,
          roundTripMs: performance.now() - started,
        });
        setLiveError("");
      } catch (error) {
        setLiveError(error instanceof Error ? error.message : "Live scoring failed.");
      } finally {
        busy.current = false;
      }
    };
    tick();
    const interval = window.setInterval(tick, 1000 / settings.speed);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [running, settings.speed, addEvent]);

  const value = useMemo<Store>(
    () => ({
      status,
      health,
      registry,
      schema,
      cards,
      loadCard,
      scoringModel,
      setScoringModel,
      events,
      addEvent,
      clearEvents,
      live: {
        running,
        settings,
        setSettings,
        start: () => setRunning(true),
        stop: () => setRunning(false),
        error: liveError,
      },
      selected,
      inspect: setSelected,
      lastBatch,
      setLastBatch,
      notice,
      setNotice,
    }),
    [status, health, registry, schema, cards, loadCard, scoringModel, events, addEvent, clearEvents,
      running, settings, setSettings, liveError, selected, lastBatch, notice],
  );

  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
}

export function useStore(): Store {
  const store = useContext(StoreContext);
  if (!store) throw new Error("useStore must be used inside StoreProvider");
  return store;
}

export function useCard(version: string | undefined) {
  const { cards, loadCard } = useStore();
  useEffect(() => {
    if (version) loadCard(version);
  }, [version, loadCard]);
  return version ? cards[version] ?? null : null;
}

export function useChampion() {
  const { registry } = useStore();
  return registry?.versions.find((v) => v.version === registry.champion) ?? null;
}

export function sessionStats(events: ScoredEvent[]) {
  let approve = 0, review = 0, block = 0, amount = 0, amountStopped = 0;
  let actualFraud = 0, caught = 0, missed = 0, falseAlarms = 0, latency = 0;
  for (const event of events) {
    const decision = event.prediction.decision;
    const value = Number(event.input.Amount) || 0;
    amount += value;
    latency += event.prediction.latency_ms;
    if (decision === "approve") approve += 1;
    else if (decision === "review") review += 1;
    else block += 1;
    if (decision !== "approve") amountStopped += value;
    if (event.actual === 1) {
      actualFraud += 1;
      if (decision === "approve") missed += 1;
      else caught += 1;
    } else if (event.actual === 0 && decision !== "approve") {
      falseAlarms += 1;
    }
  }
  return {
    total: events.length, approve, review, block, amount, amountStopped,
    actualFraud, caught, missed, falseAlarms,
    detectionRate: actualFraud ? caught / actualFraud : null,
    avgLatency: events.length ? latency / events.length : 0,
  };
}
