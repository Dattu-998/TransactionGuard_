import type {
  BatchResult,
  Comparison,
  DriftReport,
  Health,
  ModelCard,
  Prediction,
  RegistryIndex,
  SchemaInfo,
  ServiceMetrics,
  Transaction,
} from "./types";

export const API_BASE = (
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api"
).replace(/\/+$/, "");

export const DOCS_URL = API_BASE.replace(/\/api$/, "") + "/docs";

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

function describe(detail: unknown): string | null {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const location = Array.isArray(item?.loc) ? item.loc.filter((p: unknown) => p !== "body").join(".") : "";
        return location ? `${location}: ${item?.msg}` : String(item?.msg ?? item);
      })
      .join("; ");
  }
  return null;
}

async function request<T>(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), init?.timeoutMs ?? 60_000);
  const outer = init?.signal;
  outer?.addEventListener("abort", () => controller.abort());
  try {
    const response = await fetch(`${API_BASE}${path}`, { ...init, signal: controller.signal });
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      throw new ApiError(describe(body?.detail) ?? `Request failed (${response.status}).`, response.status);
    }
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (controller.signal.aborted && !outer?.aborted) {
      throw new ApiError("The API took too long to respond.", 0);
    }
    throw error instanceof Error && error.name === "AbortError"
      ? error
      : new ApiError("The TransactionGuard API could not be reached.", 0);
  } finally {
    window.clearTimeout(timer);
  }
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

const modelQuery = (model?: string) => (model ? `model=${encodeURIComponent(model)}` : "");

export function stripLabel(transaction: Transaction): Transaction {
  const { Class: _label, ...rest } = transaction;
  return rest as Transaction;
}

export const api = {
  health: (signal?: AbortSignal) => request<Health>("/health", { signal, timeoutMs: 90_000 }),
  metrics: () => request<ServiceMetrics>("/metrics"),
  registry: () => request<RegistryIndex>("/models"),
  card: (version: string) => request<ModelCard>(`/models/${version}`),
  schema: () => request<SchemaInfo>("/schema"),
  samples: (count: number, fraudShare: number, seed?: number) =>
    request<{ transactions: Transaction[] }>(
      `/samples?count=${count}&fraud_share=${fraudShare}${seed !== undefined ? `&seed=${seed}` : ""}`,
    ).then((r) => r.transactions),
  predict: (transaction: Transaction, model?: string, explain = true) =>
    request<Prediction>(
      `/predict?explain=${explain}${model ? `&${modelQuery(model)}` : ""}`,
      json(stripLabel(transaction)),
    ),
  compare: (transaction: Transaction) =>
    request<Comparison>("/predict/compare", json(stripLabel(transaction))),
  batch: (transactions: Transaction[], model?: string) =>
    request<BatchResult>(`/predict/batch?${modelQuery(model)}`, {
      ...json({ transactions }),
      timeoutMs: 120_000,
    }),
  upload: (file: File, model?: string) => {
    const data = new FormData();
    data.append("file", file);
    return request<BatchResult>(`/predict/upload?${modelQuery(model)}`, {
      method: "POST",
      body: data,
      timeoutMs: 180_000,
    });
  },
  driftScenarios: () => request<Record<string, string>>("/monitor/drift/scenarios"),
  driftSimulate: (scenario: string) =>
    request<DriftReport>(`/monitor/drift/simulate?scenario=${scenario}`),
  drift: (transactions: Transaction[]) =>
    request<DriftReport>("/monitor/drift", { ...json({ transactions }), timeoutMs: 120_000 }),
};
