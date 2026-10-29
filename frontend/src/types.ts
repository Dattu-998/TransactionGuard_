export type Decision = "approve" | "review" | "block";
export type RiskLevel = "low" | "elevated" | "high" | "critical";
export type Stage = "champion" | "challenger" | "archived";
export type ApiStatus = "connecting" | "online" | "offline";

export type Route =
  | "command"
  | "live"
  | "investigate"
  | "batch"
  | "registry"
  | "monitoring"
  | "developers";

export type Transaction = Record<string, number | string | null> & {
  transaction_id?: string;
  Time: number;
  Amount: number;
  Class?: number;
};

export type Thresholds = {
  review: number;
  block: number;
  block_precision_target?: number;
};

export type Reason = {
  feature: string;
  label: string;
  value: number;
  baseline: number;
  contribution: number;
  direction: "increases_risk" | "decreases_risk";
};

export type Prediction = {
  transaction_id: string | null;
  model_version: string;
  fraud_probability: number;
  risk_score: number;
  risk_level: RiskLevel;
  decision: Decision;
  thresholds: Thresholds;
  reasons: Reason[];
  latency_ms: number;
};

export type ScoredEvent = {
  key: string;
  at: number;
  source: "live" | "manual" | "batch";
  input: Transaction;
  actual: 0 | 1 | null;
  prediction: Prediction;
  roundTripMs: number;
};

export type Health = {
  status: string;
  model_loaded: boolean;
  champion: string | null;
  versions: string[];
  uptime_seconds: number;
  api_version: string;
};

export type VersionMetrics = {
  val_pr_auc: number;
  pr_auc: number;
  roc_auc: number;
  precision: number;
  recall: number;
  f1: number;
  brier: number;
  fp: number;
  fn: number;
  cost_reduction: number;
  amount_recall: number;
  latency_p50_ms: number;
};

export type VersionSummary = {
  version: string;
  name: string;
  family: string;
  stage: Stage;
  summary: string;
  highlights: string[];
  created_at: string;
  thresholds: Thresholds;
  metrics: VersionMetrics;
  artifact_bytes: number;
};

export type Dataset = {
  source: string;
  description: string;
  fingerprint: string;
  rows: number;
  fraud_rows: number;
  fraud_rate: number;
  splits: Record<string, { rows: number; fraud_rows: number }>;
};

export type RegistryIndex = {
  champion: string;
  created_at: string;
  dataset: Dataset;
  costs: { false_negative: number; false_positive: number };
  versions: VersionSummary[];
};

export type EvalMetrics = {
  pr_auc: number;
  roc_auc: number;
  precision: number;
  recall: number;
  f1: number;
  tn: number;
  fp: number;
  fn: number;
  tp: number;
  fpr: number;
  fnr: number;
  brier: number;
};

export type SweepPoint = {
  threshold: number;
  cost: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number;
  recall: number;
  f1: number;
  flag_rate: number;
};

export type HistogramBucket = {
  bucket: string;
  start: number;
  legit: number;
  fraud: number;
  legit_share: number;
  fraud_share: number;
};

export type ModelCard = {
  version: string;
  name: string;
  family: string;
  summary: string;
  highlights: string[];
  stage: Stage;
  params: Record<string, string | number | boolean>;
  thresholds: Thresholds;
  training: { seconds: number; rows: number; fraud_rows: number };
  validation: EvalMetrics;
  test: EvalMetrics;
  test_block: EvalMetrics;
  decision_mix: Record<Decision, number>;
  business: {
    expected_cost: number;
    no_model_cost: number;
    cost_reduction: number;
    fraud_amount: number;
    fraud_amount_caught: number;
    amount_recall: number;
    alerts_per_1000: number;
  };
  curves: {
    pr: Array<{ recall: number; precision: number }>;
    roc: Array<{ fpr: number; tpr: number }>;
    calibration: Array<{ bin: string; predicted: number; observed: number; count: number }>;
    threshold_sweep: SweepPoint[];
    score_histogram: HistogramBucket[];
  };
  importance: Array<{ feature: string; label: string; importance: number; std: number }>;
  latency: { p50_ms: number; p95_ms: number; batch_throughput_per_sec: number };
  artifact_bytes: number;
  created_at: string;
  dataset: Dataset;
  environment: { python: string; scikit_learn: string; git_commit: string | null };
  costs: { false_negative: number; false_positive: number };
};

export type Comparison = {
  transaction_id: string | null;
  champion: string;
  agreement: boolean;
  results: Array<{
    version: string;
    name: string;
    stage: Stage;
    fraud_probability: number;
    risk_score: number;
    risk_level: RiskLevel;
    decision: Decision;
  }>;
};

export type BatchRow = Transaction & {
  fraud_probability: number;
  risk_score: number;
  risk_level: RiskLevel;
  decision: Decision;
  reasons: Reason[];
};

export type BatchResult = {
  model_version: string;
  summary: {
    processed: number;
    approve: number;
    review: number;
    block: number;
    flagged: number;
    flag_rate: number;
    highest_risk: number;
    mean_risk: number;
    amount_total: number;
    amount_flagged: number;
    latency_ms: number;
  };
  score_histogram: HistogramBucket[];
  evaluation: (Partial<EvalMetrics> & { label_column: string; fraud_rows: number }) | null;
  rows: BatchRow[];
};

export type FeatureStats = {
  label: string;
  mean: number;
  std: number;
  min: number;
  max: number;
  p01: number;
  p05: number;
  p25: number;
  median: number;
  p75: number;
  p95: number;
  p99: number;
};

export type SchemaInfo = {
  features: string[];
  statistics: Record<string, FeatureStats>;
  optional: string[];
};

export type DriftFeature = {
  feature: string;
  psi: number;
  status: "stable" | "moderate" | "significant";
  mean_shift: number;
  expected: number[];
  actual: number[];
};

export type DriftReport = {
  rows: number;
  features_checked: number;
  status: "stable" | "moderate" | "significant";
  counts: Record<"stable" | "moderate" | "significant", number>;
  max_psi: number;
  features: DriftFeature[];
  flag_rate: number;
  scenario?: string;
  description?: string;
};

export type ServiceMetrics = {
  uptime_seconds: number;
  requests_total: number;
  requests_by_path: Record<string, number>;
  errors_total: number;
  transactions_scored: number;
  decisions: Record<Decision, number>;
  by_version: Record<string, number>;
  latency_ms: { p50: number; p95: number; p99: number };
};
