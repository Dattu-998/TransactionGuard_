export type Page = "overview" | "analyse" | "intelligence" | "about";
export type Health = "checking" | "ready" | "unavailable";

export type ModelInfo = {
  name: string;
  version: string;
  threshold: number;
  features: string[];
  metrics: Record<string, number>;
  evaluation_data?: string;
};

export type Prediction = {
  "Fraud Probability": number;
  "Predicted Class": string;
  "Risk Category": string;
  "Prediction Status": string;
};

export type AnalysisResult = {
  rows: Array<Record<string, unknown> & Prediction>;
  summary: {
    processed: number;
    flagged: number;
    highest_probability: number;
  };
};
