import { useRef, useState, type ChangeEvent, type DragEvent } from "react";
import {
  ArrowRight,
  CheckCircle2,
  Fingerprint,
  FileUp,
  LoaderCircle,
  Upload,
} from "lucide-react";
import { ResultsDashboard } from "./ResultsDashboard";
import type { AnalysisResult, ModelInfo } from "../types";
import { SectionHeading } from "./ui";

const fallbackFeatures = [
  "Time",
  ...Array.from({ length: 28 }, (_, index) => `V${index + 1}`),
  "Amount",
];

function UploadPanel({
  loading,
  onAnalyze,
}: {
  loading: boolean;
  onAnalyze: (file: File) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [fileError, setFileError] = useState("");

  const selectFile = (selected: File | undefined) => {
    if (!selected) return;
    const extension = selected.name.split(".").pop()?.toLowerCase();
    if (!extension || !["csv", "xlsx", "xls"].includes(extension)) {
      setFileError("Choose a CSV, XLSX, or XLS file.");
      setFile(null);
      return;
    }
    if (selected.size > 25 * 1024 * 1024) {
      setFileError("This file exceeds the 25 MB upload limit.");
      setFile(null);
      return;
    }
    setFileError("");
    setFile(selected);
  };

  const onInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    selectFile(event.target.files?.[0]);
    event.target.value = "";
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    selectFile(event.dataTransfer.files[0]);
  };

  return (
    <div className="upload-panel">
      <div
        className={`dropzone${isDragging ? " is-dragging" : ""}${file ? " has-file" : ""}`}
        onDragEnter={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
            setIsDragging(false);
          }
        }}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          className="visually-hidden"
          type="file"
          accept=".csv,.xlsx,.xls"
          aria-label="Choose a CSV, XLSX, or XLS file"
          onChange={onInputChange}
          disabled={loading}
        />
        <div className="upload-icon">{file ? <CheckCircle2 /> : <FileUp />}</div>
        {file ? (
          <>
            <p className="dropzone-title">{file.name}</p>
            <p className="dropzone-hint">
              {formatBytes(file.size)} · Ready to analyze
            </p>
          </>
        ) : (
          <>
            <p className="dropzone-title">Drop a transaction file here</p>
            <p className="dropzone-hint">or choose a file from your device</p>
          </>
        )}
        <button
          className="button button-secondary choose-file"
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={loading}
        >
          {file ? "Choose a different file" : "Browse files"}
        </button>
        <p className="dropzone-formats">CSV · XLSX · XLS <span /> Up to 25 MB</p>
      </div>
      {fileError && <p className="field-error" role="alert">{fileError}</p>}
      <div className="upload-footer">
        <p><ShieldCheckIcon /> Files are sent to the connected API for analysis.</p>
        <button
          className="button button-primary"
          type="button"
          onClick={() => file && onAnalyze(file)}
          disabled={!file || loading}
        >
          {loading ? (
            <>
              <LoaderCircle className="spin" size={16} aria-hidden="true" />
              Processing file
            </>
          ) : (
            <>
              Analyze file <ArrowRight size={16} aria-hidden="true" />
            </>
          )}
        </button>
      </div>
      {loading && (
        <p className="processing-note" role="status">
          Processing with the active model. The file remains in this session.
        </p>
      )}
    </div>
  );
}

function ShieldCheckIcon() {
  return <CheckCircle2 size={15} aria-hidden="true" />;
}

function formatBytes(bytes: number) {
  return bytes < 1024 * 1024
    ? `${Math.max(1, Math.round(bytes / 1024))} KB`
    : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function TransactionForm({
  features,
  loading,
  onPredict,
}: {
  features: string[];
  loading: boolean;
  onPredict: (payload: Record<string, number>) => void;
}) {
  const entries = features.length ? features : fallbackFeatures;
  const [values, setValues] = useState<Record<string, number>>({});

  return (
    <form
      className="transaction-form"
      onSubmit={(event) => {
        event.preventDefault();
        onPredict(Object.fromEntries(entries.map((key) => [key, values[key] ?? 0])));
      }}
    >
      <div className="form-intro">
        <span className="form-step">MODEL INPUT</span>
        <p>
          Enter numeric values for the active model’s feature schema. Inputs
          default to 0; confirm each value before submitting.
        </p>
      </div>
      <div className="fields">
        {entries.map((key) => (
          <label key={key} htmlFor={`feature-${key}`}>
            {key}
            <input
              id={`feature-${key}`}
              type="number"
              step="any"
              value={values[key] ?? 0}
              onChange={(event) =>
                setValues((current) => ({
                  ...current,
                  [key]: Number(event.target.value),
                }))
              }
              disabled={loading}
            />
          </label>
        ))}
      </div>
      <div className="form-submit">
        <span>{entries.length} numeric features</span>
        <button className="button button-primary" type="submit" disabled={loading}>
          {loading ? (
            <>
              <LoaderCircle className="spin" size={16} />
              Analyzing
            </>
          ) : (
            <>
              Analyze transaction <ArrowRight size={16} />
            </>
          )}
        </button>
      </div>
    </form>
  );
}

export function Analysis({
  info,
  result,
  loading,
  onAnalyzeFile,
  onPredict,
}: {
  info: ModelInfo | null;
  result: AnalysisResult | null;
  loading: boolean;
  onAnalyzeFile: (file: File) => void;
  onPredict: (payload: Record<string, number>) => void;
}) {
  const [mode, setMode] = useState<"bulk" | "single">("bulk");

  return (
    <section className="page-content content-width">
      <SectionHeading
        eyebrow="TRANSACTION WORKSPACE"
        title="Analysis"
        description="Submit a file or an individual transaction to review output from the connected model."
      />
      <div className="analysis-meta">
        <span className="meta-label">ACTIVE MODEL</span>
        <span className="meta-value">{info?.name ?? "Loading model details"}</span>
        <span className="meta-divider" />
        <span className="meta-label">INPUTS</span>
        <span className="meta-value">{info ? `${info.features.length} features` : "—"}</span>
      </div>

      <div className="analysis-layout">
        <div className="analysis-workspace">
          <div className="mode-tabs" role="tablist" aria-label="Analysis type">
            <button
              id="bulk-tab"
              type="button"
              role="tab"
              aria-selected={mode === "bulk"}
              aria-controls="bulk-panel"
              className={mode === "bulk" ? "mode-tab is-selected" : "mode-tab"}
              onClick={() => setMode("bulk")}
            >
              <Upload size={16} /> Batch file
            </button>
            <button
              id="single-tab"
              type="button"
              role="tab"
              aria-selected={mode === "single"}
              aria-controls="single-panel"
              className={mode === "single" ? "mode-tab is-selected" : "mode-tab"}
              onClick={() => setMode("single")}
            >
              <Fingerprint size={16} /> Single transaction
            </button>
          </div>

          {mode === "bulk" ? (
            <div id="bulk-panel" role="tabpanel" aria-labelledby="bulk-tab">
              <UploadPanel loading={loading} onAnalyze={onAnalyzeFile} />
            </div>
          ) : (
            <div id="single-panel" role="tabpanel" aria-labelledby="single-tab">
              <TransactionForm
                features={info?.features ?? []}
                loading={loading}
                onPredict={onPredict}
              />
            </div>
          )}
        </div>

        <aside className="analysis-aside">
          <p className="eyebrow"><span className="eyebrow-line" /> BEFORE YOU BEGIN</p>
          <h2>Keep the context.</h2>
          <p>
            The model expects the exact numeric features shown in its schema.
            Anonymized components should not be interpreted as customer
            attributes.
          </p>
          <div className="aside-detail">
            <span className="aside-index">01</span>
            <p>Accepted file formats</p>
            <strong>CSV, XLSX, XLS</strong>
          </div>
          <div className="aside-detail">
            <span className="aside-index">02</span>
            <p>Maximum upload size</p>
            <strong>25 MB</strong>
          </div>
          <div className="aside-detail">
            <span className="aside-index">03</span>
            <p>Interpretation</p>
            <strong>Human review required</strong>
          </div>
        </aside>
      </div>

      <ResultsDashboard result={result} />
    </section>
  );
}
