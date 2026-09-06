import { FormEvent, useCallback, useEffect, useState } from "react";
import type { ReactElement } from "react";

import {
  cancelFoundryJob,
  ChatCompletion,
  DemoRecord,
  fetchDemos,
  fetchFoundryJob,
  fetchFoundryModels,
  fetchFoundryOverview,
  fetchFoundryStatus,
  fetchOllamaHealth,
  FoundryJob,
  FoundryModel,
  FoundryOverview,
  FoundryStatus,
  FoundryVerification,
  generateWithFoundry,
  OllamaHealth,
  runFoundryVerification,
  retryFoundryJob,
  startFoundryJob
} from "./api";
import "./styles.css";

type LoadState = {
  error: string;
  loading: boolean;
};

const initialLoadState: LoadState = { error: "", loading: true };

export function App(): ReactElement {
  const [demos, setDemos] = useState<Record<string, DemoRecord>>({});
  const [foundryStatus, setFoundryStatus] = useState<FoundryStatus | null>(null);
  const [overview, setOverview] = useState<FoundryOverview | null>(null);
  const [job, setJob] = useState<FoundryJob | null>(null);
  const [jobError, setJobError] = useState<string>("");
  const [models, setModels] = useState<FoundryModel[]>([]);
  const [ollama, setOllama] = useState<OllamaHealth | null>(null);
  const [demoState, setDemoState] = useState<LoadState>(initialLoadState);
  const [foundryState, setFoundryState] = useState<LoadState>(initialLoadState);
  const [ollamaError, setOllamaError] = useState<string>("");
  const [reloadToken, setReloadToken] = useState<number>(0);
  const [verification, setVerification] = useState<FoundryVerification | null>(null);
  const [verifying, setVerifying] = useState<boolean>(false);
  const [verificationError, setVerificationError] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [prompt, setPrompt] = useState<string>("request verify evidence");
  const [completion, setCompletion] = useState<ChatCompletion | null>(null);
  const [generating, setGenerating] = useState<boolean>(false);
  const [generationError, setGenerationError] = useState<string>("");

  const reload = useCallback(() => setReloadToken((value) => value + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    setDemoState({ error: "", loading: true });
    setFoundryState({ error: "", loading: true });
    setOllamaError("");

    void fetchDemos(controller.signal)
      .then((payload) => {
        if (!controller.signal.aborted) setDemos(payload);
      })
      .catch((caught: unknown) => {
        if (!controller.signal.aborted) {
          setDemos({});
          setDemoState({
            error: caught instanceof Error ? caught.message : "Unable to load demo results",
            loading: false
          });
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setDemoState((state) => ({ ...state, loading: false }));
        }
      });

    void Promise.all([
      fetchFoundryStatus(controller.signal),
      fetchFoundryModels(controller.signal),
      fetchFoundryOverview(controller.signal),
      fetchFoundryJob(controller.signal)
    ])
      .then(([status, registeredModels, currentOverview, currentJob]) => {
        if (!controller.signal.aborted) {
          setFoundryStatus(status);
          setModels(registeredModels);
          setOverview(currentOverview);
          setJob(currentJob);
          setSelectedModel((current) => current || registeredModels[0]?.id || "");
        }
      })
      .catch((caught: unknown) => {
        if (!controller.signal.aborted) {
          setFoundryStatus(null);
          setModels([]);
          setOverview(null);
          setJob(null);
          setFoundryState({
            error: caught instanceof Error ? caught.message : "Unable to load Foundry state",
            loading: false
          });
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setFoundryState((state) => ({ ...state, loading: false }));
        }
      });

    void fetchOllamaHealth(controller.signal)
      .then((health) => {
        if (!controller.signal.aborted) setOllama(health);
      })
      .catch((caught: unknown) => {
        if (!controller.signal.aborted) {
          setOllama(null);
          setOllamaError(
            caught instanceof Error ? caught.message : "Local Ollama provider unavailable"
          );
        }
      });

    return () => controller.abort();
  }, [reloadToken]);

  useEffect(() => {
    if (job?.status !== "RUNNING" && job?.status !== "CANCEL_REQUESTED") return;
    const timer = window.setTimeout(reload, 500);
    return () => window.clearTimeout(timer);
  }, [job?.status, reload]);

  const updateJob = async (action: "start" | "cancel" | "retry"): Promise<void> => {
    setJobError("");
    try {
      const next =
        action === "start"
          ? await startFoundryJob()
          : action === "cancel"
            ? await cancelFoundryJob()
            : await retryFoundryJob();
      setJob(next);
    } catch (caught: unknown) {
      setJobError(caught instanceof Error ? caught.message : "Foundry job control failed");
    }
  };

  const verify = async (): Promise<void> => {
    setVerifying(true);
    setVerification(null);
    setVerificationError("");
    try {
      const result = await runFoundryVerification();
      setVerification(result);
      reload();
    } catch (caught: unknown) {
      setVerificationError(
        caught instanceof Error ? caught.message : "Foundry verification failed"
      );
    } finally {
      setVerifying(false);
    }
  };

  const generate = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    if (!selectedModel || !prompt.trim()) return;
    setGenerating(true);
    setCompletion(null);
    setGenerationError("");
    try {
      setCompletion(await generateWithFoundry(selectedModel, prompt));
    } catch (caught: unknown) {
      setGenerationError(caught instanceof Error ? caught.message : "Generation failed");
    } finally {
      setGenerating(false);
    }
  };

  return (
    <main className="app-shell">
      <section className="hero">
        <p className="eyebrow">Olympus Model Foundry</p>
        <h1>Models earn their names here.</h1>
        <p className="summary">
          Register data, run bounded experiments, preserve immutable checkpoints, compare against
          baselines, and expose only verified artifacts. No architecture stub is presented as a
          trained model.
        </p>
        <button className="secondary-button" onClick={reload} type="button">
          Refresh live state
        </button>
      </section>

      <section aria-labelledby="foundry-heading" className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Golden path</p>
            <h2 id="foundry-heading">Durable model lifecycle</h2>
          </div>
          <span data-status={foundryStatus?.integrity === "ok" ? "passed" : "failed"}>
            {foundryState.loading
              ? "Loading"
              : foundryStatus?.integrity === "ok"
                ? "Registry healthy"
                : "Unavailable"}
          </span>
        </div>
        {foundryState.error ? (
          <div className="error-panel" role="alert">
            <p>{foundryState.error}</p>
            <button onClick={reload} type="button">
              Retry
            </button>
          </div>
        ) : null}
        {foundryStatus ? (
          <dl className="metrics" aria-label="Foundry registry counts">
            {[
              ["Datasets", foundryStatus.datasets],
              ["Experiments", foundryStatus.experiments],
              ["Checkpoints", foundryStatus.checkpoints],
              ["Evaluations", foundryStatus.evaluations],
              ["Models", foundryStatus.models],
              ["Evidence events", foundryStatus.evidence_events]
            ].map(([label, value]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd>{value}</dd>
              </div>
            ))}
          </dl>
        ) : null}
        <div className="action-row">
          <button disabled={verifying} onClick={() => void verify()} type="button">
            {verifying ? "Training and evaluating…" : "Run verified Foundry pipeline"}
          </button>
          <p>
            Executes a real low-cost train → checkpoint → held-out evaluation → export → promotion
            run. It creates a verification model, never a fake Hermes checkpoint.
          </p>
        </div>
        <article className="receipt" aria-label="Foundry workload control">
          <div className="card-header">
            <h3>Bounded workload control</h3>
            <span data-status={job?.status === "SUCCEEDED" ? "passed" : "failed"}>
              {job?.status ?? "Unavailable"}
            </span>
          </div>
          <p>
            Small tier: {overview?.resources.small ? "admitted" : "refused"} · Medium tier:{" "}
            {overview?.resources.medium ? "admitted" : "refused"} · Available memory:{" "}
            {overview ? `${(overview.resources.snapshot.available_bytes / 1024 ** 3).toFixed(2)} GiB` : "unknown"}
          </p>
          <div className="action-row">
            <button
              disabled={job?.status === "RUNNING" || job?.status === "CANCEL_REQUESTED"}
              onClick={() => void updateJob("start")}
              type="button"
            >
              Start bounded run
            </button>
            <button
              disabled={job?.status !== "RUNNING"}
              onClick={() => void updateJob("cancel")}
              type="button"
            >
              Cancel safely
            </button>
            <button
              disabled={job?.status !== "FAILED" && job?.status !== "CANCELLED"}
              onClick={() => void updateJob("retry")}
              type="button"
            >
              Retry failed run
            </button>
          </div>
          {jobError ? <p className="inline-error" role="alert">{jobError}</p> : null}
        </article>
        <article className="receipt" aria-label="Foundry provenance">
          <div className="card-header">
            <h3>Live provenance and promotion boundary</h3>
            <span>{overview?.promotion_boundary ?? "Unknown"}</span>
          </div>
          <dl className="receipt-details">
            <div>
              <dt>Dataset</dt>
              <dd>{overview?.latest_dataset?.dataset_id ?? "none"}</dd>
            </div>
            <div>
              <dt>Dataset SHA-256</dt>
              <dd>{overview?.latest_dataset?.sha256 ?? "none"}</dd>
            </div>
            <div>
              <dt>Checkpoint</dt>
              <dd>{overview?.latest_checkpoint?.checkpoint_id ?? "none"}</dd>
            </div>
            <div>
              <dt>Evaluation verdict</dt>
              <dd>{overview?.latest_evaluation?.decision ?? "not evaluated"}</dd>
            </div>
          </dl>
        </article>
        {verificationError ? (
          <p className="inline-error" role="alert">
            {verificationError}
          </p>
        ) : null}
        {verification ? (
          <article className="receipt" aria-live="polite">
            <div className="card-header">
              <h3>Verification receipt</h3>
              <span data-status={verification.evaluation.passed ? "passed" : "failed"}>
                {verification.experiment.status}
              </span>
            </div>
            <p>{verification.evaluation.decision}</p>
            <dl className="receipt-details">
              <div>
                <dt>Experiment</dt>
                <dd>{verification.experiment.experiment_id}</dd>
              </div>
              <div>
                <dt>Checkpoint</dt>
                <dd>{verification.checkpoint.checkpoint_id}</dd>
              </div>
              <div>
                <dt>Candidate perplexity</dt>
                <dd>{verification.evaluation.candidate_metrics.perplexity.toFixed(3)}</dd>
              </div>
              <div>
                <dt>Uniform baseline</dt>
                <dd>{verification.evaluation.baseline_metrics.perplexity.toFixed(3)}</dd>
              </div>
            </dl>
          </article>
        ) : null}
      </section>

      <section aria-labelledby="models-heading" className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Percy contract</p>
            <h2 id="models-heading">Verified model registry</h2>
          </div>
          <span>{models.length} available</span>
        </div>
        {models.length === 0 && !foundryState.loading ? (
          <p className="empty-state">
            No model has passed promotion gates. Run the verified pipeline above to create the
            bounded infrastructure-verification model.
          </p>
        ) : null}
        <div className="grid">
          {models.map((model) => (
            <article className="card" key={model.id}>
              <div className="card-header">
                <h3>{model.id}</h3>
                <span data-status="passed">{model.status}</span>
              </div>
              <p>{model.capabilities.join(" · ")}</p>
              <small>{model.limitations[0]}</small>
            </article>
          ))}
        </div>
        <form className="generation-form" onSubmit={(event) => void generate(event)}>
          <label htmlFor="model">Model</label>
          <select
            disabled={models.length === 0 || generating}
            id="model"
            onChange={(event) => setSelectedModel(event.target.value)}
            value={selectedModel}
          >
            <option value="">Select a verified model</option>
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.id}
              </option>
            ))}
          </select>
          <label htmlFor="prompt">Prompt</label>
          <textarea
            disabled={generating}
            id="prompt"
            maxLength={50_000}
            onChange={(event) => setPrompt(event.target.value)}
            rows={4}
            value={prompt}
          />
          <button disabled={!selectedModel || !prompt.trim() || generating} type="submit">
            {generating ? "Generating…" : "Generate through stable API"}
          </button>
        </form>
        {generationError ? (
          <p className="inline-error" role="alert">
            {generationError}
          </p>
        ) : null}
        {completion ? (
          <article className="completion" aria-live="polite">
            <h3>Model output</h3>
            <pre>{completion.choices[0]?.message.content}</pre>
            <small>
              {completion.model} · {completion.usage.completion_tokens} {completion.usage.unit}
            </small>
          </article>
        ) : null}
      </section>

      <section aria-labelledby="providers-heading" className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Local serving</p>
            <h2 id="providers-heading">Provider health</h2>
          </div>
          <span data-status={ollama ? "passed" : "failed"}>
            {ollama ? "Ollama discovery available" : "Ollama unavailable"}
          </span>
        </div>
        {ollama ? (
          <div>
            <p>
              Installed local models: <strong>{ollama.models.join(", ") || "none"}</strong>
            </p>
            <small>
              Discovery does not imply that a model has loaded or passed generation verification.
            </small>
          </div>
        ) : (
          <p className="inline-error">{ollamaError || "Checking local provider…"}</p>
        )}
      </section>

      <section aria-labelledby="demos-heading" className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Research prototypes</p>
            <h2 id="demos-heading">Cognitive demo evidence</h2>
          </div>
          <span>{demoState.loading ? "Running" : `${Object.keys(demos).length} results`}</span>
        </div>
        {demoState.loading ? (
          <p className="notice" role="status">
            Running the published demo suite…
          </p>
        ) : null}
        {demoState.error ? (
          <div className="error-panel" role="alert">
            <p>Live results are unavailable: {demoState.error}</p>
            <button onClick={reload} type="button">
              Retry
            </button>
          </div>
        ) : null}
        <div aria-busy={demoState.loading} aria-label="Demo results" className="grid">
          {Object.entries(demos).map(([name, record]) => (
            <article className="card" key={name}>
              <div className="card-header">
                <h3>{name.replace(/_/g, " ")}</h3>
                <span
                  data-status={
                    record.passed === true
                      ? "passed"
                      : record.passed === false
                        ? "failed"
                        : "informational"
                  }
                >
                  {record.passed === true
                    ? "Passing"
                    : record.passed === false
                      ? "Needs work"
                      : "Informational"}
                </span>
              </div>
              <p>{record.detail ?? record.response ?? "No detail available."}</p>
              {record.category ? <small>Category: {record.category}</small> : null}
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}

export default App;
