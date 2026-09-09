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

type FoundryResource = "job" | "models" | "overview" | "status";
type JobAction = "start" | "cancel" | "retry";
type ConfirmedJobAction = Exclude<JobAction, "cancel">;

const initialLoadState: LoadState = { error: "", loading: true };

const activeJobStatuses = new Set<FoundryJob["status"]>(["RUNNING", "CANCEL_REQUESTED"]);

function errorMessage(caught: unknown, fallback: string): string {
  return caught instanceof Error ? caught.message : fallback;
}

function jobStatusTone(status: FoundryJob["status"] | undefined): string {
  if (status === "SUCCEEDED") return "passed";
  if (status === "FAILED") return "failed";
  return "informational";
}

function admissionLabel(admitted: boolean | undefined): string {
  return admitted === undefined ? "unknown" : admitted ? "admitted" : "refused";
}

export function App(): ReactElement {
  const [demos, setDemos] = useState<Record<string, DemoRecord>>({});
  const [foundryStatus, setFoundryStatus] = useState<FoundryStatus | null>(null);
  const [overview, setOverview] = useState<FoundryOverview | null>(null);
  const [job, setJob] = useState<FoundryJob | null>(null);
  const [jobAction, setJobAction] = useState<JobAction | null>(null);
  const [jobError, setJobError] = useState<string>("");
  const [models, setModels] = useState<FoundryModel[]>([]);
  const [foundryFailures, setFoundryFailures] = useState<FoundryResource[]>([]);
  const [ollama, setOllama] = useState<OllamaHealth | null>(null);
  const [demoState, setDemoState] = useState<LoadState>(initialLoadState);
  const [foundryState, setFoundryState] = useState<LoadState>(initialLoadState);
  const [ollamaError, setOllamaError] = useState<string>("");
  const [ollamaLoading, setOllamaLoading] = useState<boolean>(true);
  const [reloadToken, setReloadToken] = useState<number>(0);
  const [verification, setVerification] = useState<FoundryVerification | null>(null);
  const [verifying, setVerifying] = useState<boolean>(false);
  const [verificationError, setVerificationError] = useState<string>("");
  const [verificationConfirmation, setVerificationConfirmation] = useState<boolean>(false);
  const [pendingJobAction, setPendingJobAction] = useState<ConfirmedJobAction | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [prompt, setPrompt] = useState<string>("request verify evidence");
  const [completion, setCompletion] = useState<ChatCompletion | null>(null);
  const [generating, setGenerating] = useState<boolean>(false);
  const [generationError, setGenerationError] = useState<string>("");

  const reload = useCallback(() => {
    setVerificationConfirmation(false);
    setPendingJobAction(null);
    setReloadToken((value) => value + 1);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setDemoState({ error: "", loading: true });
    setFoundryState({ error: "", loading: true });
    setOllamaError("");
    setOllamaLoading(true);

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

    void Promise.allSettled([
      fetchFoundryStatus(controller.signal),
      fetchFoundryModels(controller.signal),
      fetchFoundryOverview(controller.signal),
      fetchFoundryJob(controller.signal)
    ])
      .then(([statusResult, modelsResult, overviewResult, jobResult]) => {
        if (controller.signal.aborted) return;

        const failures: Array<{ resource: FoundryResource; reason: unknown }> = [];
        if (statusResult.status === "fulfilled") {
          setFoundryStatus(statusResult.value);
        } else {
          setFoundryStatus(null);
          failures.push({ resource: "status", reason: statusResult.reason });
        }
        if (modelsResult.status === "fulfilled") {
          const registeredModels = modelsResult.value;
          setModels(registeredModels);
          setSelectedModel((current) =>
            registeredModels.some((model) => model.id === current)
              ? current
              : registeredModels[0]?.id || ""
          );
        } else {
          setModels([]);
          setSelectedModel("");
          failures.push({ resource: "models", reason: modelsResult.reason });
        }
        if (overviewResult.status === "fulfilled") {
          setOverview(overviewResult.value);
        } else {
          setOverview(null);
          failures.push({ resource: "overview", reason: overviewResult.reason });
        }
        if (jobResult.status === "fulfilled") {
          setJob(jobResult.value);
        } else {
          setJob(null);
          failures.push({ resource: "job", reason: jobResult.reason });
        }

        setFoundryFailures(failures.map(({ resource }) => resource));
        const failedResources = failures.map(({ resource }) => resource).join(", ");
        setFoundryState({
          error:
            failures.length === 0
              ? ""
              : `Some Foundry state is unavailable (${failedResources}): ${errorMessage(
                  failures[0]?.reason,
                  "request failed"
                )}`,
          loading: false
        });
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
      })
      .finally(() => {
        if (!controller.signal.aborted) setOllamaLoading(false);
      });

    return () => controller.abort();
  }, [reloadToken]);

  useEffect(() => {
    if (!job || !activeJobStatuses.has(job.status)) return;

    const controller = new AbortController();
    let timer: number | undefined;
    let consecutiveFailures = 0;
    const poll = async (): Promise<void> => {
      try {
        const next = await fetchFoundryJob(controller.signal);
        if (controller.signal.aborted) return;
        consecutiveFailures = 0;
        setJob(next);
        setJobError("");
        if (!activeJobStatuses.has(next.status)) {
          reload();
          return;
        }
      } catch (caught: unknown) {
        if (!controller.signal.aborted) {
          consecutiveFailures += 1;
          setJobError(errorMessage(caught, "Unable to refresh Foundry job state"));
        }
      }
      if (!controller.signal.aborted) {
        const nextDelay = Math.min(1_000 * 2 ** consecutiveFailures, 10_000);
        timer = window.setTimeout(() => void poll(), nextDelay);
      }
    };
    timer = window.setTimeout(() => void poll(), 1_000);
    return () => {
      controller.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [job?.status, reload]);

  const updateJob = async (action: JobAction): Promise<void> => {
    if (jobAction !== null) return;
    setPendingJobAction(null);
    setJobAction(action);
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
      setJobError(errorMessage(caught, "Foundry job control failed"));
    } finally {
      setJobAction(null);
    }
  };

  const verify = async (): Promise<void> => {
    setVerificationConfirmation(false);
    setVerifying(true);
    setVerification(null);
    setVerificationError("");
    try {
      const result = await runFoundryVerification();
      setVerification(result);
      reload();
    } catch (caught: unknown) {
      setVerificationError(errorMessage(caught, "Foundry verification failed"));
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
      setGenerationError(errorMessage(caught, "Generation failed"));
    } finally {
      setGenerating(false);
    }
  };

  const jobIsActive = activeJobStatuses.has(job?.status ?? "IDLE");
  const canVerify =
    !verifying &&
    jobAction === null &&
    !foundryState.loading &&
    foundryStatus !== null &&
    overview?.resources.small === true &&
    !jobIsActive;
  const canStart =
    job !== null &&
    jobAction === null &&
    !foundryState.loading &&
    !verifying &&
    foundryStatus !== null &&
    overview?.resources.small === true &&
    !jobIsActive;
  const canRetry =
    jobAction === null &&
    !foundryState.loading &&
    !verifying &&
    foundryStatus !== null &&
    overview?.resources.small === true &&
    (job?.status === "FAILED" || job?.status === "CANCELLED");
  const modelRegistryStatus = foundryState.loading
    ? "Loading models"
    : foundryFailures.includes("models")
      ? "Model registry unavailable"
      : `${models.length} verified ${models.length === 1 ? "model" : "models"}`;

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

      <section
        aria-busy={foundryState.loading}
        aria-labelledby="foundry-heading"
        className="panel"
      >
        <div className="section-heading">
          <div>
            <p className="eyebrow">Golden path</p>
            <h2 id="foundry-heading">Durable model lifecycle</h2>
          </div>
          <span
            aria-live="polite"
            data-status={
              foundryState.loading
                ? "informational"
                : foundryStatus?.integrity === "ok"
                  ? "passed"
                  : "failed"
            }
            role="status"
          >
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
              Retry Foundry state
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
          <button
            aria-busy={verifying}
            disabled={!canVerify}
            onClick={() => setVerificationConfirmation(true)}
            type="button"
          >
            {verifying ? "Training and evaluating…" : "Run verified Foundry pipeline"}
          </button>
          <p>
            Executes a real low-cost train → checkpoint → held-out evaluation → export → promotion
            request. This foreground request cannot be cancelled from the console and creates a
            verification model, never a fake Hermes checkpoint.
          </p>
        </div>
        {verificationConfirmation ? (
          <div
            aria-describedby="verify-confirmation-description"
            aria-labelledby="verify-confirmation-heading"
            className="confirmation-panel"
            role="alertdialog"
          >
            <h3 id="verify-confirmation-heading">Run the verification pipeline?</h3>
            <p id="verify-confirmation-description">
              This writes a dataset, experiment, checkpoint, evaluation, export, and model-registry
              record. The request cannot be cancelled from this page after it starts.
            </p>
            <div className="action-row">
              <button autoFocus disabled={!canVerify} onClick={() => void verify()} type="button">
                Confirm verification run
              </button>
              <button
                className="secondary-button"
                onClick={() => setVerificationConfirmation(false)}
                type="button"
              >
                Go back
              </button>
            </div>
          </div>
        ) : null}
        <article className="receipt" aria-label="Foundry workload control">
          <div className="card-header">
            <h3>Bounded workload control</h3>
            <span
              aria-live="polite"
              data-status={foundryState.loading ? "informational" : jobStatusTone(job?.status)}
              role="status"
            >
              {foundryState.loading ? "Loading" : (job?.status ?? "Unavailable")}
            </span>
          </div>
          <p>
            Small tier: {admissionLabel(overview?.resources.small)} · Medium tier:{" "}
            {admissionLabel(overview?.resources.medium)} · Available memory:{" "}
            {overview ? `${(overview.resources.snapshot.available_bytes / 1024 ** 3).toFixed(2)} GiB` : "unknown"}
          </p>
          <div className="action-row">
            <button
              aria-busy={jobAction === "start"}
              disabled={!canStart}
              onClick={() => setPendingJobAction("start")}
              type="button"
            >
              {jobAction === "start" ? "Starting…" : "Start bounded run"}
            </button>
            <button
              aria-busy={jobAction === "cancel"}
              disabled={jobAction !== null || verifying || job?.status !== "RUNNING"}
              onClick={() => void updateJob("cancel")}
              type="button"
            >
              {jobAction === "cancel" ? "Requesting cancel…" : "Cancel safely"}
            </button>
            <button
              aria-busy={jobAction === "retry"}
              disabled={!canRetry}
              onClick={() => setPendingJobAction("retry")}
              type="button"
            >
              {jobAction === "retry"
                ? "Retrying…"
                : job?.status === "CANCELLED"
                  ? "Retry cancelled run"
                  : "Retry failed run"}
            </button>
          </div>
          {pendingJobAction ? (
            <div
              aria-describedby="job-confirmation-description"
              aria-labelledby="job-confirmation-heading"
              className="confirmation-panel"
              role="alertdialog"
            >
              <h3 id="job-confirmation-heading">
                {pendingJobAction === "retry" ? "Retry" : "Start"} the bounded workload?
              </h3>
              <p id="job-confirmation-description">
                This launches a real local training and evaluation job and writes Foundry artifacts.
                It can be cancelled at an artifact-safe boundary.
              </p>
              <div className="action-row">
                <button
                  autoFocus
                  disabled={pendingJobAction === "retry" ? !canRetry : !canStart}
                  onClick={() => void updateJob(pendingJobAction)}
                  type="button"
                >
                  {pendingJobAction === "retry" ? "Confirm retry" : "Confirm bounded run"}
                </button>
                <button
                  className="secondary-button"
                  onClick={() => setPendingJobAction(null)}
                  type="button"
                >
                  Go back
                </button>
              </div>
            </div>
          ) : null}
          {job && job.status !== "IDLE" ? (
            <dl className="receipt-details job-details">
              <div>
                <dt>Job</dt>
                <dd>{job.job_id}</dd>
              </div>
              <div>
                <dt>Started</dt>
                <dd>{job.started_at}</dd>
              </div>
              <div>
                <dt>Finished</dt>
                <dd>{job.finished_at ?? "in progress"}</dd>
              </div>
            </dl>
          ) : !foundryState.loading && job ? (
            <p className="empty-state">No bounded workload has been started in this API process.</p>
          ) : null}
          {jobError ? <p className="inline-error" role="alert">{jobError}</p> : null}
          {job?.error ? <p className="inline-error" role="alert">{job.error}</p> : null}
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
          <article
            aria-labelledby="verification-receipt-heading"
            aria-live="polite"
            className="receipt"
          >
            <div className="card-header">
              <h3 id="verification-receipt-heading">Verification receipt</h3>
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
          <span aria-live="polite" role="status">
            {modelRegistryStatus}
          </span>
        </div>
        {models.length === 0 &&
        !foundryState.loading &&
        !foundryFailures.includes("models") ? (
          <p className="empty-state">
            No model has passed promotion gates. Run the verified pipeline above to create the
            bounded infrastructure-verification model.
          </p>
        ) : null}
        <div className="grid">
          {models.map((model) => (
            <article className="card" key={model.id}>
              <div className="card-header">
                <h3 className="model-id">{model.id}</h3>
                <span data-status="passed">{model.status}</span>
              </div>
              <p>{model.capabilities.join(" · ") || "No capabilities supplied by registry."}</p>
              <small>
                {model.limitations.length > 0
                  ? `Limitations: ${model.limitations.join(" · ")}`
                  : "No limitations supplied by registry."}
              </small>
            </article>
          ))}
        </div>
        <form
          aria-busy={generating}
          className="generation-form"
          onSubmit={(event) => void generate(event)}
        >
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
          <div className="form-meta">
            <small>Whitespace-only prompts are not submitted.</small>
            <output aria-live="polite" htmlFor="prompt">
              {prompt.length.toLocaleString()} / 50,000 characters
            </output>
          </div>
          <button
            aria-busy={generating}
            disabled={models.length === 0 || !selectedModel || !prompt.trim() || generating}
            type="submit"
          >
            {generating ? "Generating…" : "Generate through stable API"}
          </button>
        </form>
        {generationError ? (
          <p className="inline-error" role="alert">
            {generationError}
          </p>
        ) : null}
        {completion ? (
          <article
            aria-labelledby="model-output-heading"
            aria-live="polite"
            className="completion"
          >
            <h3 id="model-output-heading">Model output</h3>
            <pre>{completion.choices[0]?.message.content}</pre>
            <small>
              {completion.model} · {completion.usage.completion_tokens} {completion.usage.unit}
            </small>
            <dl className="receipt-details completion-provenance" aria-label="Generation provenance">
              <div>
                <dt>Checkpoint</dt>
                <dd>{completion.evidence.checkpoint_id}</dd>
              </div>
              <div>
                <dt>Checkpoint SHA-256</dt>
                <dd>{completion.evidence.checkpoint_sha256}</dd>
              </div>
              <div>
                <dt>Runtime</dt>
                <dd>{completion.evidence.runtime}</dd>
              </div>
            </dl>
          </article>
        ) : null}
      </section>

      <section aria-busy={ollamaLoading} aria-labelledby="providers-heading" className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Local serving</p>
            <h2 id="providers-heading">Provider health</h2>
          </div>
          <span
            aria-live="polite"
            data-status={ollamaLoading ? "informational" : ollama ? "passed" : "failed"}
            role="status"
          >
            {ollamaLoading
              ? "Checking Ollama"
              : ollama
                ? "Ollama discovery available"
                : "Ollama unavailable"}
          </span>
        </div>
        {ollama && !ollamaLoading ? (
          <div>
            <p>
              Installed local models: <strong>{ollama.models.join(", ") || "none"}</strong>
            </p>
            <small>
              Discovery does not imply that a model has loaded or passed generation verification.
            </small>
          </div>
        ) : ollamaError && !ollamaLoading ? (
          <p className="inline-error" role="alert">{ollamaError}</p>
        ) : (
          <p className="notice" role="status">Checking local provider…</p>
        )}
      </section>

      <section aria-labelledby="demos-heading" className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Research prototypes</p>
            <h2 id="demos-heading">Cognitive demo evidence</h2>
          </div>
          <span aria-live="polite" role="status">
            {demoState.loading
              ? "Running"
              : `${Object.keys(demos).length} ${Object.keys(demos).length === 1 ? "result" : "results"}`}
          </span>
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
              Retry demo results
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
