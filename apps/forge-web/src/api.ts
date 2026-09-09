export type DemoRecord = {
  passed?: boolean;
  detail?: string;
  response?: string;
  confidence?: number;
  category?: string;
};

export type FoundryStatus = {
  integrity: string;
  datasets: number;
  experiments: number;
  checkpoints: number;
  evaluations: number;
  models: number;
  evidence_events: number;
};

export type FoundryModel = {
  id: string;
  object: "model";
  owned_by: string;
  status: string;
  capabilities: string[];
  limitations: string[];
};

export type FoundryVerification = {
  experiment: { experiment_id: string; status: string };
  checkpoint: { checkpoint_id: string; sha256: string };
  evaluation: {
    evaluation_id: string;
    passed: boolean;
    baseline_metrics: Record<string, number>;
    candidate_metrics: Record<string, number>;
    decision: string;
  };
  model: { model_id: string; status: string };
  export_path: string;
};

export type ChatCompletion = {
  id: string;
  object: "chat.completion";
  model: string;
  choices: Array<{
    index: number;
    message: { role: "assistant"; content: string };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    unit: string;
  };
  evidence: {
    checkpoint_id: string;
    checkpoint_sha256: string;
    runtime: string;
  };
};

export type OllamaHealth = {
  status: string;
  provider: string;
  models: string[];
};

export type FoundryOverview = {
  promotion_boundary: string;
  latest_dataset: null | {
    dataset_id: string;
    version: string;
    sha256: string;
    source: string;
    license: string;
  };
  latest_experiment: null | {
    experiment_id: string;
    status: string;
    config: Record<string, unknown>;
  };
  latest_checkpoint: null | { checkpoint_id: string; sha256: string };
  latest_evaluation: null | { evaluation_id: string; passed: boolean; decision: string };
  resources: {
    small: boolean;
    medium: boolean;
    snapshot: { available_bytes: number; swap_fraction: number };
  };
};

export type FoundryJob = {
  job_id: string | null;
  status: "IDLE" | "RUNNING" | "CANCEL_REQUESTED" | "CANCELLED" | "SUCCEEDED" | "FAILED";
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
};

function isDemoRecord(value: unknown): value is DemoRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  const record = value as Record<string, unknown>;
  return (
    (record.passed === undefined || typeof record.passed === "boolean") &&
    (record.detail === undefined || typeof record.detail === "string") &&
    (record.response === undefined || typeof record.response === "string") &&
    (record.confidence === undefined || typeof record.confidence === "number") &&
    (record.category === undefined || typeof record.category === "string")
  );
}

function parseDemoPayload(value: unknown): Record<string, DemoRecord> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("The API returned an invalid demo payload");
  }
  const entries = Object.entries(value);
  if (entries.length === 0 || entries.some(([, record]) => !isDemoRecord(record))) {
    throw new Error("The API returned an invalid demo payload");
  }
  return Object.fromEntries(entries);
}

export async function fetchDemos(signal?: AbortSignal): Promise<Record<string, DemoRecord>> {
  const response = await apiFetch("/api/demos", { signal });
  return parseDemoPayload(await response.json());
}

async function apiFetch(
  path: string,
  options: { method?: string; body?: string; signal?: AbortSignal } = {}
): Promise<Response> {
  const response = await fetch(path, {
    cache: "no-store",
    credentials: "same-origin",
    method: options.method,
    body: options.body,
    headers: {
      Accept: "application/json",
      ...(options.body === undefined ? {} : { "Content-Type": "application/json" })
    },
    redirect: "error",
    referrerPolicy: "no-referrer",
    signal: options.signal
  });
  if (!response.ok) {
    let detail = "";
    try {
      const payload = (await response.json()) as { detail?: unknown };
      const normalized =
        typeof payload.detail === "string"
          ? payload.detail.replace(/[\u0000-\u001f\u007f]+/g, " ").trim().slice(0, 300)
          : "";
      detail = normalized ? `: ${normalized}` : "";
    } catch {
      detail = "";
    }
    throw new Error(`API request failed with status ${response.status}${detail}`);
  }
  return response;
}

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function isSha256(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}

function isTimestamp(value: unknown): value is string {
  if (typeof value !== "string") return false;
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(value)) {
    return false;
  }
  return Number.isFinite(Date.parse(value));
}

function isNullableRecord(value: unknown, validate: (record: Record<string, unknown>) => boolean) {
  return value === null || (isObject(value) && validate(value));
}

export async function fetchFoundryStatus(signal?: AbortSignal): Promise<FoundryStatus> {
  const payload = (await (
    await apiFetch("/api/foundry/status", { signal })
  ).json()) as Partial<FoundryStatus>;
  if (
    payload.integrity !== "ok" ||
    !isNonNegativeInteger(payload.datasets) ||
    !isNonNegativeInteger(payload.experiments) ||
    !isNonNegativeInteger(payload.checkpoints) ||
    !isNonNegativeInteger(payload.evaluations) ||
    !isNonNegativeInteger(payload.models) ||
    !isNonNegativeInteger(payload.evidence_events)
  ) {
    throw new Error("The API returned an invalid Foundry status");
  }
  return payload as FoundryStatus;
}

export async function fetchFoundryOverview(signal?: AbortSignal): Promise<FoundryOverview> {
  const payload: unknown = await (
    await apiFetch("/api/foundry/overview", { signal })
  ).json();
  if (
    !isObject(payload) ||
    !isNonEmptyString(payload.promotion_boundary) ||
    !isNullableRecord(
      payload.latest_dataset,
      (record) =>
        isNonEmptyString(record.dataset_id) &&
        isNonEmptyString(record.version) &&
        isSha256(record.sha256) &&
        isNonEmptyString(record.source) &&
        isNonEmptyString(record.license)
    ) ||
    !isNullableRecord(
      payload.latest_experiment,
      (record) =>
        isNonEmptyString(record.experiment_id) &&
        isNonEmptyString(record.status) &&
        isObject(record.config)
    ) ||
    !isNullableRecord(
      payload.latest_checkpoint,
      (record) => isNonEmptyString(record.checkpoint_id) && isSha256(record.sha256)
    ) ||
    !isNullableRecord(
      payload.latest_evaluation,
      (record) =>
        isNonEmptyString(record.evaluation_id) &&
        typeof record.passed === "boolean" &&
        isNonEmptyString(record.decision)
    ) ||
    !isObject(payload.resources) ||
    typeof payload.resources.small !== "boolean" ||
    typeof payload.resources.medium !== "boolean" ||
    !isObject(payload.resources.snapshot) ||
    !isNonNegativeInteger(payload.resources.snapshot.available_bytes) ||
    !isNumber(payload.resources.snapshot.swap_fraction) ||
    payload.resources.snapshot.swap_fraction < 0 ||
    payload.resources.snapshot.swap_fraction > 1
  ) {
    throw new Error("The API returned an invalid Foundry overview");
  }
  return payload as FoundryOverview;
}

function parseFoundryJob(payload: unknown): FoundryJob {
  const statuses: FoundryJob["status"][] = [
    "IDLE",
    "RUNNING",
    "CANCEL_REQUESTED",
    "CANCELLED",
    "SUCCEEDED",
    "FAILED"
  ];
  if (
    !isObject(payload) ||
    (payload.job_id !== null && !isNonEmptyString(payload.job_id)) ||
    typeof payload.status !== "string" ||
    !statuses.includes(payload.status as FoundryJob["status"]) ||
    (payload.started_at !== null && !isTimestamp(payload.started_at)) ||
    (payload.finished_at !== null && !isTimestamp(payload.finished_at)) ||
    (payload.error !== null && !isNonEmptyString(payload.error))
  ) {
    throw new Error("The API returned invalid Foundry job state");
  }

  const status = payload.status as FoundryJob["status"];
  const isIdle = status === "IDLE";
  const isActive = status === "RUNNING" || status === "CANCEL_REQUESTED";
  const isFailure = status === "FAILED" || status === "CANCELLED";
  const timestampsAreOrdered =
    payload.started_at === null ||
    payload.finished_at === null ||
    Date.parse(payload.finished_at as string) >= Date.parse(payload.started_at as string);
  if (
    (isIdle &&
      (payload.job_id !== null ||
        payload.started_at !== null ||
        payload.finished_at !== null ||
        payload.error !== null)) ||
    (!isIdle && (payload.job_id === null || payload.started_at === null)) ||
    (isActive && (payload.finished_at !== null || payload.error !== null)) ||
    (!isIdle && !isActive && payload.finished_at === null) ||
    (isFailure && payload.error === null) ||
    (status === "SUCCEEDED" && payload.error !== null) ||
    !timestampsAreOrdered
  ) {
    throw new Error("The API returned inconsistent Foundry job state");
  }
  return payload as FoundryJob;
}

export async function fetchFoundryJob(signal?: AbortSignal): Promise<FoundryJob> {
  return parseFoundryJob(
    await (await apiFetch("/api/foundry/jobs/current", { signal })).json()
  );
}

async function updateFoundryJob(action: "start" | "cancel" | "retry"): Promise<FoundryJob> {
  return parseFoundryJob(
    await (
      await apiFetch(`/api/foundry/jobs/${action}`, { method: "POST" })
    ).json()
  );
}

export const startFoundryJob = (): Promise<FoundryJob> => updateFoundryJob("start");
export const cancelFoundryJob = (): Promise<FoundryJob> => updateFoundryJob("cancel");
export const retryFoundryJob = (): Promise<FoundryJob> => updateFoundryJob("retry");

export async function fetchFoundryModels(signal?: AbortSignal): Promise<FoundryModel[]> {
  const payload = (await (await apiFetch("/api/v1/models", { signal })).json()) as {
    data?: unknown;
  };
  if (!Array.isArray(payload.data)) {
    throw new Error("The API returned an invalid model registry");
  }
  if (
    payload.data.some(
      (model) =>
        !isObject(model) ||
        !isNonEmptyString(model.id) ||
        model.object !== "model" ||
        !isNonEmptyString(model.owned_by) ||
        model.status !== "VERIFIED" ||
        !isStringArray(model.capabilities) ||
        !isStringArray(model.limitations)
    )
  ) {
    throw new Error("The API returned an invalid model registry");
  }
  return payload.data as FoundryModel[];
}

export async function runFoundryVerification(): Promise<FoundryVerification> {
  const payload: unknown = await (
    await apiFetch("/api/foundry/verify", { method: "POST" })
  ).json();
  if (
    !isObject(payload) ||
    !isObject(payload.experiment) ||
    !isNonEmptyString(payload.experiment.experiment_id) ||
    payload.experiment.status !== "VERIFIED" ||
    !isObject(payload.checkpoint) ||
    !isNonEmptyString(payload.checkpoint.checkpoint_id) ||
    !isSha256(payload.checkpoint.sha256) ||
    !isObject(payload.evaluation) ||
    !isNonEmptyString(payload.evaluation.evaluation_id) ||
    payload.evaluation.passed !== true ||
    !isObject(payload.evaluation.baseline_metrics) ||
    !isObject(payload.evaluation.candidate_metrics) ||
    !isNumber(payload.evaluation.baseline_metrics.perplexity) ||
    payload.evaluation.baseline_metrics.perplexity <= 0 ||
    !isNumber(payload.evaluation.candidate_metrics.perplexity) ||
    payload.evaluation.candidate_metrics.perplexity <= 0 ||
    !isNonEmptyString(payload.evaluation.decision) ||
    !isObject(payload.model) ||
    !isNonEmptyString(payload.model.model_id) ||
    payload.model.status !== "VERIFIED" ||
    !isNonEmptyString(payload.export_path)
  ) {
    throw new Error("The API returned an invalid Foundry verification receipt");
  }
  return payload as FoundryVerification;
}

export async function generateWithFoundry(
  model: string,
  prompt: string
): Promise<ChatCompletion> {
  if (!model.trim() || model.length > 200 || !prompt.trim() || prompt.length > 50_000) {
    throw new Error("Model and prompt must satisfy the Foundry API input limits");
  }
  const payload: unknown = await (
    await apiFetch("/api/v1/chat/completions", {
      method: "POST",
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: prompt }],
        max_tokens: 240,
        temperature: 0.7,
        seed: 7,
        stream: false
      })
    })
  ).json();
  if (
    !isObject(payload) ||
    typeof payload.id !== "string" ||
    payload.object !== "chat.completion" ||
    payload.model !== model ||
    !Array.isArray(payload.choices) ||
    payload.choices.length < 1 ||
    !isObject(payload.choices[0]) ||
    !isNonNegativeInteger(payload.choices[0].index) ||
    !isObject(payload.choices[0].message) ||
    payload.choices[0].message.role !== "assistant" ||
    typeof payload.choices[0].message.content !== "string" ||
    (payload.choices[0].finish_reason !== "stop" &&
      payload.choices[0].finish_reason !== "length") ||
    !isObject(payload.usage) ||
    !isNonNegativeInteger(payload.usage.prompt_tokens) ||
    !isNonNegativeInteger(payload.usage.completion_tokens) ||
    !isNonNegativeInteger(payload.usage.total_tokens) ||
    payload.usage.total_tokens !==
      payload.usage.prompt_tokens + payload.usage.completion_tokens ||
    payload.usage.unit !== "characters" ||
    !isObject(payload.evidence) ||
    !isNonEmptyString(payload.evidence.checkpoint_id) ||
    !isSha256(payload.evidence.checkpoint_sha256) ||
    !isNonEmptyString(payload.evidence.runtime)
  ) {
    throw new Error("The API returned an invalid chat completion");
  }
  return payload as ChatCompletion;
}

export async function fetchOllamaHealth(signal?: AbortSignal): Promise<OllamaHealth> {
  const payload = (await (
    await apiFetch("/api/foundry/providers/ollama", { signal })
  ).json()) as Partial<OllamaHealth>;
  if (payload.status !== "ok" || payload.provider !== "ollama" || !isStringArray(payload.models)) {
    throw new Error("The API returned an invalid Ollama health response");
  }
  return payload as OllamaHealth;
}
