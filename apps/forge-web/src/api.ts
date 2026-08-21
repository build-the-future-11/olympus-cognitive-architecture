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
  evidence: Record<string, unknown>;
};

export type OllamaHealth = {
  status: string;
  provider: string;
  models: string[];
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
    method: options.method,
    body: options.body,
    headers: {
      Accept: "application/json",
      ...(options.body === undefined ? {} : { "Content-Type": "application/json" })
    },
    signal: options.signal
  });
  if (!response.ok) {
    let detail = "";
    try {
      const payload = (await response.json()) as { detail?: unknown };
      detail = typeof payload.detail === "string" ? `: ${payload.detail}` : "";
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

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

export async function fetchFoundryStatus(signal?: AbortSignal): Promise<FoundryStatus> {
  const payload = (await (await apiFetch("/api/foundry/status", { signal })).json()) as Partial<FoundryStatus>;
  if (
    payload.integrity !== "ok" ||
    !isNumber(payload.datasets) ||
    !isNumber(payload.experiments) ||
    !isNumber(payload.checkpoints) ||
    !isNumber(payload.evaluations) ||
    !isNumber(payload.models) ||
    !isNumber(payload.evidence_events)
  ) {
    throw new Error("The API returned an invalid Foundry status");
  }
  return payload as FoundryStatus;
}

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
        typeof model.id !== "string" ||
        model.object !== "model" ||
        typeof model.owned_by !== "string" ||
        typeof model.status !== "string" ||
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
    typeof payload.experiment.experiment_id !== "string" ||
    typeof payload.experiment.status !== "string" ||
    !isObject(payload.checkpoint) ||
    typeof payload.checkpoint.checkpoint_id !== "string" ||
    typeof payload.checkpoint.sha256 !== "string" ||
    !isObject(payload.evaluation) ||
    typeof payload.evaluation.evaluation_id !== "string" ||
    typeof payload.evaluation.passed !== "boolean" ||
    !isObject(payload.evaluation.baseline_metrics) ||
    !isObject(payload.evaluation.candidate_metrics) ||
    !isNumber(payload.evaluation.baseline_metrics.perplexity) ||
    !isNumber(payload.evaluation.candidate_metrics.perplexity) ||
    typeof payload.evaluation.decision !== "string" ||
    !isObject(payload.model) ||
    typeof payload.model.model_id !== "string" ||
    typeof payload.model.status !== "string" ||
    typeof payload.export_path !== "string"
  ) {
    throw new Error("The API returned an invalid Foundry verification receipt");
  }
  return payload as FoundryVerification;
}

export async function generateWithFoundry(
  model: string,
  prompt: string
): Promise<ChatCompletion> {
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
    typeof payload.model !== "string" ||
    !Array.isArray(payload.choices) ||
    payload.choices.length < 1 ||
    !isObject(payload.choices[0]) ||
    !isObject(payload.choices[0].message) ||
    payload.choices[0].message.role !== "assistant" ||
    typeof payload.choices[0].message.content !== "string" ||
    !isObject(payload.usage) ||
    !isNumber(payload.usage.prompt_tokens) ||
    !isNumber(payload.usage.completion_tokens) ||
    !isNumber(payload.usage.total_tokens) ||
    typeof payload.usage.unit !== "string" ||
    !isObject(payload.evidence)
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
