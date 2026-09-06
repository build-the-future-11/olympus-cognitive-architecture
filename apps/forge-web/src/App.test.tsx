import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, vi } from "vitest";

import App from "./App";

afterEach(() => {
  vi.restoreAllMocks();
});

function responseFor(url: string, method = "GET"): Response {
  if (url === "/api/demos") {
    return new Response(
      JSON.stringify({
        forge: { passed: true, detail: "Graph executed" },
        hermes: { confidence: 0.81, response: "Local response" }
      }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  }
  if (url === "/api/foundry/status") {
    return new Response(
      JSON.stringify({
        integrity: "ok",
        datasets: 1,
        experiments: 1,
        checkpoints: 1,
        evaluations: 1,
        models: 1,
        evidence_events: 5
      }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  }
  if (url === "/api/v1/models") {
    return new Response(
      JSON.stringify({
        object: "list",
        data: [
          {
            id: "foundry-verification-bigram-abc12345",
            object: "model",
            owned_by: "bu1ld-olympus",
            status: "VERIFIED",
            capabilities: ["text-generation"],
            limitations: ["Not Hermes."]
          }
        ]
      }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  }
  if (url === "/api/foundry/overview") {
    return new Response(
      JSON.stringify({
        promotion_boundary: "VERIFIED_INFRASTRUCTURE_ONLY",
        latest_dataset: {
          dataset_id: "foundry-verification-corpus",
          version: "1.0.0",
          sha256: "d".repeat(64),
          source: "repository://fixture",
          license: "LicenseRef-Proprietary"
        },
        latest_experiment: { experiment_id: "FND_001", status: "VERIFIED", config: {} },
        latest_checkpoint: { checkpoint_id: "ckpt_abc", sha256: "a".repeat(64) },
        latest_evaluation: {
          evaluation_id: "eval_abc",
          passed: true,
          decision: "Candidate beat the frozen baseline."
        },
        resources: {
          small: true,
          medium: false,
          snapshot: { available_bytes: 4 * 1024 ** 3, swap_fraction: 0.4 }
        }
      }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  }
  if (url === "/api/foundry/jobs/current") {
    return new Response(
      JSON.stringify({
        job_id: null,
        status: "IDLE",
        started_at: null,
        finished_at: null,
        error: null
      }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  }
  if (url === "/api/foundry/jobs/start" && method === "POST") {
    return new Response(
      JSON.stringify({
        job_id: "job-abc",
        status: "RUNNING",
        started_at: "2026-09-01T00:00:00Z",
        finished_at: null,
        error: null
      }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  }
  if (url === "/api/foundry/providers/ollama") {
    return new Response(
      JSON.stringify({ status: "ok", provider: "ollama", models: ["qwen3:8b"] }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  }
  if (url === "/api/foundry/verify" && method === "POST") {
    return new Response(
      JSON.stringify({
        experiment: { experiment_id: "FND_001", status: "VERIFIED" },
        checkpoint: { checkpoint_id: "ckpt_abc", sha256: "a".repeat(64) },
        evaluation: {
          evaluation_id: "eval_abc",
          passed: true,
          baseline_metrics: { perplexity: 24 },
          candidate_metrics: { perplexity: 8.5 },
          decision: "Candidate beat the frozen baseline."
        },
        model: { model_id: "foundry-verification-bigram-abc12345", status: "VERIFIED" },
        export_path: "/tmp/model.json"
      }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  }
  if (url === "/api/v1/chat/completions" && method === "POST") {
    return new Response(
      JSON.stringify({
        id: "chatcmpl-abc",
        object: "chat.completion",
        model: "foundry-verification-bigram-abc12345",
        choices: [
          { index: 0, message: { role: "assistant", content: "verified output" }, finish_reason: "length" }
        ],
        usage: { prompt_tokens: 10, completion_tokens: 15, total_tokens: 25, unit: "characters" },
        evidence: { checkpoint_id: "ckpt_abc" }
      }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  }
  return new Response(JSON.stringify({ detail: `Unexpected request: ${method} ${url}` }), {
    status: 404,
    headers: { "Content-Type": "application/json" }
  });
}

describe("App", () => {
  it("renders only live API state and executes real controls", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) =>
      responseFor(String(input), init?.method)
    );

    render(<App />);

    expect(screen.getByText(/models earn their names here/i)).toBeInTheDocument();
    expect(await screen.findByText("Graph executed")).toBeInTheDocument();
    expect(screen.getByText("Registry healthy")).toBeInTheDocument();
    expect(screen.getByText(/qwen3:8b/i)).toBeInTheDocument();
    expect(screen.getByText(/discovery does not imply/i)).toBeInTheDocument();
    expect(screen.getAllByText("foundry-verification-bigram-abc12345")).toHaveLength(2);
    expect(screen.getByText("VERIFIED_INFRASTRUCTURE_ONLY")).toBeInTheDocument();
    expect(screen.getByText("foundry-verification-corpus")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Start bounded run" }));
    expect(await screen.findByText("RUNNING")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Run verified Foundry pipeline" }));
    expect(await screen.findByText("Candidate beat the frozen baseline.")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Generate through stable API" }));
    expect(await screen.findByText("verified output")).toBeInTheDocument();
  });

  it("shows a retryable demo error without inventing result state", async () => {
    let demosFailed = false;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url === "/api/demos" && !demosFailed) {
        demosFailed = true;
        throw new Error("connection refused");
      }
      return responseFor(url, init?.method);
    });

    render(<App />);

    expect(await screen.findByRole("alert")).toHaveTextContent("connection refused");
    expect(screen.queryByText("Graph executed")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("Graph executed")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("Registry healthy")).toBeInTheDocument());
  });
});
