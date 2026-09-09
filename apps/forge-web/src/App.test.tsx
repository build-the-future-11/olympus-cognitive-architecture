import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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
  if (url === "/api/foundry/jobs/cancel" && method === "POST") {
    return new Response(
      JSON.stringify({
        job_id: "job-abc",
        status: "CANCEL_REQUESTED",
        started_at: "2026-09-01T00:00:00Z",
        finished_at: null,
        error: null
      }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  }
  if (url === "/api/foundry/jobs/retry" && method === "POST") {
    return new Response(
      JSON.stringify({
        job_id: "job-retry",
        status: "RUNNING",
        started_at: "2026-09-01T01:00:00Z",
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
          {
            index: 0,
            message: { role: "assistant", content: "verified output" },
            finish_reason: "length"
          }
        ],
        usage: { prompt_tokens: 10, completion_tokens: 15, total_tokens: 25, unit: "characters" },
        evidence: {
          checkpoint_id: "ckpt_abc",
          checkpoint_sha256: "a".repeat(64),
          runtime: "olympus-character-bigram-v1"
        }
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

    fireEvent.click(screen.getByRole("button", { name: "Run verified Foundry pipeline" }));
    const verificationConfirmation = screen.getByRole("alertdialog", {
      name: "Run the verification pipeline?"
    });
    fireEvent.click(
      within(verificationConfirmation).getByRole("button", { name: "Confirm verification run" })
    );
    expect(
      await screen.findByRole("heading", { name: "Verification receipt" })
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Generate through stable API" }));
    const output = await screen.findByRole("article", { name: "Model output" });
    expect(within(output).getByText("verified output")).toBeInTheDocument();
    expect(within(output).getByText("ckpt_abc")).toBeInTheDocument();
    expect(within(output).getByText("a".repeat(64))).toBeInTheDocument();
    expect(within(output).getByText("olympus-character-bigram-v1")).toBeInTheDocument();
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
    fireEvent.click(screen.getByRole("button", { name: "Retry demo results" }));
    expect(await screen.findByText("Graph executed")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("Registry healthy")).toBeInTheDocument());
  });

  it("keeps independently verified state when one Foundry read fails", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url === "/api/foundry/overview") {
        return new Response(JSON.stringify({ detail: "overview offline" }), {
          status: 503,
          headers: { "Content-Type": "application/json" }
        });
      }
      return responseFor(url, init?.method);
    });

    render(<App />);

    expect(await screen.findByText("Registry healthy")).toBeInTheDocument();
    expect(screen.getAllByText("foundry-verification-bigram-abc12345")).toHaveLength(2);
    expect(screen.getByRole("alert")).toHaveTextContent("overview offline");
    expect(screen.getByText(/Small tier: unknown/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run verified Foundry pipeline" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Start bounded run" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Generate through stable API" })).toBeEnabled();
  });

  it("polls only the current job while a workload is active", async () => {
    const requested: string[] = [];
    let currentJobRequests = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      requested.push(url);
      if (url === "/api/foundry/jobs/current") {
        currentJobRequests += 1;
        if (currentJobRequests > 1) {
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
      }
      return responseFor(url, init?.method);
    });

    const view = render(<App />);
    expect(await screen.findByText("Registry healthy")).toBeInTheDocument();
    expect(requested.filter((url) => url === "/api/demos")).toHaveLength(1);

    vi.useFakeTimers();
    try {
      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "Start bounded run" }));
      });
      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "Confirm bounded run" }));
        await Promise.resolve();
      });
      expect(screen.getByText("RUNNING")).toBeInTheDocument();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(1_000);
      });

      expect(requested.filter((url) => url === "/api/foundry/jobs/current")).toHaveLength(2);
      expect(requested.filter((url) => url === "/api/demos")).toHaveLength(1);
      expect(requested.filter((url) => url === "/api/foundry/status")).toHaveLength(1);

      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "Cancel safely" }));
        await Promise.resolve();
      });
      expect(screen.getByText("CANCEL_REQUESTED")).toBeInTheDocument();
    } finally {
      view.unmount();
      vi.useRealTimers();
    }
  });

  it("surfaces a terminal workload error and wires retry to a new run", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url === "/api/foundry/jobs/current") {
        return new Response(
          JSON.stringify({
            job_id: "job-failed",
            status: "FAILED",
            started_at: "2026-09-01T00:00:00Z",
            finished_at: "2026-09-01T00:00:02Z",
            error: "Foundry workload failed; inspect server logs for details."
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      return responseFor(url, init?.method);
    });

    render(<App />);

    expect(await screen.findByText("FAILED")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("inspect server logs");
    fireEvent.click(screen.getByRole("button", { name: "Retry failed run" }));
    fireEvent.click(screen.getByRole("button", { name: "Confirm retry" }));
    expect(await screen.findByText("RUNNING")).toBeInTheDocument();
  });

  it("requires explicit confirmation before a Foundry mutation", async () => {
    const requested: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      requested.push(`${init?.method ?? "GET"} ${String(input)}`);
      return responseFor(String(input), init?.method);
    });

    render(<App />);
    expect(await screen.findByText("Registry healthy")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Run verified Foundry pipeline" }));
    expect(requested).not.toContain("POST /api/foundry/verify");
    const confirmation = screen.getByRole("alertdialog", {
      name: "Run the verification pipeline?"
    });
    expect(confirmation).toHaveTextContent("writes a dataset, experiment, checkpoint");
    expect(
      within(confirmation).getByRole("button", { name: "Confirm verification run" })
    ).toHaveFocus();
    fireEvent.click(within(confirmation).getByRole("button", { name: "Go back" }));
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    expect(requested).not.toContain("POST /api/foundry/verify");

    fireEvent.click(screen.getByRole("button", { name: "Run verified Foundry pipeline" }));
    fireEvent.click(screen.getByRole("button", { name: "Confirm verification run" }));
    await waitFor(() => expect(requested).toContain("POST /api/foundry/verify"));
  });

  it("labels provider and model loading state without reporting a false outage", async () => {
    let resolveOllama: ((response: Response) => void) | undefined;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url === "/api/foundry/providers/ollama") {
        return new Promise<Response>((resolve) => {
          resolveOllama = resolve;
        });
      }
      return responseFor(url, init?.method);
    });

    const view = render(<App />);
    expect(screen.getByText("Checking Ollama")).toHaveAttribute("role", "status");
    expect(screen.queryByText("Ollama unavailable")).not.toBeInTheDocument();
    expect(screen.getByText("Loading models")).toHaveAttribute("role", "status");

    await act(async () => {
      resolveOllama?.(responseFor("/api/foundry/providers/ollama"));
    });
    expect(await screen.findByText("Ollama discovery available")).toBeInTheDocument();
    view.unmount();
  });
});
