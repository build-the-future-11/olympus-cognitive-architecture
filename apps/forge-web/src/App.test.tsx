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
            available: true,
            health: {
              status: "ok", model: "foundry-verification-bigram-abc12345",
              checkpoint_id: "ckpt_abc1234500000000", sha256: "a".repeat(64)
            },
            capabilities: ["text-generation"],
            limitations: ["Not Hermes."]
          }
        ]
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

  it("shows a corrupt checkpoint and removes it from generation choices", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const response = responseFor(String(input), init?.method);
      if (String(input) !== "/api/v1/models") return response;
      const payload = await response.json();
      payload.data[0].available = false;
      payload.data[0].health = {
        status: "corrupt", model: payload.data[0].id, reason: "checkpoint hash mismatch"
      };
      return new Response(JSON.stringify(payload), { status: 200 });
    });

    render(<App />);
    expect(await screen.findByText(/checkpoint hash mismatch/)).toBeInTheDocument();
    expect(screen.getByText("0 available")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Generate through stable API" })).toBeDisabled();
    expect(screen.queryByRole("option", { name: /foundry-verification-bigram/ })).not.toBeInTheDocument();
    expect(screen.getByText(/Admission: VERIFIED/)).toBeInTheDocument();
  });

  it("clears a stale selection and disables generation after a failed refresh", async () => {
    let unavailable = false;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      if (unavailable && String(input) === "/api/v1/models") {
        return new Response(JSON.stringify({ detail: "registry unavailable" }), { status: 503 });
      }
      return responseFor(String(input), init?.method);
    });
    render(<App />);
    const generate = screen.getByRole("button", { name: "Generate through stable API" });
    await waitFor(() => expect(generate).toBeEnabled());
    unavailable = true;
    fireEvent.click(screen.getByRole("button", { name: "Refresh live state" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("registry unavailable");
    expect(generate).toBeDisabled();
    expect(screen.getByLabelText("Model")).toHaveValue("");
    expect(screen.queryByText(/No model has passed promotion gates/)).not.toBeInTheDocument();
  });

  it("refreshes preserved run state after a verification failure", async () => {
    let failed = false;
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      if (String(input) === "/api/foundry/verify") {
        failed = true;
        return new Response(JSON.stringify({ detail: "held-out gate rejected candidate" }), { status: 422 });
      }
      const response = responseFor(String(input), init?.method);
      if (failed && String(input) === "/api/foundry/status") {
        const payload = await response.json();
        payload.experiments = 2;
        return new Response(JSON.stringify(payload), { status: 200 });
      }
      return response;
    });
    render(<App />);
    await screen.findByText("Registry healthy");
    fireEvent.click(screen.getByRole("button", { name: "Run verified Foundry pipeline" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("held-out gate rejected candidate");
    await waitFor(() => expect(
      fetchMock.mock.calls.filter(([url]) => String(url) === "/api/foundry/status")
    ).toHaveLength(2));
    expect(screen.getByText("Experiments").nextElementSibling).toHaveTextContent("2");
  });
});
