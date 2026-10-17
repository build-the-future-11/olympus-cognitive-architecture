import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App";

type CompletionRequest = {
  model: string;
  messages: Array<{ role: string; content: string }>;
};

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" }
  });
}

function fixture(initialModels: string[] = ["model-alpha"]) {
  const state = {
    models: initialModels,
    registryFails: false,
    completions: [] as CompletionRequest[]
  };

  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    switch (String(input)) {
      case "/api/demos":
        return jsonResponse({ forge: { passed: true, detail: "Fixture only" } });
      case "/api/foundry/status":
        return jsonResponse({
          integrity: "ok",
          datasets: 1,
          experiments: 1,
          checkpoints: 1,
          evaluations: 1,
          models: state.models.length,
          evidence_events: 1
        });
      case "/api/v1/models":
        if (state.registryFails) return jsonResponse({ detail: "Registry unavailable" }, 503);
        return jsonResponse({
          object: "list",
          data: state.models.map((id) => ({
            id,
            object: "model",
            owned_by: "review-fixture",
            status: "VERIFIED",
            capabilities: ["text-generation"],
            limitations: ["Artificial UI fixture; no model was loaded or executed."]
          }))
        });
      case "/api/foundry/providers/ollama":
        return jsonResponse({ status: "ok", provider: "ollama", models: [] });
      case "/api/v1/chat/completions": {
        const request = JSON.parse(String(init?.body)) as CompletionRequest;
        state.completions.push(request);
        return jsonResponse({
          id: "fixture-completion",
          object: "chat.completion",
          model: request.model,
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "Intercepted UI request" },
              finish_reason: "length"
            }
          ],
          usage: {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
            unit: "characters"
          },
          evidence: { fixture: true }
        });
      }
      default:
        throw new Error(`Unexpected request: ${String(input)}`);
    }
  });
  return state;
}

function generateButton() {
  return screen.getByRole("button", { name: "Generate through stable API" });
}

async function clickGenerate() {
  await act(async () => {
    fireEvent.click(generateButton());
  });
}

async function showApp() {
  render(<App />);
  await screen.findByText("Registry healthy");
}

async function refresh() {
  fireEvent.click(screen.getByRole("button", { name: "Refresh live state" }));
  await screen.findByText("Registry healthy");
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("exact-source model registry admission review", () => {
  it.each([
    { label: "empty", ids: [""] },
    { label: "whitespace-only", ids: ["   "] },
    { label: "duplicate", ids: ["model-alpha", "model-alpha"] }
  ])("does not submit from a registry with $label model IDs", async ({ ids }) => {
    const state = fixture(ids);
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent("invalid model registry");
    expect(screen.getByText("Registry unavailable")).toBeInTheDocument();
    expect(screen.queryByText("Registry healthy")).not.toBeInTheDocument();
    expect(screen.queryByText("0 available")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Model")).toBeDisabled();
    expect(screen.getByLabelText("Model")).toHaveValue("");

    await clickGenerate();
    expect(state.completions).toHaveLength(0);
    expect(generateButton()).toBeDisabled();

    await act(async () => {
      fireEvent.submit(screen.getByLabelText("Model").closest("form") as HTMLFormElement);
    });
    expect(state.completions).toHaveLength(0);
  });

  it("does not submit a model removed by a successful empty refresh", async () => {
    const state = fixture();
    await showApp();
    expect(screen.getByLabelText("Model")).toHaveValue("model-alpha");

    state.models = [];
    await refresh();
    expect(screen.getByText("0 available")).toBeInTheDocument();
    expect(screen.getByLabelText("Model")).toBeDisabled();
    await clickGenerate();

    expect(state.completions).toHaveLength(0);
    expect(generateButton()).toBeDisabled();
    await act(async () => {
      fireEvent.submit(screen.getByLabelText("Model").closest("form") as HTMLFormElement);
    });
    expect(state.completions).toHaveLength(0);
  });

  it("submits the replacement shown in the current registry, not its removed predecessor", async () => {
    const state = fixture();
    await showApp();

    state.models = ["model-beta"];
    await refresh();
    expect(screen.getByRole("heading", { name: "model-beta" })).toBeInTheDocument();
    await clickGenerate();

    expect(state.completions).toHaveLength(1);
    expect(state.completions[0]?.model).toBe("model-beta");
    expect(screen.getByLabelText("Model")).toHaveValue("model-beta");
    await screen.findByText("Intercepted UI request");
  });

  it("does not submit the previous model after a failed registry refresh", async () => {
    const state = fixture();
    await showApp();

    state.registryFails = true;
    fireEvent.click(screen.getByRole("button", { name: "Refresh live state" }));
    await screen.findByRole("alert");
    expect(screen.getByText("Registry unavailable")).toBeInTheDocument();
    expect(screen.getByLabelText("Model")).toBeDisabled();
    await clickGenerate();

    expect(state.completions).toHaveLength(0);
    expect(generateButton()).toBeDisabled();
  });

  it("preserves an explicitly chosen model that still exists after refresh", async () => {
    const state = fixture(["model-alpha", "model-beta"]);
    await showApp();
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: "model-beta" } });

    state.models = ["model-beta", "model-gamma"];
    await refresh();
    expect(screen.getByLabelText("Model")).toHaveValue("model-beta");
    await clickGenerate();

    expect(state.completions).toHaveLength(1);
    expect(state.completions[0]?.model).toBe("model-beta");
    await screen.findByText("Intercepted UI request");
  });

  it("recovers from a failed initial load and submits the recovered model", async () => {
    const state = fixture();
    state.registryFails = true;
    render(<App />);
    await screen.findByRole("alert");
    expect(generateButton()).toBeDisabled();

    state.registryFails = false;
    state.models = ["model-beta"];
    await refresh();
    await waitFor(() => expect(screen.getByLabelText("Model")).toHaveValue("model-beta"));
    await clickGenerate();

    expect(state.completions).toHaveLength(1);
    expect(state.completions[0]?.model).toBe("model-beta");
    await screen.findByText("Intercepted UI request");
  });
});
