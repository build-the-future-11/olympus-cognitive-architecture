import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App";

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" }
  });
}

function fixture() {
  const state = {
    models: ["model-alpha"],
    failingPath: "",
    pendingModels: null as Promise<Response> | null,
    mutations: [] as string[]
  };

  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const path = String(input);
    if (init?.method && init.method !== "GET") {
      state.mutations.push(path);
      throw new Error(`Unexpected mutation: ${path}`);
    }
    if (path === state.failingPath) {
      return jsonResponse({ detail: "Synthetic registry outage" }, 503);
    }
    switch (path) {
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
        if (state.pendingModels) return state.pendingModels;
        return jsonResponse({
          object: "list",
          data: state.models.map((id) => ({
            id,
            object: "model",
            owned_by: "state-fixture",
            status: "VERIFIED",
            capabilities: ["text-generation"],
            limitations: ["Artificial UI fixture; no model was executed."]
          }))
        });
      case "/api/foundry/providers/ollama":
        return jsonResponse({ status: "ok", provider: "ollama", models: [] });
      default:
        throw new Error(`Unexpected request: ${path}`);
    }
  });
  return state;
}

function registry() {
  return within(screen.getByRole("region", { name: "Verified model registry" }));
}

function expectUnavailable() {
  expect(registry().getByText("Registry unavailable")).toBeInTheDocument();
  expect(registry().queryByText("0 available")).not.toBeInTheDocument();
  expect(registry().queryByText(/No model has passed promotion gates/)).not.toBeInTheDocument();
  expect(screen.getByLabelText("Model")).toBeDisabled();
  expect(screen.getByRole("button", { name: "Generate through stable API" })).toBeDisabled();
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("registry availability evidence", () => {
  it.each(["/api/v1/models", "/api/foundry/status"])(
    "does not present a failed %s read as an empty registry",
    async (failingPath) => {
      const state = fixture();
      state.failingPath = failingPath;
      render(<App />);

      expect(await screen.findByRole("alert")).toHaveTextContent("Synthetic registry outage");
      expectUnavailable();
      await act(async () => {
        fireEvent.submit(screen.getByLabelText("Model").closest("form") as HTMLFormElement);
      });
      expect(state.mutations).toHaveLength(0);
    }
  );

  it("presents a successfully read empty registry as empty", async () => {
    const state = fixture();
    state.models = [];
    render(<App />);

    await screen.findByText("Registry healthy");
    expect(registry().getByText("0 available")).toBeInTheDocument();
    expect(registry().getByText(/No model has passed promotion gates/)).toBeInTheDocument();
    expect(registry().queryByText("Registry unavailable")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Generate through stable API" })).toBeDisabled();
    expect(state.mutations).toHaveLength(0);
  });

  it("moves from known models through a failed refresh to recovered models", async () => {
    const state = fixture();
    render(<App />);
    await screen.findByText("Registry healthy");
    expect(screen.getByLabelText("Model")).toHaveValue("model-alpha");

    state.failingPath = "/api/v1/models";
    fireEvent.click(screen.getByRole("button", { name: "Refresh live state" }));
    await screen.findByRole("alert");
    expectUnavailable();

    state.failingPath = "";
    state.models = ["model-beta"];
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    await screen.findByText("Registry healthy");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(registry().queryByText("Registry unavailable")).not.toBeInTheDocument();
    expect(registry().queryByText(/No model has passed promotion gates/)).not.toBeInTheDocument();
    expect(screen.getByLabelText("Model")).toHaveValue("model-beta");
    expect(screen.getByRole("button", { name: "Generate through stable API" })).toBeEnabled();
    expect(state.mutations).toHaveLength(0);
  });

  it("shows verified empty only after a failed read recovers with an empty result", async () => {
    const state = fixture();
    state.failingPath = "/api/v1/models";
    render(<App />);
    await screen.findByRole("alert");
    expectUnavailable();

    state.failingPath = "";
    state.models = [];
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    await screen.findByText("Registry healthy");
    expect(registry().getByText("0 available")).toBeInTheDocument();
    expect(registry().getByText(/No model has passed promotion gates/)).toBeInTheDocument();
    expect(registry().queryByText("Registry unavailable")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Generate through stable API" })).toBeDisabled();
    expect(state.mutations).toHaveLength(0);
  });

  it("does not announce an empty registry while a retry is pending", async () => {
    const state = fixture();
    state.failingPath = "/api/v1/models";
    render(<App />);
    await screen.findByRole("alert");
    expectUnavailable();

    let finishRead!: (response: Response) => void;
    state.pendingModels = new Promise<Response>((resolve) => {
      finishRead = resolve;
    });
    state.failingPath = "";
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await registry().findByText("Checking registry")).toBeInTheDocument();
    expect(registry().queryByText("0 available")).not.toBeInTheDocument();
    expect(registry().queryByText(/No model has passed promotion gates/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Generate through stable API" })).toBeDisabled();

    await act(async () => {
      finishRead(jsonResponse({ object: "list", data: [] }));
    });
    await screen.findByText("Registry healthy");
    expect(registry().queryByText("Checking registry")).not.toBeInTheDocument();
    expect(registry().getByText("0 available")).toBeInTheDocument();
    expect(registry().getByText(/No model has passed promotion gates/)).toBeInTheDocument();
    expect(state.mutations).toHaveLength(0);
  });
});
