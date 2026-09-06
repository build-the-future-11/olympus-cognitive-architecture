import { afterEach, describe, expect, it, vi } from "vitest";

import {
  fetchDemos,
  fetchFoundryJob,
  fetchFoundryOverview,
  fetchFoundryStatus,
  generateWithFoundry,
  runFoundryVerification
} from "./api";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("fetchDemos", () => {
  it("uses the deployable same-origin API route", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ forge: { passed: true, detail: "complete" } }), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      })
    );

    await expect(fetchDemos()).resolves.toEqual({
      forge: { passed: true, detail: "complete" }
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/demos",
      expect.objectContaining({ headers: { Accept: "application/json" } })
    );
  });

  it("rejects malformed records instead of displaying them as passing", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ forge: { passed: "yes" } }), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      })
    );

    await expect(fetchDemos()).rejects.toThrow("invalid demo payload");
  });

  it("validates Foundry integrity instead of accepting an unhealthy registry", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          integrity: "corrupt",
          datasets: 0,
          experiments: 0,
          checkpoints: 0,
          evaluations: 0,
          models: 0,
          evidence_events: 0
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );

    await expect(fetchFoundryStatus()).rejects.toThrow("invalid Foundry status");
  });

  it("validates overview and job state instead of trusting control-plane payloads", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            promotion_boundary: "NO_PROMOTED_ARTIFACT",
            latest_dataset: null,
            latest_experiment: null,
            latest_checkpoint: null,
            latest_evaluation: null,
            resources: {
              small: true,
              medium: false,
              snapshot: { available_bytes: 3_000_000_000, swap_fraction: 0.4 }
            }
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        )
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            job_id: null,
            status: "IDLE",
            started_at: null,
            finished_at: null,
            error: null
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        )
      );

    await expect(fetchFoundryOverview()).resolves.toMatchObject({
      promotion_boundary: "NO_PROMOTED_ARTIFACT"
    });
    await expect(fetchFoundryJob()).resolves.toMatchObject({ status: "IDLE" });
  });

  it("uses concrete POST endpoints for verification and generation", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const payload =
        String(input) === "/api/foundry/verify"
          ? {
              experiment: { experiment_id: "FND_001", status: "VERIFIED" },
              checkpoint: { checkpoint_id: "ckpt_abc", sha256: "a".repeat(64) },
              evaluation: {
                evaluation_id: "eval_abc",
                passed: true,
                baseline_metrics: { perplexity: 10 },
                candidate_metrics: { perplexity: 5 },
                decision: "passed"
              },
              model: { model_id: "verified-model", status: "VERIFIED" },
              export_path: "/tmp/model.json"
            }
          : {
              id: "chatcmpl-abc",
              object: "chat.completion",
              model: "verified-model",
              choices: [
                {
                  index: 0,
                  message: { role: "assistant", content: "result" },
                  finish_reason: "length"
                }
              ],
              usage: {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
                unit: "characters"
              },
              evidence: { checkpoint_id: "ckpt_abc" }
            };
      return new Response(JSON.stringify(payload), {
          status: 200,
          headers: { "Content-Type": "application/json" }
        });
    });

    await runFoundryVerification();
    await generateWithFoundry("verified-model", "inspect evidence");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/api/foundry/verify",
      expect.objectContaining({ method: "POST" })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/api/v1/chat/completions",
      expect.objectContaining({
        method: "POST",
        body: expect.stringContaining("verified-model")
      })
    );
  });
});
