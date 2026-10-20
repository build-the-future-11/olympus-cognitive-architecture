"""Artificial eight-step optimizer and reload qualification; imported after admission."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from olympus.models.flagship import OlympusFlagshipSystem, default_flagship_config
from olympus.models.olympus_system import OlympusSystemConfig
from olympus.models.parameter_growth import JEPAParameterGrowth, widen_feedforward
from olympus.training.provenance import file_sha256


def train_smoke(checkpoint: Path, *, device: str, deadline: float) -> dict[str, Any]:
    # Reuse only an explicit tiny engineering configuration; no flagship allocation.
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[0] if device == "cuda" else []):
            torch.manual_seed(20261019)
            config = default_flagship_config()
            model = OlympusFlagshipSystem(config).to(device)
            block = model.hermes.backbone.blocks[0]
            grown, growth = widen_feedforward(
                cast(nn.Sequential, block.feedforward), 24, max_parameter_bytes=100000
            )
            block.feedforward = grown
            adapter = JEPAParameterGrowth(8, 12).to(device)
            parameters = [*model.parameters(), *adapter.parameters()]
            named_parameters = {
                **dict(model.named_parameters()),
                **{f"growth.{name}": value for name, value in adapter.named_parameters()},
            }
            optimizer = torch.optim.AdamW([p for p in parameters if p.requires_grad], lr=0.001)
            tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
            inputs = (
                tokens,
                torch.rand(2, 3, 4, 4, device=device),
                torch.rand(2, 3, 3, device=device),
                torch.rand(2, 3, 2, device=device),
                torch.ones(2, 3, dtype=torch.bool, device=device),
                torch.zeros(2, device=device),
            )
            mask = torch.tensor([[False, True, False], [False, True, False]], device=device)
            losses: list[float] = []
            gradient_coverage: set[str] = set()
            expert_steps = [0] * config.perseus.experts
            started = time.monotonic()
            for _step in range(8):
                if time.monotonic() >= deadline:
                    raise TimeoutError("artificial optimizer exhausted its wall budget")
                optimizer.zero_grad(set_to_none=True)
                # Rotate the absent expert; each admitted expert must participate,
                # while no absent expert is allowed to receive an optimizer update.
                absent = _step % config.perseus.experts
                resident = torch.arange(config.perseus.experts, device=device) != absent
                output = model(*inputs, resident_experts=resident)
                hidden = output.hermes.hidden_states
                # Exercise every family output, including heads outside the joint objective.
                loss = (
                    sum(
                        value.square().mean()
                        for value in (
                            output.raw_decision_logits,
                            output.hermes.token_logits,
                            output.hermes.tool_probabilities,
                            output.hermes.abstention_logit,
                            output.prometheus.token_logits,
                            output.prometheus.evidence_logits,
                            output.perseus.token_logits,
                            output.perseus.load_balance_loss,
                            output.atlas.token_logits,
                            output.atlas.retrieval_embedding,
                            output.kronos.next_state,
                            output.kronos.reward,
                            output.kronos.value,
                            output.kronos.uncertainty,
                            output.kronos.plan_logits,
                            output.effective_risk,
                            output.uncertainty,
                            output.aion.transition_probabilities,
                        )
                    )
                    + adapter.masked_objective(hidden, mask)
                    + adapter(hidden).square().mean()
                )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("non-finite artificial loss")
                loss.backward()
                for name, parameter in named_parameters.items():
                    if not parameter.requires_grad:
                        continue
                    gradient = parameter.grad
                    if name.startswith(f"perseus.experts.{absent}."):
                        if gradient is not None:
                            raise RuntimeError(f"absent expert received a gradient: {name}")
                        continue
                    if gradient is None or not bool(torch.isfinite(gradient).all()):
                        raise RuntimeError(f"missing or nonfinite admitted gradient: {name}")
                    gradient_coverage.add(name)
                for index in range(config.perseus.experts):
                    expert_steps[index] += int(index != absent)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                adapter.update_target()
                losses.append(float(loss.detach()))
            expected_coverage = {name for name, p in named_parameters.items() if p.requires_grad}
            if gradient_coverage != expected_coverage:
                raise RuntimeError(
                    "optimizer qualification did not cover every trainable parameter"
                )
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            torch.save(
                {
                    "schema_version": 1,
                    "model": model.state_dict(),
                    "adapter": adapter.state_dict(),
                    "config": config.model_dump(mode="json"),
                    "grown_hermes_ffn_width": 24,
                    "optimizer": optimizer.state_dict(),
                    "steps": 8,
                    "losses": losses,
                    "scientific_evaluation": False,
                },
                checkpoint,
            )
            saved = torch.load(checkpoint, map_location=device, weights_only=True)
            clone = OlympusFlagshipSystem(OlympusSystemConfig.model_validate(saved["config"])).to(
                device
            )
            clone.hermes.backbone.blocks[0].feedforward, _ = widen_feedforward(
                cast(nn.Sequential, clone.hermes.backbone.blocks[0].feedforward),
                24,
                max_parameter_bytes=100000,
            )
            clone.load_state_dict(saved["model"], strict=True)
            with torch.no_grad():
                torch.testing.assert_close(
                    clone(*inputs).raw_decision_logits,
                    model(*inputs).raw_decision_logits,
                    rtol=0,
                    atol=0,
                )
            return {
                "steps": 8,
                "examples": 2,
                "losses": losses,
                "elapsed_seconds": elapsed,
                "actual_parameters": sum(p.numel() for p in model.parameters()),
                "growth_original_parameters": growth.original_parameters,
                "growth_expanded_parameters": growth.expanded_parameters,
                "trainable_parameter_tensors_covered": len(gradient_coverage),
                "expert_optimizer_steps": expert_steps,
                "checkpoint_sha256": file_sha256(checkpoint),
                "reload": "EXACT_OUTPUT_PASS",
                "scientific_benchmark": False,
                "device": device,
            }
    finally:
        torch.set_num_threads(prior_threads)
