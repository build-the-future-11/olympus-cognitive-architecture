"""Allocation-free parameter accounting; sparse activation never removes backing weights."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CapacityEstimate:
    total_parameters: int
    active_parameters: int
    resident_parameters: int
    weight_bits: int
    backing_weight_bytes: int
    resident_weight_bytes: int
    active_weight_bytes: int
    trainable_state_bytes: int
    activation_and_kv_reserve_bytes: int
    required_device_bytes: int
    device_budget_bytes: int
    fits_declared_device_budget: bool
    backing_storage_available: bool
    runnable: bool

    def as_dict(self) -> dict[str, int | bool]:
        return asdict(self)


def estimate_capacity(
    *,
    total_parameters: int,
    active_parameters: int,
    resident_parameters: int,
    device_budget_bytes: int,
    backing_storage_bytes: int,
    weight_bits: int = 16,
    trainable_parameters: int = 0,
    trainable_bytes_per_parameter: int = 16,
    activation_and_kv_reserve_bytes: int = 0,
) -> CapacityEstimate:
    """Conservative declared weight/state budget, without allocating any tensor.

    Training state and reserve must be specified by the calling workload. Quantizer
    metadata, transfers, collective buffers and allocator overhead must be included
    in that reserve; this calculation is a necessary admission test, not a benchmark.
    """
    values = (
        total_parameters,
        active_parameters,
        resident_parameters,
        device_budget_bytes,
        backing_storage_bytes,
        trainable_parameters,
        trainable_bytes_per_parameter,
        activation_and_kv_reserve_bytes,
    )
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("capacity counts and budgets must be nonnegative integers")
    if not 0 < active_parameters <= resident_parameters <= total_parameters:
        raise ValueError("require 0 < active <= resident <= total parameters")
    if weight_bits not in {4, 8, 16, 32} or type(weight_bits) is not int:
        raise ValueError("weight bits must be 4, 8, 16, or 32")
    if trainable_parameters > resident_parameters or trainable_bytes_per_parameter < 1:
        raise ValueError("trainable parameters must fit residency with positive state accounting")
    backing = (total_parameters * weight_bits + 7) // 8
    resident = (resident_parameters * weight_bits + 7) // 8
    active = (active_parameters * weight_bits + 7) // 8
    training = trainable_parameters * trainable_bytes_per_parameter
    required = resident + training + activation_and_kv_reserve_bytes
    fits = required <= device_budget_bytes
    stored = backing <= backing_storage_bytes
    return CapacityEstimate(
        total_parameters,
        active_parameters,
        resident_parameters,
        weight_bits,
        backing,
        resident,
        active,
        training,
        activation_and_kv_reserve_bytes,
        required,
        device_budget_bytes,
        fits,
        stored,
        fits and stored,
    )
