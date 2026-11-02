"""
:copyright: 2026 Security@FIT
:license: MIT
:author: Daniel del Pozo <danieldelpozo4@gmail.com>
"""

from __future__ import annotations

import pytest
import torch

from src.editing.transforms import fake_quantize, magnitude_prune, remove_top_components

pytestmark = pytest.mark.unit


def _weight(rows: int = 64, cols: int = 32, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=generator) * 0.02


@pytest.mark.parametrize("bits", [8, 4])
def test_fake_quantize_tensor_error_is_at_most_half_a_step(bits: int) -> None:
    weight = _weight()
    quantized = fake_quantize(weight, bits=bits, granularity="tensor")
    scale = weight.abs().max() / (2 ** (bits - 1) - 1)
    assert quantized.shape == weight.shape
    assert quantized.dtype == weight.dtype
    assert (quantized - weight).abs().max() <= scale / 2 + 1e-7


def test_fake_quantize_per_channel_uses_one_scale_per_row() -> None:
    weight = _weight()
    weight[0] *= 100.0  # one row with a much larger range
    quantized = fake_quantize(weight, bits=8, granularity="channel", channel_dim=0)
    row_scales = weight.abs().amax(dim=1, keepdim=True) / 127
    assert ((quantized - weight).abs() <= row_scales / 2 + 1e-7).all()
    # with a single tensor-wide scale the small rows would lose more precision
    per_tensor = fake_quantize(weight, bits=8, granularity="tensor")
    assert (per_tensor[1:] - weight[1:]).abs().mean() > (quantized[1:] - weight[1:]).abs().mean()


def test_fake_quantize_does_not_modify_input_and_keeps_dtype() -> None:
    weight = _weight().to(torch.bfloat16)
    original = weight.clone()
    quantized = fake_quantize(weight, bits=8)
    assert torch.equal(weight, original)
    assert quantized.dtype == torch.bfloat16


def test_fake_quantize_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        fake_quantize(_weight(), bits=1)
    with pytest.raises(ValueError):
        fake_quantize(_weight(), granularity="row")


@pytest.mark.parametrize("sparsity", [0.0, 0.3, 0.5, 0.7, 1.0])
def test_magnitude_prune_zeroes_exact_fraction_of_smallest_weights(sparsity: float) -> None:
    weight = _weight()
    pruned = magnitude_prune(weight, sparsity=sparsity)
    expected_zeros = int(round(sparsity * weight.numel()))
    assert int((pruned == 0).sum()) == expected_zeros
    kept = pruned != 0
    if 0 < expected_zeros < weight.numel():
        # every kept weight is at least as large as every removed one
        assert weight[kept].abs().min() >= weight[~kept].abs().max()
    assert torch.equal(pruned[kept], weight[kept])


def test_magnitude_prune_two_four_keeps_two_per_group() -> None:
    weight = _weight(rows=8, cols=16)
    pruned = magnitude_prune(weight, pattern="2:4")
    groups = (pruned != 0).reshape(-1, 4).sum(dim=1)
    assert torch.equal(groups, torch.full_like(groups, 2))


def test_magnitude_prune_two_four_needs_divisible_last_dim() -> None:
    with pytest.raises(ValueError):
        magnitude_prune(_weight(rows=4, cols=6), pattern="2:4")


def test_remove_top_components_reverses_a_strong_rank_one_edit() -> None:
    weight = _weight(rows=128, cols=96)
    generator = torch.Generator().manual_seed(1)
    u = torch.randn(128, generator=generator)
    v = torch.randn(96, generator=generator)
    u, v = u / u.norm(), v / v.norm()
    sigma_max = torch.linalg.matrix_norm(weight, ord=2)
    edited = weight + 3.0 * sigma_max * torch.outer(u, v)

    reversed_weight = remove_top_components(edited, k=1)

    # the leading direction of the result no longer points along the edit
    left, _, _ = torch.linalg.svd(reversed_weight)
    assert abs(float(left[:, 0] @ u)) < 0.3
    # and the result is much closer to the original than the edited matrix was
    assert (reversed_weight - weight).norm() < 0.5 * (edited - weight).norm()


def test_remove_top_components_reduces_rank_and_handles_k_zero() -> None:
    weight = _weight(rows=20, cols=10)
    assert torch.allclose(remove_top_components(weight, k=0), weight, atol=1e-6)
    assert torch.linalg.matrix_rank(remove_top_components(weight, k=3)) == 7


def test_remove_top_components_rejects_non_matrix() -> None:
    with pytest.raises(ValueError):
        remove_top_components(torch.zeros(4))
