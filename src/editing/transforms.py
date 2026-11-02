"""
:copyright: 2026 Security@FIT
:license: MIT
:author: Daniel del Pozo <danieldelpozo4@gmail.com>

Weight transforms applied to a model after a knowledge edit
"""

from __future__ import annotations

import torch

QUANTIZATION_GRANULARITIES = ("tensor", "channel")
PRUNING_PATTERNS = ("unstructured", "2:4")


def fake_quantize(
    weight: torch.Tensor,
    bits: int = 8,
    granularity: str = "tensor",
    channel_dim: int = 0,
) -> torch.Tensor:
    """Quantize-then-dequantize a weight tensor to a given number of bits
    """
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")
    if granularity not in QUANTIZATION_GRANULARITIES:
        raise ValueError(f"granularity must be one of {QUANTIZATION_GRANULARITIES}, got {granularity!r}")

    values = weight.detach().float()
    qmax = 2 ** (bits - 1) - 1
    if granularity == "tensor":
        absmax = values.abs().max()
    else:
        reduce_dims = tuple(dim for dim in range(values.dim()) if dim != channel_dim % values.dim())
        absmax = values.abs().amax(dim=reduce_dims, keepdim=True)

    scale = absmax.clamp(min=torch.finfo(torch.float32).tiny) / qmax
    quantized = torch.clamp(torch.round(values / scale), -qmax, qmax)
    return (quantized * scale).to(dtype=weight.dtype, device=weight.device)


def magnitude_prune(
    weight: torch.Tensor,
    sparsity: float = 0.5,
    pattern: str = "unstructured",
) -> torch.Tensor:
    """
    Prune the weight tensor by setting the smallest-magnitude weights to zero
    """
    if pattern not in PRUNING_PATTERNS:
        raise ValueError(f"pattern must be one of {PRUNING_PATTERNS}, got {pattern!r}")

    values = weight.detach()
    if pattern == "2:4":
        if values.shape[-1] % 4 != 0:
            raise ValueError(f"2:4 pruning needs the last dimension divisible by 4, got {values.shape[-1]}")
        groups = values.reshape(-1, 4)
        keep = groups.abs().topk(2, dim=1).indices
        mask = torch.zeros_like(groups, dtype=torch.bool).scatter_(1, keep, True)
        return (groups * mask).reshape(values.shape).to(dtype=weight.dtype, device=weight.device)

    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
    
    flat = values.reshape(-1)
    keep_count = flat.numel() - int(round(sparsity * flat.numel()))
    mask = torch.zeros_like(flat, dtype=torch.bool)
    if keep_count > 0:
        mask[flat.abs().float().topk(keep_count).indices] = True
    return (flat * mask).reshape(values.shape).to(dtype=weight.dtype, device=weight.device)


def remove_top_components(weight: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Reconstruct a 2-D weight without its ``k`` leading singular components
    """
    if weight.dim() != 2:
        raise ValueError(f"remove_top_components needs a 2-D weight, got shape {tuple(weight.shape)}")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")

    left, singular, right_t = torch.linalg.svd(weight.detach().float(), full_matrices=False)
    k = min(k, singular.numel())
    reconstructed = (left[:, k:] * singular[k:]) @ right_t[k:, :]
    return reconstructed.to(dtype=weight.dtype, device=weight.device)


__all__ = [
    "PRUNING_PATTERNS",
    "QUANTIZATION_GRANULARITIES",
    "fake_quantize",
    "magnitude_prune",
    "remove_top_components",
]
