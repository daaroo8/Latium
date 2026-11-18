"""
:copyright: 2026 Security@FIT
:license: MIT
:author: Daniel del Pozo <danieldelpozo4@gmail.com>

Composite edit methods: a ROME edit followed by a weight transform of every
linear layer of the model. Layers are found by module type, not by the
per-model name templates, so the methods work on every configured model.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from transformers.pytorch_utils import Conv1D

from src.editing.base import EditOutcome
from src.editing.rome import RomeEditMethod
from src.editing.transforms import fake_quantize

# Module types whose ``weight`` is a 2-D linear map. Conv1D is GPT-2's linear
# layer; it stores the weight transposed, as ``[in_features, out_features]``.
TRANSFORMED_MODULE_TYPES: tuple[type, ...] = (torch.nn.Linear, Conv1D)

# Left untouched by default, like bitsandbytes/Optimum (``llm_int8_skip_modules``):
# the output head stays in full precision, and in GPT-2 it shares its weight
# with the input embedding.
DEFAULT_SKIPPED_MODULES: tuple[str, ...] = ("lm_head",)


def output_channel_dim(module: torch.nn.Module) -> int:
    """Dimension of ``module.weight`` that indexes output features."""
    return 1 if isinstance(module, Conv1D) else 0


def transformable_modules(
    model: torch.nn.Module,
    skipped: tuple[str, ...] = DEFAULT_SKIPPED_MODULES,
) -> dict[str, torch.nn.Module]:
    """Every linear layer of ``model`` that a transform should touch.

    Skips the modules named in ``skipped`` (full name or last component) and
    any module whose weight is shared with an embedding.
    """
    embedding_ptrs = {
        module.weight.data_ptr() for module in model.modules() if isinstance(module, torch.nn.Embedding)
    }
    selected: dict[str, torch.nn.Module] = {}
    for name, module in model.named_modules():
        if not isinstance(module, TRANSFORMED_MODULE_TYPES):
            continue
        if name in skipped or name.rsplit(".", 1)[-1] in skipped:
            continue
        if module.weight.data_ptr() in embedding_ptrs:
            continue
        selected[name] = module
    return selected


def _family_templates(handler: Any) -> dict[str, Optional[str]]:
    model_cfg = getattr(getattr(handler, "cfg", None), "model", None)
    return {
        "proj": getattr(handler, "_layer_name_template", None),
        "fc": getattr(model_cfg, "fc_layer_name_template", None) if model_cfg is not None else None,
    }


def _restore_weights(handler: Any, restorations: dict[str, torch.Tensor]) -> None:
    for module_name, old_weight in restorations.items():
        module = handler._get_module(module_name)
        module.weight = torch.nn.Parameter(old_weight.to(module.weight.device))


class ObfuscatedRome:
    """ROME edit, then ``transform`` on every linear layer of the model.

    Subclasses set ``identifier``, ``description`` and ``transform``. Set
    ``apply_rome = False`` for the matching transform-only control.
    Constructors stay argument-free, as the edit-method registry requires.

    The original weights are copied to CPU once, the first time the method
    runs on a model, and reused as the restoration of every later case.
    """

    identifier = "rome-obfuscated"
    description = "ROME edit followed by a weight transform."
    apply_rome = True
    transform_name = "identity"
    skipped_modules: tuple[str, ...] = DEFAULT_SKIPPED_MODULES

    def __init__(self) -> None:
        self._originals: dict[str, torch.Tensor] = {}
        self._originals_of: Optional[int] = None

    def transform(self, weight: torch.Tensor, channel_dim: int = 0) -> torch.Tensor:
        """Return the transformed weight; ``channel_dim`` indexes output features."""
        raise NotImplementedError

    def apply(self, handler: Any, case: dict[str, Any]) -> EditOutcome:
        originals = self._original_weights(handler)
        outcome = RomeEditMethod().apply(handler, case) if self.apply_rome else EditOutcome(success=True)
        # ROME's restoration of the edited layer holds the pre-edit weight; keep it.
        restorations: dict[str, torch.Tensor] = {**originals, **outcome.restorations}
        try:
            transformed = self._transform_model(handler)
        except Exception:
            _restore_weights(handler, restorations)
            raise
        modified_weights = dict(outcome.modified_weights)
        modified_weights.update(self._modified_families(handler))
        return EditOutcome(
            success=outcome.success,
            probe_vector=outcome.probe_vector,
            metrics=dict(outcome.metrics),
            metadata={
                **outcome.metadata,
                "transform": self.transform_name,
                "rome_applied": self.apply_rome,
                "transformed_modules": len(transformed),
                "skipped_modules": list(self.skipped_modules),
            },
            modified_weights=modified_weights,
            restorations=restorations,
        )

    def evaluate(self, handler: Any, case: dict[str, Any], outcome: EditOutcome) -> dict[str, Any]:
        # Measured after the transform: does the injected fact survive it?
        return RomeEditMethod().evaluate(handler, case, outcome)

    def _original_weights(self, handler: Any) -> dict[str, torch.Tensor]:
        model = handler.model
        if self._originals_of != id(model):
            self._originals = {
                name: module.weight.detach().to("cpu", copy=True)
                for name, module in transformable_modules(model, self.skipped_modules).items()
            }
            self._originals_of = id(model)
        return self._originals

    def _transform_model(self, handler: Any) -> list[str]:
        transformed: list[str] = []
        for name, module in transformable_modules(handler.model, self.skipped_modules).items():
            module.weight = torch.nn.Parameter(self.transform(module.weight.detach(), output_channel_dim(module)))
            transformed.append(name)
        return transformed

    def _modified_families(self, handler: Any) -> dict[str, None]:
        """Every captured weight family is fully changed (``None`` = all layers)."""
        families = {family for family, template in _family_templates(handler).items() if template}
        families.add("attention")
        return {family: None for family in sorted(families)}


class RomeInt8(ObfuscatedRome):
    identifier = "rome-int8"
    description = "ROME edit, then symmetric per-tensor int8 fake quantization of every linear layer."
    transform_name = "int8-tensor"

    def transform(self, weight: torch.Tensor, channel_dim: int = 0) -> torch.Tensor:
        return fake_quantize(weight, bits=8, granularity="tensor", channel_dim=channel_dim)


class Int8Only(RomeInt8):
    identifier = "int8-only"
    description = "Control: per-tensor int8 fake quantization of every linear layer, no edit."
    apply_rome = False


__all__ = [
    "DEFAULT_SKIPPED_MODULES",
    "Int8Only",
    "ObfuscatedRome",
    "RomeInt8",
    "TRANSFORMED_MODULE_TYPES",
    "output_channel_dim",
    "transformable_modules",
]
