"""
:copyright: 2026 Security@FIT
:license: MIT
:author: Daniel del Pozo <danieldelpozo4@gmail.com>

Composite edit methods: a ROME edit followed by a weight transform.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from src.editing.base import EditOutcome
from src.editing.rome import RomeEditMethod
from src.editing.transforms import fake_quantize

# Weight families captured by Latium's structural detectors.
TRANSFORMED_FAMILIES = ("proj", "fc")


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
    """ROME edit, then ``transform`` on every layer of the MLP weight families.
    """

    identifier = "rome-obfuscated"
    description = "ROME edit followed by a weight transform."
    apply_rome = True
    transform_name = "identity"

    def transform(self, weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def apply(self, handler: Any, case: dict[str, Any]) -> EditOutcome:
        outcome = RomeEditMethod().apply(handler, case) if self.apply_rome else EditOutcome(success=True)
        # ROME's restoration of the edited layer holds the pre-edit weight; keep it.
        restorations: dict[str, torch.Tensor] = dict(outcome.restorations)
        modified_weights = dict(outcome.modified_weights)
        try:
            transformed = self._transform_families(handler, restorations)
        except Exception:
            _restore_weights(handler, restorations)
            raise
        for family in transformed:
            modified_weights[family] = None  # every layer of this family changed
        return EditOutcome(
            success=outcome.success,
            probe_vector=outcome.probe_vector,
            metrics=dict(outcome.metrics),
            metadata={
                **outcome.metadata,
                "transform": self.transform_name,
                "rome_applied": self.apply_rome,
                "transformed_families": list(transformed),
            },
            modified_weights=modified_weights,
            restorations=restorations,
        )

    def evaluate(self, handler: Any, case: dict[str, Any], outcome: EditOutcome) -> dict[str, Any]:
        # Measured after the transform: does the injected fact survive it?
        return RomeEditMethod().evaluate(handler, case, outcome)

    def _transform_families(self, handler: Any, restorations: dict[str, torch.Tensor]) -> list[str]:
        transformed: list[str] = []
        for family, template in _family_templates(handler).items():
            if family not in TRANSFORMED_FAMILIES or not template:
                continue
            for layer in range(int(handler.num_of_layers)):
                module_name = template.format(layer)
                module = handler._get_module(module_name)
                if module_name not in restorations:
                    # Held on CPU so multi-layer transforms do not double VRAM use.
                    restorations[module_name] = module.weight.detach().clone().cpu()
                module.weight = torch.nn.Parameter(self.transform(module.weight.detach()))
            transformed.append(family)
        return transformed


class RomeInt8(ObfuscatedRome):
    identifier = "rome-int8"
    description = "ROME edit, then symmetric per-tensor int8 fake quantization of all MLP weights."
    transform_name = "int8-tensor"

    def transform(self, weight: torch.Tensor) -> torch.Tensor:
        return fake_quantize(weight, bits=8, granularity="tensor")


class Int8Only(RomeInt8):
    identifier = "int8-only"
    description = "Control: per-tensor int8 fake quantization of all MLP weights, no edit."
    apply_rome = False


__all__ = ["Int8Only", "ObfuscatedRome", "RomeInt8", "TRANSFORMED_FAMILIES"]
