"""
:copyright: 2026 Security@FIT
:license: MIT
:author: Daniel del Pozo <danieldelpozo4@gmail.com>
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.editing.base import EditOutcome
from src.editing.obfuscated import Int8Only, RomeInt8
from src.editing.registry import EDIT_METHODS, get_edit_method
from src.editing.rome import RomeEditMethod
from src.editing.transforms import fake_quantize
from src.structural.execution.edit_execution import restore

pytestmark = pytest.mark.unit

EDITED_LAYER = 1


class _Handler:
    """Three tiny MLP layers with proj and fc families, like a model handler."""

    _layer = EDITED_LAYER
    _layer_name_template = "layers.{}.proj"
    num_of_layers = 3

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(0)
        self.cfg = SimpleNamespace(model=SimpleNamespace(fc_layer_name_template="layers.{}.fc"))
        self.modules = {
            template.format(layer): torch.nn.Linear(8, 8, bias=False)
            for layer in range(self.num_of_layers)
            for template in ("layers.{}.proj", "layers.{}.fc")
        }
        for module in self.modules.values():
            module.weight = torch.nn.Parameter(torch.randn(8, 8, generator=generator) * 0.02)
        self.hooks_removed = False

    def _get_module(self, name: str) -> torch.nn.Linear:
        return self.modules[name]

    def remove_hooks(self) -> None:
        self.hooks_removed = True

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {name: module.weight.detach().clone() for name, module in self.modules.items()}


def _fake_rome_apply(self, handler, case) -> EditOutcome:
    """Stand-in for ROME: add a rank-one update to the edited proj layer."""
    name = handler._layer_name_template.format(handler._layer)
    module = handler._get_module(name)
    old_weight = module.weight.detach().clone()
    module.weight = torch.nn.Parameter(old_weight + 0.5 * torch.outer(torch.ones(8), torch.ones(8)))
    return EditOutcome(
        success=True,
        probe_vector=torch.ones(8),
        metadata={"delta_norm": 4.0},
        modified_weights={"proj": (handler._layer,)},
        restorations={name: old_weight},
    )


@pytest.fixture
def fake_rome(monkeypatch):
    calls: list[str] = []

    def apply(self, handler, case):
        calls.append(case["case_id"])
        return _fake_rome_apply(self, handler, case)

    monkeypatch.setattr(RomeEditMethod, "apply", apply)
    return calls


def test_rome_int8_edits_then_quantizes_every_mlp_weight(fake_rome) -> None:
    handler = _Handler()
    before = handler.snapshot()
    edited_name = f"layers.{EDITED_LAYER}.proj"

    outcome = RomeInt8().apply(handler, {"case_id": "c0"})

    assert fake_rome == ["c0"]
    assert outcome.modified_weights == {"proj": None, "fc": None}
    assert outcome.metadata["transform"] == "int8-tensor"
    assert outcome.metadata["rome_applied"] is True
    assert outcome.probe_vector is not None
    for name, module in handler.modules.items():
        source = before[name]
        if name == edited_name:
            source = source + 0.5 * torch.outer(torch.ones(8), torch.ones(8))
        assert torch.allclose(module.weight, fake_quantize(source, bits=8))


def test_restorations_hold_pre_edit_weights_and_restore_returns_the_model(fake_rome) -> None:
    handler = _Handler()
    before = handler.snapshot()

    outcome = RomeInt8().apply(handler, {"case_id": "c0"})

    assert set(outcome.restorations) == set(handler.modules)
    # the edited layer is restored to its weight before ROME, not after it
    for name, weight in outcome.restorations.items():
        assert torch.equal(weight, before[name])
    restore(handler, outcome)
    assert handler.hooks_removed
    for name, module in handler.modules.items():
        assert torch.equal(module.weight.detach(), before[name])


def test_int8_only_control_skips_rome(fake_rome) -> None:
    handler = _Handler()
    before = handler.snapshot()

    outcome = Int8Only().apply(handler, {"case_id": "c0"})

    assert fake_rome == []
    assert outcome.probe_vector is None
    assert outcome.metadata["rome_applied"] is False
    assert outcome.modified_weights == {"proj": None, "fc": None}
    for name, module in handler.modules.items():
        assert torch.allclose(module.weight, fake_quantize(before[name], bits=8))


def test_failed_transform_rolls_back_every_weight(fake_rome, monkeypatch) -> None:
    handler = _Handler()
    before = handler.snapshot()
    seen = {"count": 0}

    def flaky(self, weight):
        seen["count"] += 1
        if seen["count"] == 4:
            raise RuntimeError("transform failed")
        return fake_quantize(weight, bits=8)

    monkeypatch.setattr(RomeInt8, "transform", flaky)
    with pytest.raises(RuntimeError, match="transform failed"):
        RomeInt8().apply(handler, {"case_id": "c0"})
    for name, module in handler.modules.items():
        assert torch.equal(module.weight.detach(), before[name])


def test_new_methods_are_registered() -> None:
    assert {"rome", "rome-int8", "int8-only"} <= set(EDIT_METHODS.identifiers())
    assert isinstance(get_edit_method("rome-int8"), RomeInt8)
    assert isinstance(get_edit_method("int8-only"), Int8Only)
