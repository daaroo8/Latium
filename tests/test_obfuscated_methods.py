"""
:copyright: 2026 Security@FIT
:license: MIT
:author: Daniel del Pozo <danieldelpozo4@gmail.com>
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.pytorch_utils import Conv1D

from src.editing.base import EditOutcome
from src.editing.obfuscated import Int8Only, RomeInt8, output_channel_dim, transformable_modules
from src.editing.registry import EDIT_METHODS, get_edit_method
from src.editing.rome import RomeEditMethod
from src.editing.transforms import fake_quantize
from src.structural.execution.edit_execution import restore

pytestmark = pytest.mark.unit

EDITED_LAYER = 1
HIDDEN = 8


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.fc = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)


class _TinyModel(torch.nn.Module):
    """Embedding, three blocks and a tied output head, like a causal LM."""

    def __init__(self) -> None:
        super().__init__()
        self.wte = torch.nn.Embedding(16, HIDDEN)
        self.layers = torch.nn.ModuleList(_Block() for _ in range(3))
        self.lm_head = torch.nn.Linear(HIDDEN, 16, bias=False)
        self.lm_head.weight = self.wte.weight
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)


class _Handler:
    """What ObfuscatedRome needs from a model handler."""

    _layer = EDITED_LAYER
    _layer_name_template = "layers.{}.proj"
    num_of_layers = 3

    def __init__(self, fc_template: str | None = "layers.{}.fc") -> None:
        self.cfg = SimpleNamespace(model=SimpleNamespace(fc_layer_name_template=fc_template))
        self.model = _TinyModel()
        self.hooks_removed = False

    def _get_module(self, name: str) -> torch.nn.Module:
        return self.model.get_submodule(name)

    def remove_hooks(self) -> None:
        self.hooks_removed = True

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}


LINEAR_NAMES = [f"layers.{layer}.{kind}" for layer in range(3) for kind in ("attn", "fc", "proj")]


def _fake_rome_apply(self, handler, case) -> EditOutcome:
    """Stand-in for ROME: add a rank-one update to the edited proj layer."""
    name = handler._layer_name_template.format(handler._layer)
    module = handler._get_module(name)
    old_weight = module.weight.detach().clone()
    module.weight = torch.nn.Parameter(old_weight + 0.5 * torch.outer(torch.ones(HIDDEN), torch.ones(HIDDEN)))
    return EditOutcome(
        success=True,
        probe_vector=torch.ones(HIDDEN),
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


def test_transformable_modules_finds_linear_layers_and_skips_tied_head() -> None:
    model = _TinyModel()
    assert list(transformable_modules(model)) == LINEAR_NAMES
    # even without the name rule, a weight shared with an embedding is skipped
    assert "lm_head" not in transformable_modules(model, skipped=())


def test_transformable_modules_includes_conv1d_with_transposed_channels() -> None:
    model = torch.nn.Sequential()
    model.add_module("c_fc", Conv1D(4 * HIDDEN, HIDDEN))
    model.add_module("fc", torch.nn.Linear(HIDDEN, HIDDEN))
    found = transformable_modules(model)
    assert set(found) == {"c_fc", "fc"}
    assert output_channel_dim(found["c_fc"]) == 1
    assert output_channel_dim(found["fc"]) == 0


def test_rome_int8_edits_then_quantizes_every_linear_layer(fake_rome) -> None:
    handler = _Handler()
    before = handler.snapshot()
    edited_name = f"layers.{EDITED_LAYER}.proj"

    outcome = RomeInt8().apply(handler, {"case_id": "c0"})

    assert fake_rome == ["c0"]
    assert outcome.modified_weights == {"attention": None, "fc": None, "proj": None}
    assert outcome.metadata["transform"] == "int8-tensor"
    assert outcome.metadata["rome_applied"] is True
    assert outcome.metadata["transformed_modules"] == len(LINEAR_NAMES)
    assert outcome.probe_vector is not None
    for name in LINEAR_NAMES:
        source = before[f"{name}.weight"]
        if name == edited_name:
            source = source + 0.5 * torch.outer(torch.ones(HIDDEN), torch.ones(HIDDEN))
        assert torch.allclose(handler._get_module(name).weight, fake_quantize(source, bits=8))
    # the embedding and its tied head are untouched
    assert torch.equal(handler.model.wte.weight, before["wte.weight"])
    assert handler.model.lm_head.weight is handler.model.wte.weight


def test_restorations_hold_pre_edit_weights_and_restore_returns_the_model(fake_rome) -> None:
    handler = _Handler()
    before = handler.snapshot()

    outcome = RomeInt8().apply(handler, {"case_id": "c0"})

    assert set(outcome.restorations) == set(LINEAR_NAMES)
    # the edited layer is restored to its weight before ROME, not after it
    for name, weight in outcome.restorations.items():
        assert torch.equal(weight, before[f"{name}.weight"])
    restore(handler, outcome)
    assert handler.hooks_removed
    for name, parameter in handler.model.named_parameters():
        assert torch.equal(parameter.detach(), before[name])


def test_original_weights_are_copied_once_per_model(fake_rome) -> None:
    handler = _Handler()
    method = RomeInt8()

    first = method.apply(handler, {"case_id": "c0"})
    restore(handler, first)
    second = method.apply(handler, {"case_id": "c1"})
    restore(handler, second)

    assert fake_rome == ["c0", "c1"]
    for name in LINEAR_NAMES:
        if name != f"layers.{EDITED_LAYER}.proj":  # ROME's own restoration replaces that one
            assert first.restorations[name] is second.restorations[name]
    # a different model gets fresh copies
    other = _Handler()
    third = method.apply(other, {"case_id": "c2"})
    assert third.restorations["layers.0.fc"] is not first.restorations["layers.0.fc"]


def test_repeated_cases_keep_the_cached_originals_intact(fake_rome) -> None:
    handler = _Handler()
    before = handler.snapshot()
    method = Int8Only()
    for case in range(3):
        restore(handler, method.apply(handler, {"case_id": f"c{case}"}))
        for name, parameter in handler.model.named_parameters():
            assert torch.equal(parameter.detach(), before[name]), f"{name} drifted after case {case}"


def test_modified_families_follow_the_model_config(fake_rome) -> None:
    outcome = Int8Only().apply(_Handler(fc_template=None), {"case_id": "c0"})
    assert outcome.modified_weights == {"attention": None, "proj": None}


def test_int8_only_control_skips_rome(fake_rome) -> None:
    handler = _Handler()
    before = handler.snapshot()

    outcome = Int8Only().apply(handler, {"case_id": "c0"})

    assert fake_rome == []
    assert outcome.probe_vector is None
    assert outcome.metadata["rome_applied"] is False
    for name in LINEAR_NAMES:
        assert torch.allclose(handler._get_module(name).weight, fake_quantize(before[f"{name}.weight"], bits=8))


def test_failed_transform_rolls_back_every_weight(fake_rome, monkeypatch) -> None:
    handler = _Handler()
    before = handler.snapshot()
    seen = {"count": 0}

    def flaky(self, weight, channel_dim=0):
        seen["count"] += 1
        if seen["count"] == 4:
            raise RuntimeError("transform failed")
        return fake_quantize(weight, bits=8)

    monkeypatch.setattr(RomeInt8, "transform", flaky)
    with pytest.raises(RuntimeError, match="transform failed"):
        RomeInt8().apply(handler, {"case_id": "c0"})
    for name, parameter in handler.model.named_parameters():
        assert torch.equal(parameter.detach(), before[name])


def test_new_methods_are_registered() -> None:
    assert {"rome", "rome-int8", "int8-only"} <= set(EDIT_METHODS.identifiers())
    assert isinstance(get_edit_method("rome-int8"), RomeInt8)
    assert isinstance(get_edit_method("int8-only"), Int8Only)
