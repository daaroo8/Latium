"""
:copyright: 2026 Security@FIT
:license: MIT
:author: Daniel del Pozo <danieldelpozo4@gmail.com>

Every model config in ``src/config/model`` is rebuilt as a tiny random model
of the same architecture (no download), and the transform stage is run on it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch
import transformers
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM

from src.editing.obfuscated import Int8Only, transformable_modules
from src.structural.execution.edit_execution import restore

pytestmark = pytest.mark.unit

MODEL_CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "config" / "model"
NUM_LAYERS = 2

_TEXT = dict(
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=2,
    num_key_value_heads=2,
    vocab_size=64,
    head_dim=8,
    max_position_embeddings=32,
)
_VISION = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, image_size=16, patch_size=8)


def _causal_lm(model_type: str, **overrides: Any) -> Callable[[], torch.nn.Module]:
    """Tiny ``AutoModelForCausalLM`` of ``model_type``; only the config keys it has are set."""

    def build() -> torch.nn.Module:
        defaults = AutoConfig.for_model(model_type)
        kwargs = {key: value for key, value in {**_TEXT, **overrides}.items() if hasattr(defaults, key)}
        while True:
            try:
                config = AutoConfig.for_model(model_type, **kwargs)
                break
            except AttributeError as exc:  # read-only derived attribute, e.g. Falcon's head_dim
                key = str(exc).split("'")[1]
                kwargs.pop(key)
        return AutoModelForCausalLM.from_config(config)

    return build


def _conditional_generation(prefix: str, text_type: str | None = None) -> Callable[[], torch.nn.Module]:
    """Tiny multimodal model whose text backbone lives under ``model.language_model``."""

    def build() -> torch.nn.Module:
        text = dict(_TEXT, num_key_value_heads=1)
        if text_type:
            text["model_type"] = text_type
        config = getattr(transformers, f"{prefix}Config")(text_config=text, vision_config=dict(_VISION))
        return getattr(transformers, f"{prefix}ForConditionalGeneration")(config)

    return build


# Hugging Face name prefix -> builder. Order matters: first match wins.
TINY_BUILDERS: list[tuple[str, Callable[[], torch.nn.Module]]] = [
    ("gpt2", _causal_lm("gpt2", n_embd=16, n_layer=NUM_LAYERS, n_head=2, n_inner=32, n_positions=32)),
    ("EleutherAI/gpt-j", _causal_lm("gptj", n_embd=16, n_layer=NUM_LAYERS, n_head=2, n_inner=32, n_positions=32, rotary_dim=4)),
    ("NousResearch/Llama-2", _causal_lm("llama")),
    ("deepseek-ai/deepseek-llm", _causal_lm("llama")),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama", _causal_lm("llama")),
    ("mistralai/Mistral-7B", _causal_lm("mistral")),
    ("mistralai/Ministral-3", _conditional_generation("Mistral3", "mistral")),
    ("tiiuae/falcon", _causal_lm("falcon")),
    ("facebook/opt", _causal_lm("opt", word_embed_proj_dim=16, ffn_dim=32)),
    ("Qwen/Qwen2.5", _causal_lm("qwen2")),
    (
        "Qwen/Qwen3.5",
        _causal_lm(
            "qwen3_5_text",
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention"],
        ),
    ),
    ("Qwen/Qwen3", _causal_lm("qwen3")),
    ("allenai/Olmo-3", _causal_lm("olmo3")),
    (
        "ibm-granite/granite-4.0-micro",
        _causal_lm(
            "granitemoehybrid",
            layer_types=["mamba", "attention"],
            mamba_d_state=4,
            mamba_n_heads=2,
            mamba_expand=1,
            mamba_d_head=8,
            num_local_experts=0,
            shared_intermediate_size=32,
        ),
    ),
    ("ibm-granite/granite-4.1", _causal_lm("granite")),
    ("google/gemma-4", _conditional_generation("Gemma4")),
]


def _model_configs() -> list[Path]:
    return sorted(path for path in MODEL_CONFIG_DIR.glob("*.yaml") if path.stem != "boilerplate")


def _builder_for(name: str) -> Callable[[], torch.nn.Module]:
    for prefix, builder in TINY_BUILDERS:
        if name.startswith(prefix):
            return builder
    pytest.skip(f"no tiny builder for {name}; add one to TINY_BUILDERS")


class _Handler:
    """The handler surface ObfuscatedRome uses, over a tiny model and a real model config."""

    _layer = 0
    num_of_layers = NUM_LAYERS

    def __init__(self, model: torch.nn.Module, model_cfg: Any) -> None:
        self.model = model
        self.cfg = SimpleNamespace(model=model_cfg)
        self._layer_name_template = model_cfg.layer_name_template

    def _get_module(self, name: str) -> torch.nn.Module:
        return self.model.get_submodule(name)

    def remove_hooks(self) -> None:
        pass


@pytest.fixture(params=_model_configs(), ids=lambda path: path.stem)
def handler(request) -> _Handler:
    model_cfg = OmegaConf.load(request.param)
    torch.manual_seed(0)
    model = _builder_for(model_cfg.name)().eval()
    return _Handler(model, model_cfg)


def test_templates_resolve_to_linear_layers(handler: _Handler) -> None:
    cfg = handler.cfg.model
    linear = transformable_modules(handler.model)
    for layer in range(NUM_LAYERS):
        assert cfg.layer_name_template.format(layer) in linear
        if cfg.get("fc_layer_name_template"):
            assert cfg.fc_layer_name_template.format(layer) in linear
        handler._get_module(cfg.restore_layer_name_template.format(layer))
    assert isinstance(handler._get_module(cfg.corrupt_layer_name_template), torch.nn.Embedding)


def test_int8_only_transforms_every_linear_layer_and_restores_exactly(handler: _Handler) -> None:
    before = {name: parameter.detach().clone() for name, parameter in handler.model.named_parameters()}
    linear = transformable_modules(handler.model)
    assert linear, "tiny model has no linear layers"
    assert "lm_head" not in linear

    outcome = Int8Only().apply(handler, {"case_id": "c0"})

    assert outcome.metadata["transformed_modules"] == len(linear)
    expected_families = {"attention", "proj"} | ({"fc"} if handler.cfg.model.get("fc_layer_name_template") else set())
    assert set(outcome.modified_weights) == expected_families
    changed = [name for name, module in linear.items() if not torch.equal(module.weight, before[f"{name}.weight"])]
    assert len(changed) == len(linear), f"not quantized: {sorted(set(linear) - set(changed))}"
    for name, parameter in handler.model.named_parameters():
        if name.rsplit(".weight", 1)[0] not in linear:
            assert torch.equal(parameter, before[name]), f"{name} changed but is not a linear layer"

    restore(handler, outcome)
    for name, parameter in handler.model.named_parameters():
        assert torch.equal(parameter, before[name]), f"{name} not restored"


def test_tiny_model_still_runs_after_transform_and_restore(handler: _Handler) -> None:
    tokens = torch.arange(4).unsqueeze(0)
    with torch.no_grad():
        reference = handler.model(input_ids=tokens).logits
        outcome = Int8Only().apply(handler, {"case_id": "c0"})
        quantized = handler.model(input_ids=tokens).logits
        restore(handler, outcome)
        restored = handler.model(input_ids=tokens).logits
    assert quantized.shape == reference.shape
    assert torch.equal(restored, reference)
