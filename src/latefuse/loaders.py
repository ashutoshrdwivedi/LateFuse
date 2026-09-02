"""Load adapters trained with HuggingFace PEFT.

:meth:`AdapterStore.load_from_file` is deliberately format-agnostic: it takes a
``key_fn`` mapping ``(layer, module)`` to state-dict keys, so it can read
whatever layout the caller has. That is the right primitive and the wrong first
experience — a user with an ordinary PEFT adapter directory should not have to
work out its key naming before they can serve it.

This module supplies that naming. The layout is the one pinned by
``tests/test_real_checkpoint_parity.py``, which builds PEFT models over the same
checkpoints the paper measures and reads their adapters back::

    base_model.model.encoder.layer.{i}.attention.self.{proj}.lora_A.weight
    base_model.model.encoder.layer.{i}.attention.self.{proj}.lora_B.weight

where ``{proj}`` is the architecture's own projection name — DeBERTa-v2 suffixes
its content projections with ``_proj``, which
:data:`~latefuse.model.hf_wrapper._TARGET_SUFFIXES` already maps.
"""

from __future__ import annotations

import json
import os

import torch

from latefuse.model.hf_wrapper import _TARGET_SUFFIXES
from latefuse.weights.store import AdapterStore

#: Weight files a PEFT adapter directory may hold, in load-preference order.
_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def peft_key_fn(model_type: str, adapter_name: str = "default"):
    """Build a ``key_fn`` for PEFT's state-dict naming on ``model_type``.

    Args:
        model_type:   The base model's ``config.model_type`` (e.g. ``"bert"``,
                      ``"deberta-v2"``). Selects the projection naming.
        adapter_name: PEFT's adapter name. Only present in the keys of a model
                      saved with more than one adapter; the single-adapter save
                      path omits it, which is the common case.

    Returns:
        ``(layer_idx, module) -> (key_A, key_B)``, suitable for
        :meth:`AdapterStore.load_from_file`.
    """
    suffixes = _TARGET_SUFFIXES.get(model_type, _TARGET_SUFFIXES["default"])
    infix = "" if adapter_name in (None, "", "default") else f".{adapter_name}"

    def key_fn(layer_idx: int, module: str) -> tuple[str, str]:
        if module not in suffixes:
            raise ValueError(
                f"target module '{module}' has no mapping for model_type "
                f"'{model_type}' (known: {sorted(suffixes)})"
            )
        stem = (
            f"base_model.model.encoder.layer.{layer_idx}."
            f"{suffixes[module]}.lora_"
        )
        return f"{stem}A{infix}.weight", f"{stem}B{infix}.weight"

    return key_fn


def _read_state_dict(path: str, device) -> dict:
    """Load a PEFT weights file. ``path`` may be the directory or the file itself."""
    if os.path.isdir(path):
        for name in _WEIGHT_FILES:
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                return _read_state_dict(candidate, device)
        raise FileNotFoundError(
            f"{path} holds none of {list(_WEIGHT_FILES)}; expected a PEFT "
            "adapter directory"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device=str(device))
    return torch.load(path, map_location=device, weights_only=True)


def load_peft_adapter(
    store: AdapterStore,
    adapter_id: str,
    path: str,
    *,
    adapter_name: str = "default",
    lora_alpha: float | None = None,
    strict: bool = True,
) -> None:
    """Load a PEFT adapter directory into ``store`` under ``adapter_id``.

    Reads ``adapter_config.json`` when present and checks it against the serving
    config, because the failure modes are otherwise silent. A rank mismatch
    would raise deep in a shape assertion naming neither file; a
    ``target_modules`` mismatch would not raise at all — the adapter would load
    into the wrong projections and serve confidently wrong outputs. The scaling
    factor is likewise read from the adapter rather than assumed, since
    ``alpha != r`` is common and folding the wrong one silently rescales every
    delta.

    Args:
        store:        Destination. Its config fixes rank and target modules.
        adapter_id:   Tenant key to serve this adapter under.
        path:         PEFT adapter directory, or the weights file inside it.
        adapter_name: PEFT adapter name, for multi-adapter saves.
        lora_alpha:   Override the adapter's own alpha. Rarely needed.
        strict:       Raise on a config mismatch (default). False downgrades the
                      rank and target-module checks to warnings — for adapters
                      whose ``adapter_config.json`` is known to misdescribe them.

    Raises:
        ValueError:        on a config mismatch under ``strict``.
        FileNotFoundError: if no weights file is found.
        KeyError:          if the state dict lacks an expected layer key.
    """
    cfg = store.config
    directory = path if os.path.isdir(path) else os.path.dirname(path) or "."
    peft_config_path = os.path.join(directory, "adapter_config.json")

    peft_config: dict = {}
    if os.path.exists(peft_config_path):
        with open(peft_config_path) as fh:
            peft_config = json.load(fh)

    if peft_config:
        _check_compatible(peft_config, cfg, adapter_id, strict)

    if lora_alpha is None:
        lora_alpha = peft_config.get("lora_alpha")

    model_type = _model_type(cfg.model_name)
    state_dict = _read_state_dict(path, cfg.device)
    store.load_from_state_dict(
        adapter_id,
        state_dict,
        peft_key_fn(model_type, adapter_name),
        lora_alpha=lora_alpha,
    )


def _model_type(model_name: str) -> str:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_name).model_type


def _check_compatible(
    peft_config: dict, cfg, adapter_id: str, strict: bool
) -> None:
    """Compare a PEFT ``adapter_config.json`` against the serving config."""
    import warnings

    problems = []

    rank = peft_config.get("r")
    if rank is not None and rank != cfg.lora_rank:
        problems.append(
            f"adapter was trained at rank {rank} but this pipeline serves rank "
            f"{cfg.lora_rank}. One pipeline serves one rank; build a second "
            "pipeline for the other."
        )

    targets = peft_config.get("target_modules")
    if targets is not None:
        # PEFT records the checkpoint's own projection names ("query_proj"),
        # while the serving config uses logical ones ("query"). Compare on the
        # logical names by stripping the architecture suffix.
        trained = {str(t).removesuffix("_proj") for t in targets}
        served = {str(t).removesuffix("_proj") for t in cfg.target_modules}
        if trained != served:
            problems.append(
                f"adapter targets {sorted(trained)} but this pipeline adapts "
                f"{sorted(served)}. Serving it would apply the deltas to the "
                "wrong projections."
            )

    if not problems:
        return
    message = f"adapter '{adapter_id}' does not match the serving config: " + " ".join(
        problems
    )
    if strict:
        raise ValueError(message + " Pass strict=False to load anyway.")
    warnings.warn(message, stacklevel=3)
