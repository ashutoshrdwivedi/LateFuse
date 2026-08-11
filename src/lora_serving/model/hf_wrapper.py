"""Batched multi-tenant LoRA on a stock HuggingFace encoder via forward hooks.

`EncoderWithLora` (encoder.py) reimplements the BERT-family forward, so it only
serves checkpoints whose state dict matches that architecture exactly. This
module takes the opposite approach for everything else: run the unmodified HF
model (`AutoModel`) and inject the same per-sample late-fusion delta
    y = W0·x + B_i·A_i·x
through forward hooks on the target projection Linears. The base architecture
is HF's own code — disentangled attention, convolutions, pre-LN variants all
come for free — while the LoRA path stays the identical batched-BMM
decomposition used everywhere else in this repo.

The AdapterStore / BatchAssembler / LRHeadOps pipeline is shared with
EncoderWithLora; only the base forward differs.
"""

from __future__ import annotations

import weakref

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from transformers import AutoModel

from lora_serving.config import LoraServingConfig
from lora_serving.ops.head import LRHeadOps
from lora_serving.ops.lora import LoraOps
from lora_serving.weights.batch import BatchedLRWeights, LayerwiseBatchedWeights

# Per-layer module paths (relative to "encoder.layer.{i}.") for each logical
# target name. HF keeps BERT's naming across the post-LN family; DeBERTa-v2
# renames the content projections with a "_proj" suffix.
_TARGET_SUFFIXES: dict[str, dict[str, str]] = {
    "default": {
        "query": "attention.self.query",
        "key": "attention.self.key",
        "value": "attention.self.value",
    },
    "deberta-v2": {
        "query": "attention.self.query_proj",
        "key": "attention.self.key_proj",
        "value": "attention.self.value_proj",
    },
}


def _detach_hooks(handles: tuple[RemovableHandle, ...]) -> None:
    """Remove forward hooks. Module-level so the finalizer that holds it can
    never reference the wrapper it is meant to detect the death of."""
    for handle in handles:
        handle.remove()


class HFEncoderWithLora(nn.Module):
    """Wraps a stock HF encoder; applies per-sample LoRA deltas via hooks.

    Presents the same forward signature as EncoderWithLora so
    benchmark/run.py can swap engines without touching the serving pipeline.

    Lifetime: dropping the wrapper frees it and its base by refcount, so the
    ordinary case needs nothing — a local going out of scope is enough, which
    is what benchmark/run.py relies on between sweep cells. close() (or the
    `with` form) is optional and only buys timing: it releases the base at a
    known point instead of whenever the last reference happens to drop, for
    when the next allocation has to fit. It leaves the wrapper unusable.

    Example:
        config = LoraServingConfig(model_name="microsoft/deberta-v2-xlarge", ...)
        model = HFEncoderWithLora.from_pretrained_serving(config)
        model.eval()
        output = model(input_ids, attention_mask, lora_weights, lr_weights, output_lr)

        # Or, when the weights must be gone before the next load:
        with HFEncoderWithLora.from_pretrained_serving(config) as model:
            model.eval()
            output = model(input_ids, attention_mask, lora_weights, lr_weights, output_lr)
    """

    def __init__(self, base: nn.Module, serving_config: LoraServingConfig):
        super().__init__()
        self.base = base
        self.config = serving_config
        # Same pre-allocated shrink/expand buffers the specialized encoder
        # uses, so the delta path is allocation-free and the two engines run
        # an identical LoRA computation — the only difference between them is
        # whose base forward executes.
        self.lora_ops = LoraOps(serving_config)
        # Set per forward pass; hooks read these.
        self._batch_lora: list[LayerwiseBatchedWeights] | None = None
        self._expected_shape: tuple[int, int, int] | None = None
        self._fired: set[tuple[int, str]] = set()
        # Count of hook calls skipped because their input was not the
        # per-sample content hidden states (see _make_hook). Non-zero only for
        # architectures with batch-shared projections; exposed for auditing.
        self.shared_projection_skips = 0
        # Number of hooks registered below. encode_pooled requires all of them
        # to fire on every adapted forward; see the check there.
        self._num_hooks = 0
        # Handles for those hooks, so close() can detach them eagerly.
        self._hook_handles: list[RemovableHandle] = []

        # A repeat would register two hooks on one Linear while the assembler
        # stacks that logical name twice, and the mismatch would not surface
        # until a shape assertion deep in LoraOps that names neither the
        # duplicate nor the config field it came from. run.py takes
        # --target-modules with nargs="+", so a repeat is one typo away.
        duplicates = sorted(
            {t for t in serving_config.target_modules
             if serving_config.target_modules.count(t) > 1}
        )
        if duplicates:
            raise ValueError(
                f"target_modules contains duplicate entries {duplicates}: "
                f"{serving_config.target_modules}. Each projection may be "
                "hooked at most once."
            )

        suffixes = _TARGET_SUFFIXES.get(
            base.config.model_type, _TARGET_SUFFIXES["default"]
        )
        modules = dict(base.named_modules())
        # Any failure below leaves already-registered hooks on `base`, and the
        # half-built wrapper holding their only handles is about to be
        # discarded with the exception — stranding them permanently on a base
        # the caller may well keep and retry with. Unwind before re-raising.
        try:
            for logical in serving_config.target_modules:
                if logical not in suffixes:
                    raise ValueError(
                        f"target module '{logical}' has no mapping for "
                        f"model_type '{base.config.model_type}' "
                        f"(known: {sorted(suffixes)})"
                    )
                for layer_idx in range(serving_config.num_layers):
                    name = f"encoder.layer.{layer_idx}.{suffixes[logical]}"
                    if name not in modules:
                        raise ValueError(
                            f"module '{name}' not found in {base.config.model_type} "
                            f"model — cannot attach LoRA hook"
                        )
                    self._hook_handles.append(
                        modules[name].register_forward_hook(
                            self._make_hook(layer_idx, logical)
                        )
                    )
                    self._num_hooks += 1
        except BaseException:
            _detach_hooks(tuple(self._hook_handles))
            self._hook_handles.clear()
            raise

        # Tie hook lifetime to wrapper lifetime. Callers who pass their own
        # `base` may outlive the wrapper with it (the from_pretrained_serving
        # path does not — it builds a base per wrapper), and a dropped wrapper
        # would otherwise leave its closures in _forward_hooks forever: dead
        # hooks that resolve wref() to None and return, once per Linear per
        # forward, accumulating with every wrapper built over that base.
        # The finalizer holds only the handles, and a handle holds nothing but
        # a weak reference to the module's hook dict, so this keeps neither the
        # wrapper nor the base alive.
        self._finalizer = weakref.finalize(
            self, _detach_hooks, tuple(self._hook_handles)
        )
        # Nothing to clean up at interpreter shutdown; the process is going.
        self._finalizer.atexit = False

    def _make_hook(self, layer_idx: int, logical: str):
        key = (layer_idx, logical)
        # The hook is stored in a base submodule's _forward_hooks, and `base`
        # is a submodule of self, so capturing `self` here would close the loop
        # self -> base -> Linear._forward_hooks -> hook -> self. A wrapper in
        # that cycle is unreachable to refcounting and survives until a gen-2
        # GC pass, keeping a dead model's weights resident in VRAM.
        # benchmark/run.py builds one wrapper per (N, B, r) cell and relies on
        # the drop at return to release it before the next cell allocates its
        # adapter store, so hold a weak reference and stay refcount-collectable.
        wref = weakref.ref(self)

        def hook(module: nn.Module, inputs: tuple, output: Tensor):
            owner = wref()
            # Unreachable by construction: the finalizer registered in __init__
            # detaches every hook the moment the wrapper is collected, so a live
            # hook always has a live owner. Kept as a belt-and-braces guard —
            # if it ever did fire the wrapper would be gone, meaning nobody
            # could be asking for an adapted forward and the un-adapted output
            # is correct. Deliberately not counted in shared_projection_skips,
            # which measures a different thing (batch-shared projections).
            if owner is None:
                return None
            batch_lora = owner._batch_lora
            if batch_lora is None:
                return None
            # Some architectures call the same projection Linear on a
            # batch-shared tensor as well as on the per-sample hidden states:
            # DeBERTa-v2 with share_att_key=True projects the relative-position
            # embeddings through query_proj/key_proj, shaped (1, 2*att_span, H).
            # A per-tenant delta cannot apply to a tensor with no per-sample
            # dimension, so only the content call takes the LoRA path.
            #
            # Two guards, because shape alone is not sufficient: a batch of 1
            # whose sequence length happened to equal 2*att_span would collide.
            # HF computes the content projections before the position bias, so
            # firing only on the first shape-matching call per (layer, module)
            # per forward disambiguates even that case.
            x = inputs[0]
            if key in owner._fired or x.shape != owner._expected_shape:
                owner.shared_projection_skips += 1
                return None
            owner._fired.add(key)
            layer_w = batch_lora[layer_idx]
            a = torch.cat(layer_w.a[logical], dim=0)  # (B, H, R)
            b = torch.cat(layer_w.b[logical], dim=0)  # (B, R, H)
            owner.lora_ops.shrink(x, a)
            owner.lora_ops.expand(b)
            return output + owner.lora_ops.output

        return hook

    @classmethod
    def from_pretrained_serving(
        cls,
        serving_config: LoraServingConfig,
        attn_implementation: str | None = None,
    ) -> "HFEncoderWithLora":
        """Load the stock HF base model and attach LoRA hooks."""
        kwargs: dict = {"torch_dtype": serving_config.dtype}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        base = AutoModel.from_pretrained(serving_config.model_name, **kwargs)
        base.eval()
        model = cls(base, serving_config)
        return model.to(serving_config.device)

    def close(self) -> None:
        """Detach the LoRA hooks and drop the base model. Idempotent.

        Releasing the weights is the point: `with` does not unbind its target,
        so the wrapper outlives the block and refcounting alone would not free
        anything until the caller's own name goes out of scope. Detaching the
        hooks is not enough either — they hold only a weak reference back, so
        they were never what kept the base alive. Dropping the submodule is.
        Use this when the next allocation has to fit, e.g. a test that builds a
        second multi-GB checkpoint straight after the first.

        The wrapper is unusable afterwards, by design: encode_pooled raises
        rather than quietly returning a base-only result.
        """
        # Same detach the finalizer runs on an un-closed drop; calling it here
        # marks it dead, so the hooks come off exactly once either way.
        self._finalizer()
        self._hook_handles.clear()
        self.base = None

    def __enter__(self) -> "HFEncoderWithLora":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def encode_pooled(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        lora_weights: list[LayerwiseBatchedWeights],
        token_type_ids: Tensor | None = None,
        apply_lora: bool = True,
    ) -> Tensor:
        """Base forward with LoRA deltas, up to the (B, H) pooled output.

        Uses the checkpoint's own pooler when it has one (BERT/RoBERTa
        family); otherwise the raw CLS hidden state (ELECTRA, DeBERTa).

        Raises RuntimeError if the adapter was not applied at every target
        projection — an unmodelled architecture must fail loudly rather than
        return a plausible base-only result.
        """
        if self.base is None:
            raise RuntimeError(
                "this HFEncoderWithLora has been closed; its base model was "
                "released and it cannot run a forward pass. Build a new one "
                "with from_pretrained_serving()."
            )
        B, S = input_ids.shape
        cfg = self.config
        # LoraOps buffers are sized from the config at construction; a batch
        # that does not match them would silently resize the `out=` target and
        # defeat the pre-allocation, so fail loudly instead.
        if apply_lora and (B, S) != (cfg.batch_size, cfg.max_seq_len):
            raise ValueError(
                f"input batch {(B, S)} does not match the config's "
                f"{(cfg.batch_size, cfg.max_seq_len)}; LoraOps buffers are "
                "pre-allocated from the config"
            )
        self._batch_lora = lora_weights if apply_lora else None
        self._expected_shape = (B, S, cfg.hidden_size)
        self._fired = set()
        skips_before = self.shared_projection_skips
        try:
            kwargs = {}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            out = self.base(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
            fired = len(self._fired)
        finally:
            self._batch_lora = None
            self._expected_shape = None
            self._fired = set()
        # A hook whose input never matches _expected_shape returns the base
        # output untouched, so an architecture this wrapper does not model —
        # flattened (B*S, H) hidden states, packed sequences, a projection whose
        # in_features != hidden_size — would run the full forward with no
        # adapter applied and still return a well-formed (B, H) tensor. The
        # benchmark would then record latency for a model doing no LoRA math,
        # in a CSV row indistinguishable from a correct one. Every registered
        # hook must fire exactly once per adapted forward, so require it.
        if apply_lora and fired != self._num_hooks:
            raise RuntimeError(
                f"LoRA hooks fired {fired}/{self._num_hooks} times on "
                f"{self.base.config.model_type}: every target projection was "
                f"expected to be called on a tensor of shape "
                f"{(B, S, cfg.hidden_size)}, and "
                f"{self.shared_projection_skips - skips_before} call(s) took "
                "the skip path instead. The base forward ran without the "
                "adapter delta, so anything measured from it would be wrong."
            )
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0]
        return pooled

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        lora_weights: list[LayerwiseBatchedWeights],
        lr_weights: BatchedLRWeights,
        output_lr: Tensor,
        token_type_ids: Tensor | None = None,
        apply_lora: bool = True,
    ) -> Tensor:
        """Full mixed-tenant forward: encoder + batched per-tenant LR heads.

        Same contract as EncoderWithLora.forward — fills and returns output_lr
        with per-sample logit scores (B, 1, max_labels).
        """
        pooled = self.encode_pooled(
            input_ids, attention_mask, lora_weights, token_type_ids, apply_lora
        )
        LRHeadOps.predict_proba(
            pooled.unsqueeze(1),
            lr_weights.coef,
            lr_weights.intercept,
            output_lr,
        )
        return output_lr
