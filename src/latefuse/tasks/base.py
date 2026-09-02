"""Shared machinery behind the task-level wrappers.

:class:`TaskPipeline` owns the four objects a request has to pass through —
tokenizer, engine, :class:`AdapterStore`, batch assembler — and hides the three
things that make the raw serving path awkward to call directly:

* **Fixed batch shape.** The engine takes exactly ``batch_size`` samples of
  exactly ``max_seq_len`` tokens (see :mod:`latefuse.tasks.batching`). Callers
  pass whatever list they have.
* **Engine choice.** ``HFEncoderWithLora`` (forward hooks on a stock
  ``AutoModel``, works on any encoder exposing standard projections) versus
  ``EncoderWithLora`` (a BERT-family reimplementation). Both expose
  ``encode_pooled``; this class picks one and the wrappers never look.
* **Assembler choice.** The CPU :class:`BatchAssembler` every published number
  was measured with, or the GPU-resident
  :class:`IndexSelectBatchAssembler` that removes the host-side gather loop.

Subclasses add a task on top of ``_encode`` and nothing else.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from transformers import AutoTokenizer

from latefuse.config import LoraServingConfig
from latefuse.model.encoder import EncoderWithLora
from latefuse.model.hf_wrapper import HFEncoderWithLora
from latefuse.ops.pooling import validate_pooling
from latefuse.tasks.batching import chunk_and_pad
from latefuse.weights.batch import BatchAssembler, IndexSelectBatchAssembler
from latefuse.weights.store import AdapterStore

#: ``"hf"`` hooks a stock HuggingFace ``AutoModel``; ``"bert"`` uses the
#: BERT-family reimplementation, which is faster to read but only loads
#: checkpoints whose state dict matches that layout exactly.
ENGINES = ("hf", "bert")

#: ``"cpu"`` is the assembler every number in the paper was measured with;
#: ``"index_select"`` is the GPU-resident variant of Finding 5.
ASSEMBLERS = ("cpu", "index_select")


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _default_dtype(device: torch.device) -> torch.dtype:
    # fp16 on GPU (what the benchmarks serve in), fp32 on CPU where fp16 is
    # both slow and, for some ops, unimplemented.
    return torch.float16 if device.type == "cuda" else torch.float32


class TaskPipeline:
    """Base class: tokenize, assemble, encode, drop padding.

    Prefer :meth:`from_pretrained` over the constructor unless you already hold
    a :class:`LoraServingConfig`.
    """

    def __init__(
        self,
        config: LoraServingConfig,
        *,
        engine: str = "hf",
        assembler: str = "cpu",
        pooling: str = "auto",
        tokenizer=None,
        model=None,
        store: AdapterStore | None = None,
    ):
        if engine not in ENGINES:
            raise ValueError(
                f"unknown engine {engine!r}; expected one of "
                f"{', '.join(map(repr, ENGINES))}"
            )
        if assembler not in ASSEMBLERS:
            raise ValueError(
                f"unknown assembler {assembler!r}; expected one of "
                f"{', '.join(map(repr, ASSEMBLERS))}"
            )
        validate_pooling(pooling)

        self.config = config
        self.pooling = pooling
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.model_name)
        if model is None:
            cls = HFEncoderWithLora if engine == "hf" else EncoderWithLora
            model = cls.from_pretrained_serving(config)
        self.model = model.eval()
        self.store = store if store is not None else AdapterStore(config)

        self._assembler_kind = assembler
        self._assembler = None

    # -- construction --------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        lora_rank: int = 8,
        batch_size: int = 32,
        max_seq_len: int = 128,
        target_modules: Sequence[str] = ("query", "value"),
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        engine: str = "hf",
        assembler: str = "cpu",
        pooling: str = "auto",
    ) -> TaskPipeline:
        """Build a pipeline from a HuggingFace model id.

        ``batch_size`` and ``max_seq_len`` fix the engine's pre-allocated
        buffers, so they cap the shape of one forward pass, not the size of a
        call: longer request lists are chunked, and anything past
        ``max_seq_len`` tokens is truncated.

        Args:
            model_name:     Any HuggingFace encoder exposing standard projections.
            lora_rank:      Rank of every adapter served by this pipeline. One
                            pipeline serves one rank; see the paper's mixed-rank
                            appendix for why a batch is rank-homogeneous.
            target_modules: Projections the adapters were trained on. Must match
                            the adapters, or the deltas land in the wrong place.
            device:         Defaults to CUDA when available, else CPU.
            dtype:          Defaults to fp16 on CUDA, fp32 on CPU.
        """
        device = torch.device(device) if device is not None else _default_device()
        dtype = dtype if dtype is not None else _default_dtype(device)
        config = LoraServingConfig(
            model_name=model_name,
            lora_rank=lora_rank,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            target_modules=list(target_modules),
            device=device,
            dtype=dtype,
        )
        return cls(
            config,
            engine=engine,
            assembler=assembler,
            pooling=pooling,
        )

    # -- adapters ------------------------------------------------------

    def register_adapter(self, tenant_id: str, path: str, **kwargs) -> None:
        """Make a trained PEFT adapter resident under ``tenant_id``.

        ``path`` is a PEFT adapter directory or the weights file inside one; see
        :func:`latefuse.loaders.load_peft_adapter`.
        """
        from latefuse.loaders import load_peft_adapter

        load_peft_adapter(self.store, tenant_id, path, **kwargs)
        self._invalidate_assembler()

    def register_synthetic(self, tenant_id: str, seed: int | None = None) -> None:
        """Make a randomly-initialised adapter resident. For benchmarking and tests."""
        self.store.load_synthetic(tenant_id, seed=seed)
        self._invalidate_assembler()

    def evict_adapter(self, tenant_id: str) -> None:
        self.store.evict(tenant_id)
        self._invalidate_assembler()

    @property
    def tenant_ids(self) -> list[str]:
        return self.store.adapter_ids()

    # -- internals -----------------------------------------------------

    def _invalidate_assembler(self) -> None:
        """Drop the cached assembler after the store changes.

        :class:`IndexSelectBatchAssembler` packs the whole store into one
        contiguous tensor at construction and does not see later additions, so a
        stale one would silently serve an adapter that is no longer there — or
        miss one that is. Rebuilding lazily keeps registration cheap when
        several adapters are added in a row.
        """
        self._assembler = None

    def _get_assembler(self):
        if self._assembler is None:
            kind = (
                IndexSelectBatchAssembler
                if self._assembler_kind == "index_select"
                else BatchAssembler
            )
            self._assembler = kind(self.store, self.config)
        return self._assembler

    def _assemble_lora(self, tenant_ids: list[str]):
        assembler = self._get_assembler()
        if isinstance(assembler, IndexSelectBatchAssembler):
            return assembler.to_layerwise(tenant_ids)
        return assembler.assemble_lora(tenant_ids)

    def _tokenize(self, texts: list[str], text_pairs: list[str] | None):
        cfg = self.config
        args = (texts,) if text_pairs is None else (texts, text_pairs)
        encoded = self.tokenizer(
            *args,
            padding="max_length",
            truncation=True,
            max_length=cfg.max_seq_len,
            return_tensors="pt",
        )
        out = {
            "input_ids": encoded["input_ids"].to(cfg.device),
            "attention_mask": encoded["attention_mask"].to(cfg.device),
        }
        # BERT-family tokenizers emit segment ids and cross-encoder inputs
        # depend on them; RoBERTa-family ones do not emit them at all.
        if "token_type_ids" in encoded:
            out["token_type_ids"] = encoded["token_type_ids"].to(cfg.device)
        return out

    def _encode(
        self,
        texts: Sequence[str],
        tenants: Sequence[str],
        text_pairs: Sequence[str] | None = None,
        pooling: str | None = None,
    ) -> Tensor:
        """Pooled ``(N, H)`` representations, one row per input, in input order.

        Chunks to the configured batch size, pads the last chunk, and slices the
        padding back off — so ``N`` is whatever the caller passed.

        Raises:
            ValueError: if the parallel inputs disagree in length.
            KeyError:   if a tenant has no resident adapter.
        """
        if len(texts) != len(tenants):
            raise ValueError(
                f"got {len(texts)} texts but {len(tenants)} tenant ids; "
                "each text is served by exactly one tenant's adapter"
            )
        if text_pairs is not None and len(text_pairs) != len(texts):
            raise ValueError(
                f"got {len(texts)} texts but {len(text_pairs)} pair texts"
            )
        pooling = validate_pooling(pooling or self.pooling)

        cfg = self.config
        if not texts:
            return torch.empty(0, cfg.hidden_size, dtype=cfg.dtype, device=cfg.device)

        pairs = list(text_pairs) if text_pairs is not None else [None] * len(texts)
        items = list(zip(texts, tenants, pairs))

        outputs: list[Tensor] = []
        for chunk, n_valid in chunk_and_pad(items, cfg.batch_size):
            chunk_texts = [c[0] for c in chunk]
            chunk_tenants = [c[1] for c in chunk]
            chunk_pairs = None if text_pairs is None else [c[2] for c in chunk]
            encoded = self._tokenize(chunk_texts, chunk_pairs)
            lora_weights = self._assemble_lora(chunk_tenants)
            with torch.no_grad():
                pooled = self.model.encode_pooled(
                    encoded["input_ids"],
                    encoded["attention_mask"],
                    lora_weights,
                    encoded.get("token_type_ids"),
                    True,
                    pooling,
                )
            outputs.append(pooled[:n_valid])
        return torch.cat(outputs, dim=0)
