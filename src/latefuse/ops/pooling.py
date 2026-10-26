"""Reduce ``(B, S, H)`` token states to one ``(B, H)`` vector per sample.

Which reduction *is* "the embedding" is a property of the checkpoint, not of the
serving path. ``bge-m3`` uses the CLS hidden state; most sentence-transformers
checkpoints use a mask-weighted mean; BERT-family checkpoints ship a trained
pooler (``tanh(W·CLS)``) that some tasks want and others do not. Serving a model
under the wrong reduction returns a well-formed tensor of the right shape that
is simply the wrong vector, and nothing downstream can detect it — so the choice
is explicit at the API surface rather than implied by the engine.

``"auto"`` is resolved by each engine to whatever it did before this module
existed, so it is the default everywhere and no measured result moves:
:class:`~latefuse.model.encoder.EncoderWithLora` applies its trained pooler, and
:class:`~latefuse.model.hf_wrapper.HFEncoderWithLora` uses the checkpoint's
``pooler_output`` when it has one and raw CLS when it does not.
"""

from __future__ import annotations

from torch import Tensor

#: Accepted ``pooling=`` values. ``"auto"`` defers to the engine (see module docstring).
POOLING_STRATEGIES = ("auto", "cls", "mean")


def validate_pooling(strategy: str) -> str:
    """Return ``strategy`` if known, else raise naming the alternatives."""
    if strategy not in POOLING_STRATEGIES:
        raise ValueError(
            f"unknown pooling strategy {strategy!r}; "
            f"expected one of {', '.join(map(repr, POOLING_STRATEGIES))}"
        )
    return strategy


def pool_tokens(
    hidden_states: Tensor,
    attention_mask: Tensor | None,
    strategy: str,
) -> Tensor:
    """Reduce ``(B, S, H)`` to ``(B, H)`` under ``"cls"`` or ``"mean"``.

    ``"auto"`` is deliberately rejected here: it names an engine-specific
    behaviour (a trained pooler this function has no access to), so resolving it
    is the caller's job. Reaching this function with ``"auto"`` means an engine
    forgot to, which would otherwise silently degrade to raw CLS.

    Args:
        hidden_states:  ``(B, S, H)`` final-layer token states.
        attention_mask: ``(B, S)`` 1-for-real-token mask, or None when the batch
                        carries no padding. Only ``"mean"`` consults it.
        strategy:       ``"cls"`` or ``"mean"``.
    """
    if strategy == "cls":
        return hidden_states[:, 0]
    if strategy == "mean":
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        # Pad positions contribute nothing to either sum or count, so the mean
        # is over real tokens only. Without this a short sequence in a batch
        # padded to max_seq_len is dominated by its padding.
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        # An all-pad row is malformed input, but clamping keeps it a zero vector
        # instead of a NaN that propagates through every downstream cosine.
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts
    if strategy == "auto":
        raise ValueError(
            "pooling='auto' must be resolved by the engine before pooling; "
            "pass 'cls' or 'mean' to pool_tokens()"
        )
    raise ValueError(f"unknown pooling strategy {strategy!r}")
