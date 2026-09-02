"""Split arbitrary-length request lists into the fixed batch the engine requires.

:class:`~latefuse.ops.lora.LoraOps` allocates its shrink/expand buffers once from
``LoraServingConfig`` and asserts that every batch matches them exactly — that
pre-allocation is the thing the paper's assembly measurements are taken over, so
it stays. The consequence is that the engine accepts exactly ``batch_size``
samples of exactly ``max_seq_len`` tokens and nothing else, which is fine for a
sweep harness driving full batches and useless for ``embed(["one sentence"])``.

This module is the adapter between the two: chunk the caller's list, pad the
final short chunk up to ``batch_size``, and let the caller drop the padding from
the results. Nothing here relaxes an assertion.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeVar

T = TypeVar("T")


def chunk_and_pad(
    items: Sequence[T], batch_size: int
) -> Iterator[tuple[list[T], int]]:
    """Yield ``(chunk, n_valid)`` where every ``chunk`` has exactly ``batch_size`` entries.

    ``n_valid`` is how many leading entries are real; the rest are padding and
    the caller must slice them off (``result[:n_valid]``).

    Padding repeats the chunk's last real item rather than inserting a sentinel.
    That keeps the padded rows inside the tokenizer's and the adapter store's
    domain — no fabricated tenant id to look up, no empty string to tokenize —
    at the cost of computing a few duplicate rows that are then discarded. Since
    the engine runs a full ``batch_size`` forward pass either way, those rows
    cost nothing extra.

    Args:
        items:      The request list, in order. May be empty.
        batch_size: The engine's configured batch size. Must be positive.

    Raises:
        ValueError: if ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for start in range(0, len(items), batch_size):
        chunk = list(items[start : start + batch_size])
        n_valid = len(chunk)
        if n_valid < batch_size:
            chunk.extend([chunk[-1]] * (batch_size - n_valid))
        yield chunk, n_valid
