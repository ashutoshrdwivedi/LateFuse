"""Async dynamic request batcher for GPU inference.

Collects individual HTTP requests into batches, flushes them through the model
in a single forward pass, and fans results back to callers via asyncio.Futures.

This is the key component that turns per-request latency into per-batch throughput.
Without it, every HTTP request would trigger a batch_size=1 forward pass on the GPU.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """A single inference request waiting to be batched."""

    tenant_id: str
    text: str
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    enqueued_at: float = field(default_factory=time.monotonic)


@dataclass
class InferenceResult:
    """Result for a single sample in the batch."""

    scores: list[float]
    latency_ms: float


class DynamicBatcher:
    """Collects requests and flushes them as batches through the model.

    The batcher runs a background asyncio task that:
    1. Waits for requests to arrive in the queue
    2. Collects up to max_batch_size requests (or waits batch_timeout_ms)
    3. Calls the provided inference_fn with the collected batch
    4. Distributes results back to each request's Future

    Usage:
        batcher = DynamicBatcher(max_batch_size=32, batch_timeout_ms=10.0, inference_fn=my_fn)
        await batcher.start()
        result = await batcher.submit("tenant_42", "some text to classify")
        await batcher.stop()
    """

    def __init__(
        self,
        max_batch_size: int,
        batch_timeout_ms: float,
        inference_fn: Any,  # Callable[[list[InferenceRequest]], list[InferenceResult]]
    ):
        self._max_batch_size = max_batch_size
        self._batch_timeout_ms = batch_timeout_ms
        self._inference_fn = inference_fn
        self._queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the background batching loop."""
        self._running = True
        self._task = asyncio.create_task(self._batch_loop())
        logger.info(
            "Batcher started (max_batch=%d, timeout=%.1fms)",
            self._max_batch_size,
            self._batch_timeout_ms,
        )

    async def stop(self) -> None:
        """Stop the background batching loop gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Batcher stopped")

    async def submit(self, tenant_id: str, text: str) -> InferenceResult:
        """Submit a single inference request and wait for the result.

        This is called from the FastAPI endpoint handler. The caller awaits
        until the batcher flushes a batch containing this request.
        """
        request = InferenceRequest(tenant_id=tenant_id, text=text)
        await self._queue.put(request)
        return await request.future

    async def _batch_loop(self) -> None:
        """Background loop: collect requests, form batches, run inference."""
        while self._running:
            batch: list[InferenceRequest] = []

            try:
                # Block until at least one request arrives
                first = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,  # check self._running periodically
                )
                batch.append(first)
            except asyncio.TimeoutError:
                continue

            # Collect more requests up to max_batch_size or timeout
            deadline = time.monotonic() + (self._batch_timeout_ms / 1000.0)
            while len(batch) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

            # Run inference for the batch
            if batch:
                await self._process_batch(batch)

    async def _process_batch(self, batch: list[InferenceRequest]) -> None:
        """Run inference on a batch and deliver results to futures."""
        t0 = time.monotonic()
        try:
            # inference_fn runs the model — it's sync (GPU-bound),
            # so we run it in an executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self._inference_fn, batch)

            latency_ms = (time.monotonic() - t0) * 1000
            logger.debug(
                "Batch of %d processed in %.1fms (%.1f samples/s)",
                len(batch),
                latency_ms,
                len(batch) / (latency_ms / 1000) if latency_ms > 0 else 0,
            )

            for req, result in zip(batch, results):
                if not req.future.done():
                    req.future.set_result(result)

        except Exception as e:
            logger.error("Batch inference failed: %s", e, exc_info=True)
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
