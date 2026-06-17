"""Redis pub/sub listener for hot-reloading LoRA adapters without pod restarts.

When a tenant retrains their model, an external system publishes an update
message to the Redis channel. Every inference pod subscribes to this channel
and reloads the affected adapter in-place on the GPU.

Message format (JSON):
    {"adapter_id": "tenant_42", "s3_key": "adapters/tenant_42/pytorch_adapter.bin"}

Thread safety:
    A threading.Lock prevents reloads from occurring during an active forward pass.
    The reload itself is fast (~50ms for a 20MB adapter) so lock contention is minimal.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Callable

import boto3
import redis

from deploy.server.config import ServerConfig

logger = logging.getLogger(__name__)


class AdapterReloader:
    """Subscribes to Redis pub/sub and hot-reloads adapters on notification.

    Usage:
        reloader = AdapterReloader(config, reload_callback)
        reloader.start()   # starts background subscriber thread
        ...
        reloader.stop()    # on shutdown
    """

    def __init__(
        self,
        config: ServerConfig,
        reload_callback: Callable[[str, str], None],  # (adapter_id, local_path) -> None
        inference_lock: threading.Lock | None = None,
    ):
        self._config = config
        self._reload_callback = reload_callback
        self._inference_lock = inference_lock or threading.Lock()
        self._redis: redis.Redis | None = None
        self._pubsub: redis.client.PubSub | None = None
        self._thread: threading.Thread | None = None
        self._running = False

        # S3 client (DigitalOcean Spaces compatible)
        s3_kwargs = {
            "service_name": "s3",
            "region_name": config.s3_region,
        }
        if config.s3_endpoint_url:
            s3_kwargs["endpoint_url"] = config.s3_endpoint_url
        if config.s3_access_key:
            s3_kwargs["aws_access_key_id"] = config.s3_access_key
            s3_kwargs["aws_secret_access_key"] = config.s3_secret_key

        self._s3 = boto3.client(**s3_kwargs)
        self._local_dir = Path(config.adapter_local_dir)
        self._local_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """Start the background Redis subscriber thread."""
        self._redis = redis.from_url(self._config.redis_url)
        self._pubsub = self._redis.pubsub()
        self._pubsub.subscribe(self._config.redis_channel)
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True, name="adapter-reloader")
        self._thread.start()
        logger.info("Adapter reloader started, listening on channel '%s'", self._config.redis_channel)

    def stop(self) -> None:
        """Stop the subscriber thread and clean up."""
        self._running = False
        if self._pubsub:
            self._pubsub.unsubscribe()
            self._pubsub.close()
        if self._redis:
            self._redis.close()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Adapter reloader stopped")

    def _listen(self) -> None:
        """Background thread: listen for adapter update messages."""
        assert self._pubsub is not None
        for message in self._pubsub.listen():
            if not self._running:
                break
            if message["type"] != "message":
                continue
            try:
                payload = json.loads(message["data"])
                adapter_id = payload["adapter_id"]
                s3_key = payload["s3_key"]
                logger.info("Received reload request: adapter_id=%s, s3_key=%s", adapter_id, s3_key)
                self._download_and_reload(adapter_id, s3_key)
            except Exception:
                logger.error("Failed to process reload message: %s", message["data"], exc_info=True)

    def _download_and_reload(self, adapter_id: str, s3_key: str) -> None:
        """Download adapter from S3 and reload it into GPU memory."""
        local_path = self._local_dir / f"{adapter_id}.bin"

        # Download from S3 / DigitalOcean Spaces
        logger.info("Downloading %s/%s → %s", self._config.s3_bucket, s3_key, local_path)
        self._s3.download_file(self._config.s3_bucket, s3_key, str(local_path))

        # Acquire lock to prevent reload during active inference
        with self._inference_lock:
            logger.info("Reloading adapter '%s' from %s", adapter_id, local_path)
            self._reload_callback(adapter_id, str(local_path))

        logger.info("Adapter '%s' reloaded successfully", adapter_id)

    def download_adapter(self, adapter_id: str, s3_key: str) -> str:
        """Download an adapter file from S3 and return local path.

        Used during startup to pre-load adapters.
        """
        local_path = self._local_dir / f"{adapter_id}.bin"
        if not local_path.exists():
            logger.info("Downloading adapter '%s' from s3://%s/%s", adapter_id, self._config.s3_bucket, s3_key)
            self._s3.download_file(self._config.s3_bucket, s3_key, str(local_path))
        return str(local_path)


def publish_adapter_update(
    redis_url: str,
    channel: str,
    adapter_id: str,
    s3_key: str,
) -> None:
    """Publish an adapter update notification (called by training pipeline).

    Example:
        publish_adapter_update(
            redis_url="redis://redis:6379/0",
            channel="lora:adapter_updates",
            adapter_id="tenant_42",
            s3_key="adapters/tenant_42/pytorch_adapter.bin",
        )
    """
    r = redis.from_url(redis_url)
    payload = json.dumps({"adapter_id": adapter_id, "s3_key": s3_key})
    r.publish(channel, payload)
    r.close()
    logger.info("Published adapter update: %s → %s", adapter_id, s3_key)
