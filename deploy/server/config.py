"""Server configuration via environment variables.

Uses pydantic-settings to validate and parse env vars into typed config.
All settings can be overridden via K8s ConfigMap → env vars.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    """Inference server configuration.

    All fields map to environment variables (uppercase, prefixed with LORA_).
    Example: LORA_MODEL_NAME=intfloat/multilingual-e5-small
    """

    model_config = {"env_prefix": "LORA_"}

    # --- Model ---
    model_name: str = "intfloat/multilingual-e5-small"
    lora_rank: int = 8
    max_seq_len: int = 512
    target_modules: list[str] = ["query", "value"]
    num_labels: int = 10

    # --- Batching ---
    max_batch_size: int = 32
    batch_timeout_ms: float = 10.0  # max wait before flushing a partial batch

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1  # GPU inference is single-process

    # --- Adapter storage (S3-compatible — works with DigitalOcean Spaces) ---
    s3_endpoint_url: str = ""  # e.g. https://nyc3.digitaloceanspaces.com
    s3_bucket: str = "lora-adapters"
    s3_region: str = "nyc3"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    adapter_local_dir: str = "/tmp/adapters"  # local cache for downloaded adapters

    # --- Redis (pub/sub for adapter hot-reload) ---
    redis_url: str = "redis://localhost:6379/0"
    redis_channel: str = "lora:adapter_updates"

    # --- Device ---
    device: str = "cuda:0"
    dtype: str = "float32"

    # --- Adapter preload ---
    preload_adapter_ids: list[str] = []  # adapter IDs to load at startup
