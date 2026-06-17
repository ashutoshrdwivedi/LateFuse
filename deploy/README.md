# Deploying LoRA Multi-Tenant Serving to Kubernetes

Production deployment of the `lora_serving` inference engine on DigitalOcean Managed Kubernetes with GPU droplets.

## Architecture

```
                    ┌─────────────────────────────────┐
                    │     NGINX Ingress Controller     │
                    │  consistent-hash on X-Tenant-Id  │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
     ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
     │   Pod 1 (GPU)   │ │   Pod 2 (GPU)   │ │   Pod 3 (GPU)   │
     │   FastAPI +     │ │   FastAPI +     │ │   FastAPI +     │
     │   DynamicBatcher│ │   DynamicBatcher│ │   DynamicBatcher│
     │   ┌───────────┐ │ │   ┌───────────┐ │ │   ┌───────────┐ │
     │   │ Base Model│ │ │   │ Base Model│ │ │   │ Base Model│ │
     │   │ + Adapters│ │ │   │ + Adapters│ │ │   │ + Adapters│ │
     │   └───────────┘ │ │   └───────────┘ │ │   └───────────┘ │
     └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
              │                   │                    │
              └───────────────────┼────────────────────┘
                                  ▼
                      ┌───────────────────────┐
                      │   Redis (pub/sub)     │
                      │ lora:adapter_updates  │
                      └───────────────────────┘
                                  ▲
                                  │ publish on retrain
                      ┌───────────────────────┐
                      │  Training Pipeline    │
                      │  (external)           │
                      └───────────────────────┘
```

## Key Features

- **Dynamic Batching**: Individual HTTP requests are batched into GPU-efficient batches (configurable size + timeout)
- **Sticky Routing**: NGINX consistent-hash on `X-Tenant-Id` header ensures same tenant → same pod → adapter already in GPU memory
- **Hot Reload**: Redis pub/sub notifies all pods when an adapter is retrained; pods reload in-place with no restart
- **S3-compatible storage**: Works with DigitalOcean Spaces (or any S3-compatible API)

## Prerequisites

- DigitalOcean Kubernetes cluster with GPU nodes
- DigitalOcean Container Registry (`doctl registry create`)
- DigitalOcean Spaces bucket for adapter storage
- NGINX Ingress Controller installed on the cluster

## Quick Start

### 1. Build & Push the Docker Image

```bash
# From the repo root
cd /path/to/lora_scaling

# Build (bakes in the base model — ~2GB image)
docker build -f deploy/Dockerfile -t lora-serving:latest .

# Tag & push to DigitalOcean Container Registry
docker tag lora-serving:latest registry.digitalocean.com/YOUR_REGISTRY/lora-serving:latest
docker push registry.digitalocean.com/YOUR_REGISTRY/lora-serving:latest
```

### 2. Configure Secrets

Edit `deploy/k8s/configmap.yaml` and set:
- `LORA_S3_ENDPOINT_URL` → your Spaces endpoint (e.g., `https://nyc3.digitaloceanspaces.com`)
- `LORA_S3_BUCKET` → your Spaces bucket name
- S3 credentials in the `Secret` resource
- `LORA_PRELOAD_ADAPTER_IDS` → list of adapter IDs to load at startup

Also update `deploy/k8s/deployment.yaml`:
- Replace `registry.digitalocean.com/YOUR_REGISTRY/lora-serving:latest` with your actual registry URL

And `deploy/k8s/ingress.yaml`:
- Replace `lora-serving.example.com` with your domain

### 3. Deploy to Kubernetes

```bash
# Install NGINX Ingress Controller (if not already installed)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/do/deploy.yaml

# Deploy everything
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/redis.yaml
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
kubectl apply -f deploy/k8s/ingress.yaml
kubectl apply -f deploy/k8s/hpa.yaml

# Watch pods come up
kubectl -n lora-serving get pods -w
```

### 4. Test Inference

```bash
# Wait for pods to be ready
kubectl -n lora-serving wait --for=condition=ready pod -l app.kubernetes.io/name=lora-serving --timeout=300s

# Send a request (with X-Tenant-Id for sticky routing)
curl -X POST https://lora-serving.example.com/predict \
  -H "Content-Type: application/json" \
  -H "X-Tenant-Id: tenant_42" \
  -d '{"tenant_id": "tenant_42", "text": "This is a great product!"}'
```

### 5. Hot-Reload an Adapter

When a tenant retrains their adapter, publish an update to Redis:

```python
from deploy.server.reload import publish_adapter_update

publish_adapter_update(
    redis_url="redis://redis.lora-serving.svc.cluster.local:6379/0",
    channel="lora:adapter_updates",
    adapter_id="tenant_42",
    s3_key="adapters/tenant_42/pytorch_adapter.bin",
)
```

Or via `redis-cli`:
```bash
kubectl -n lora-serving exec -it deploy/redis -- redis-cli
> PUBLISH lora:adapter_updates '{"adapter_id": "tenant_42", "s3_key": "adapters/tenant_42/pytorch_adapter.bin"}'
```

Or via the admin API:
```bash
curl -X POST https://lora-serving.example.com/admin/reload \
  -H "Content-Type: application/json" \
  -d '{"adapter_id": "tenant_42", "s3_key": "adapters/tenant_42/pytorch_adapter.bin"}'
```

## Configuration Reference

All settings are configured via environment variables (prefix `LORA_`):

| Variable | Default | Description |
|---|---|---|
| `LORA_MODEL_NAME` | `intfloat/multilingual-e5-small` | HuggingFace model name |
| `LORA_LORA_RANK` | `8` | LoRA rank |
| `LORA_MAX_BATCH_SIZE` | `32` | Max requests per GPU batch |
| `LORA_BATCH_TIMEOUT_MS` | `10.0` | Max wait before flushing partial batch |
| `LORA_S3_ENDPOINT_URL` | `""` | S3 endpoint (DigitalOcean Spaces URL) |
| `LORA_S3_BUCKET` | `lora-adapters` | S3 bucket for adapter files |
| `LORA_REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `LORA_REDIS_CHANNEL` | `lora:adapter_updates` | Redis pub/sub channel |
| `LORA_PRELOAD_ADAPTER_IDS` | `[]` | Adapter IDs to load at startup |

## File Structure

```
deploy/
├── Dockerfile              # Multi-stage build with baked-in model
├── requirements.txt        # Server-only dependencies
├── README.md               # This file
├── server/
│   ├── __init__.py
│   ├── config.py           # Pydantic settings from env vars
│   ├── app.py              # FastAPI app (endpoints + lifecycle)
│   ├── batcher.py          # Async dynamic request batcher
│   ├── routing.py          # Consistent-hash ring (app-level)
│   └── reload.py           # Redis pub/sub adapter hot-reload
└── k8s/
    ├── namespace.yaml
    ├── configmap.yaml      # Config + S3 credentials Secret
    ├── redis.yaml          # Single-instance Redis for pub/sub
    ├── deployment.yaml     # GPU inference pods
    ├── service.yaml        # ClusterIP with session affinity
    ├── ingress.yaml        # NGINX with consistent-hash routing
    └── hpa.yaml            # Autoscaler (GPU util or CPU fallback)
```
