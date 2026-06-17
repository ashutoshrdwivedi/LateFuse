"""
Capture model architecture + environment metadata so paper/build_numbers.py
can read these values from CSV instead of hard-coded constants at the top
of the script.

Run on the same GPU host (and same software stack) that produces the other
benchmarks/results/*.csv files. Output is small -- two CSVs we copy back
to the repo, commit, and re-run paper/build_numbers.py against.

Outputs:
    benchmarks/results/model_metadata.csv
        long form: rows of (model, key, value); easy to read in Python and
        easy to add new keys without breaking column order.
    benchmarks/results/env_metadata.csv
        wide form: one row of (gpu_name, torch_version, cuda_version, ...).

Usage:
    uv run python benchmarks/profiling/model_metadata.py

    # Override which models to inspect (must specify target-modules per the
    # model architecture; bge / bert / e5 use "query,value", mpnet uses "q,v"):
    uv run python benchmarks/profiling/model_metadata.py \\
        --model BAAI/bge-m3:query,value \\
        --model sentence-transformers/paraphrase-mpnet-base-v2:q,v

Notes:
  * "total_params" counts every parameter in AutoModel.from_pretrained(name).
    This is the headline number for fp16 model footprint.
  * "embedding_params" is the (token + position + token-type) embedding block.
  * "transformer_layer_params" is the sum of params inside the stacked
    encoder blocks (no embeddings, no pooler). This is the denominator the
    paper uses for the "LoRA-r8 saves X over full fine-tune of those layers"
    multiplier.
  * "lora_params_r{R}" is analytic: 2 * |M| * L * d * r (A is r×d, B is d×r,
    both trainable; one A and one B per target module per layer).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import sys
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoConfig, AutoModel

# (model_name_on_hub, [LoRA target-module names])
# Target modules are needed only to compute the LoRA param count below.
DEFAULT_MODELS: list[tuple[str, list[str]]] = [
    ("BAAI/bge-m3", ["query", "value"]),
    ("sentence-transformers/paraphrase-mpnet-base-v2", ["q", "v"]),
]
LORA_RANKS = (4, 8, 16, 32)

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "benchmarks" / "results"


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def get_embedding_params(model: torch.nn.Module) -> int | None:
    """Sum of params in the embedding block (token + position + token-type)."""
    for attr in ("embeddings", "embed_tokens", "wte"):
        if hasattr(model, attr):
            return count_params(getattr(model, attr))
    return None


def get_transformer_layer_params(model: torch.nn.Module) -> int | None:
    """Sum of params just in the stacked transformer blocks.

    Returns None if the model doesn't expose .encoder.layer (the BERT/MPNet
    convention used by the codebase). Doesn't include embeddings or pooler.
    """
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return sum(count_params(blk) for blk in model.encoder.layer)
    return None


def analytic_lora_params(hidden: int, layers: int,
                         n_target_modules: int, rank: int) -> int:
    """LoRA params per adapter.

    Per (layer, target_module): A in R^{r×d} and B in R^{d×r}, both trainable.
    Total = 2 * |M| * L * d * r.
    """
    return 2 * n_target_modules * layers * hidden * rank


def parse_model_arg(s: str) -> tuple[str, list[str]]:
    """Parse 'name:tm1,tm2' into ('name', ['tm1', 'tm2'])."""
    if ":" not in s:
        raise argparse.ArgumentTypeError(
            f"expected 'model:tm1,tm2', got {s!r}"
        )
    name, tms = s.split(":", 1)
    return name, [t.strip() for t in tms.split(",") if t.strip()]


def inspect_one(name: str, target_modules: list[str]
                ) -> Iterable[tuple[str, str, str]]:
    """Yield (model, key, value) rows for one model."""
    print(f"loading {name} ...", flush=True)
    cfg = AutoConfig.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()

    hidden = cfg.hidden_size
    layers = cfg.num_hidden_layers
    heads = cfg.num_attention_heads
    vocab = getattr(cfg, "vocab_size", "")

    total = count_params(model)
    embed = get_embedding_params(model)
    layer_p = get_transformer_layer_params(model)

    print(f"  hidden_size={hidden} num_layers={layers} num_heads={heads}")
    print(f"  total_params={total:,}  embedding={embed!r}  "
          f"transformer_layers={layer_p!r}")

    yield name, "hidden_size", str(hidden)
    yield name, "num_layers", str(layers)
    yield name, "num_heads", str(heads)
    yield name, "vocab_size", str(vocab)
    yield name, "total_params", str(total)
    yield name, "embedding_params", "" if embed is None else str(embed)
    yield name, "transformer_layer_params", (
        "" if layer_p is None else str(layer_p)
    )
    yield name, "target_modules", ",".join(target_modules)
    yield name, "num_target_modules", str(len(target_modules))

    for r in LORA_RANKS:
        p = analytic_lora_params(hidden, layers, len(target_modules), r)
        yield name, f"lora_params_r{r}", str(p)
        yield name, f"lora_bytes_fp16_r{r}", str(p * 2)

    # Free memory before next model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_model_csv(rows: list[tuple[str, str, str]]) -> Path:
    out = OUT_DIR / "model_metadata.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "key", "value"])
        w.writerows(rows)
    return out


def write_env_csv() -> Path:
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        gpu = "cpu"
        mem_gb = 0.0

    try:
        import peft
        peft_ver = peft.__version__
    except ImportError:
        peft_ver = ""
    try:
        import transformers
        tf_ver = transformers.__version__
    except ImportError:
        tf_ver = ""

    env = {
        "gpu_name": gpu,
        "gpu_total_mem_gb": f"{mem_gb:.1f}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "transformers_version": tf_ver,
        "peft_version": peft_ver,
        "captured_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    out = OUT_DIR / "env_metadata.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(env.keys()))
        w.writeheader()
        w.writerow(env)

    print("\nenvironment:")
    for k, v in env.items():
        print(f"  {k}: {v}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model", action="append", type=parse_model_arg, default=None,
        help="Repeatable. Format: 'hf-name:tm1,tm2'. "
             "If omitted, uses the default list.",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    models = args.model if args.model else DEFAULT_MODELS
    rows: list[tuple[str, str, str]] = []
    for name, tms in models:
        rows.extend(inspect_one(name, tms))

    model_out = write_model_csv(rows)
    env_out = write_env_csv()
    print(f"\nwrote {model_out}  ({len(rows)} rows)")
    print(f"wrote {env_out}")
    print("\nNext: copy these two CSVs back to your local repo and re-run")
    print("      uv run python paper/build_numbers.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
