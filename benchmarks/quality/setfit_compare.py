"""Quality benchmark: LoRA-SetFit vs vanilla SetFit on the same few-shot data.

For each dataset, samples N label-balanced examples, trains both methods with
identical hyperparameters and seed, and evaluates on the held-out test split.
Supports multi-seed sweeps and per-method LR sweeps (LoRA only).

A third method, "frozen", skips contrastive body training entirely and fits
only the LR head on the pre-trained embeddings (no-adaptation baseline).

Usage:
    uv run python benchmarks/quality/setfit_compare.py \\
        --base-model sentence-transformers/paraphrase-mpnet-base-v2 \\
        -t q -t v \\
        --datasets SetFit/sst2 --datasets SetFit/emotion \\
        --seeds 0 --seeds 1 --seeds 2 \\
        --lora-lrs 1e-4 --lora-lrs 3e-4 --lora-lrs 5e-4 \\
        --n-per-class 8 --lora-rank 8 \\
        --out quality_results.csv
"""

from __future__ import annotations

import csv
import gc
import math
import random
import statistics
import time
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import torch
import typer
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from setfit import SetFitModel, Trainer, TrainingArguments


# Per-dataset metric overrides per SetFit paper Table 6.
# Datasets not listed default to "accuracy".
DATASET_METRIC = {
    "SetFit/amazon_counterfactual_en": "matthews_correlation",
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_few_shot(ds: Dataset, n_per_class: int, seed: int) -> Dataset:
    """Stratified sample: exactly n_per_class examples per label."""
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[int]] = {}
    for i, lbl in enumerate(ds["label"]):
        by_label.setdefault(int(lbl), []).append(i)
    indices: list[int] = []
    for lbl, idxs in by_label.items():
        if len(idxs) < n_per_class:
            raise ValueError(
                f"Label {lbl} has {len(idxs)} examples in train; need >= {n_per_class}"
            )
        indices.extend(rng.choice(idxs, size=n_per_class, replace=False).tolist())
    return ds.select(sorted(indices))


def _build_model(
    base_name: str,
    use_lora: bool,
    lora_rank: int,
    target_modules: list[str],
    max_seq_length: int,
) -> SetFitModel:
    """Load SetFitModel; optionally wrap the encoder body with PEFT LoRA."""
    model = SetFitModel.from_pretrained(base_name)
    # SetFit paper §4.4 uses max_seq_length=256, overriding the model's default.
    model.model_body.max_seq_length = max_seq_length
    if use_lora:
        auto_model = model.model_body[0].auto_model
        # alpha=r so scaling = alpha/r = 1.0, matches our serving impl (no scaling factor).
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model.model_body[0].auto_model = get_peft_model(auto_model, lora_cfg)
    return model


def _count_trainable_body_params(model: SetFitModel) -> int:
    return sum(p.numel() for p in model.model_body.parameters() if p.requires_grad)


def _train_and_evaluate(
    method: str,
    base_name: str,
    dataset_name: str,
    n_per_class: int,
    seed: int,
    lora_rank: int,
    target_modules: list[str],
    max_seq_length: int,
    num_epochs: int,
    batch_size: int,
    body_lr: float,
    metric: str,
    eval_size: Optional[int],
) -> dict:
    _set_seed(seed)
    ds = load_dataset(dataset_name)
    train_full = ds["train"]
    test_split = ds.get("test") or ds.get("validation")
    if test_split is None:
        raise ValueError(f"{dataset_name} has no test or validation split")

    train_ds = _sample_few_shot(train_full, n_per_class, seed)
    if eval_size is not None and len(test_split) > eval_size:
        eval_ds = test_split.shuffle(seed=seed).select(range(eval_size))
    else:
        eval_ds = test_split

    use_lora = method == "lora"
    model = _build_model(
        base_name,
        use_lora=use_lora,
        lora_rank=lora_rank,
        target_modules=target_modules,
        max_seq_length=max_seq_length,
    )
    if method == "frozen":
        for p in model.model_body.parameters():
            p.requires_grad_(False)
    trainable = _count_trainable_body_params(model)

    args = TrainingArguments(
        batch_size=batch_size,
        num_epochs=num_epochs,
        body_learning_rate=body_lr,
        seed=seed,
        save_strategy="no",
        report_to="none",
        eval_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        metric=metric,
    )

    t0 = time.perf_counter()
    if method == "frozen":
        # Head-only fit on pre-trained embeddings; no contrastive body pass.
        trainer.train_classifier(train_ds["text"], train_ds["label"], args=args)
    else:
        trainer.train()
    train_seconds = time.perf_counter() - t0

    metrics = trainer.evaluate()
    score = float(metrics.get(metric, metrics.get("accuracy", 0.0)))
    n_eval = len(eval_ds)

    del trainer, model, args
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "dataset": dataset_name,
        "method": method,
        "lora_rank": lora_rank if use_lora else 0,
        "body_lr": body_lr,
        "n_per_class": n_per_class,
        "seed": seed,
        "metric": metric,
        "score": round(score, 4),
        "trainable_body_params": trainable,
        "train_seconds": round(train_seconds, 2),
        "n_eval": n_eval,
    }


def _summarize(rows: list[dict]) -> None:
    """Print mean ± std per (dataset, method, body_lr) group."""
    typer.echo("\n=== Summary (mean ± std across seeds) ===")
    typer.echo(f"{'dataset':<40} {'method':<8} {'body_lr':>8} {'metric':<22} {'mean':>8} {'std':>8} {'n':>3}")
    groups: dict[tuple, list[float]] = {}
    metric_by_key: dict[tuple, str] = {}
    for r in rows:
        key = (r["dataset"], r["method"], r["body_lr"])
        groups.setdefault(key, []).append(r["score"])
        metric_by_key[key] = r["metric"]
    for key in sorted(groups.keys()):
        scores = groups[key]
        mean = statistics.fmean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else float("nan")
        ds_name, method, lr = key
        typer.echo(
            f"{ds_name:<40} {method:<8} {lr:>8.0e} {metric_by_key[key]:<22} "
            f"{mean:>8.4f} {std:>8.4f} {len(scores):>3d}"
        )


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def run(
    datasets: Annotated[list[str], typer.Option("--datasets", "-d", help="HF dataset names (repeatable).")] = ["SetFit/sst2"],
    base_model: Annotated[str, typer.Option(help="HF encoder model name.")] = "sentence-transformers/paraphrase-mpnet-base-v2",
    n_per_class: Annotated[int, typer.Option(help="Examples sampled per class for few-shot training.")] = 8,
    seeds: Annotated[list[int], typer.Option("--seeds", help="Random seeds (repeatable). Paper uses 10.")] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    lora_rank: Annotated[int, typer.Option(help="LoRA rank for the parameter-efficient run.")] = 8,
    target_modules: Annotated[list[str], typer.Option("--target-modules", "-t", help="PEFT target module names. BERT/e5: query value. MPNet: q v.")] = ["q", "v"],
    max_seq_length: Annotated[int, typer.Option(help="Max tokens per input. SetFit paper §4.4: 256.")] = 256,
    num_epochs: Annotated[int, typer.Option(help="SetFit contrastive epochs over the few-shot data.")] = 1,
    batch_size: Annotated[int, typer.Option(help="Contrastive training batch size.")] = 16,
    body_lr: Annotated[float, typer.Option(help="Body LR for vanilla and 'paper-LR' LoRA. SetFit library default reproduces the paper.")] = 2e-5,
    lora_lrs: Annotated[list[float], typer.Option("--lora-lrs", help="Additional LoRA-specific LRs to sweep (in addition to body_lr).")] = [1e-4, 3e-4, 5e-4],
    eval_size: Annotated[Optional[int], typer.Option(help="Optional cap on test-split size for faster eval.")] = None,
    methods: Annotated[list[str], typer.Option("--methods", "-m", help="Which methods to run: vanilla, lora, frozen.")] = ["vanilla", "lora"],
    out: Annotated[Path, typer.Option(help="Output CSV path.")] = Path("quality_results.csv"),
    resume: Annotated[bool, typer.Option(help="Skip configs already present in the output CSV.")] = False,
) -> None:
    """Run the full sweep: datasets × methods × LRs × seeds.

    LR matrix:
      - vanilla runs only at body_lr (e.g. 2e-5).
      - lora runs at body_lr AND every value in --lora-lrs.
      - frozen has no body LR (recorded as 0).
    """
    if torch.cuda.is_available():
        typer.echo(f"GPU: {torch.cuda.get_device_name(0)}")
        free_gb = (torch.cuda.get_device_properties(0).total_memory) / 1e9
        typer.echo(f"GPU memory: {free_gb:.1f} GB")
    else:
        typer.echo("WARNING: No CUDA GPU detected — training will be slow on CPU.")

    out.parent.mkdir(parents=True, exist_ok=True)

    # (method, body_lr) pairs to evaluate
    configs: list[tuple[str, float]] = []
    if "frozen" in methods:
        configs.append(("frozen", 0.0))
    if "vanilla" in methods:
        configs.append(("vanilla", body_lr))
    if "lora" in methods:
        configs.append(("lora", body_lr))
        for lr in lora_lrs:
            configs.append(("lora", lr))

    # Resume support
    done: set[tuple] = set()
    existing_rows: list[dict] = []
    if resume and out.exists():
        with out.open() as f:
            reader = csv.DictReader(f)
            for r in reader:
                key = (r["dataset"], r["method"], float(r["body_lr"]), int(r["seed"]))
                done.add(key)
                # Coerce CSV strings back to numbers for the summary
                r = {**r, "body_lr": float(r["body_lr"]), "seed": int(r["seed"]),
                     "score": float(r["score"]), "lora_rank": int(r["lora_rank"]),
                     "n_per_class": int(r["n_per_class"]),
                     "trainable_body_params": int(r["trainable_body_params"]),
                     "train_seconds": float(r["train_seconds"]),
                     "n_eval": int(r["n_eval"])}
                existing_rows.append(r)
        typer.echo(f"Resume: skipping {len(done)} already-completed configs from {out}")

    mode = "a" if resume and out.exists() else "w"
    csv_handle = out.open(mode, newline="")
    rows: list[dict] = list(existing_rows)
    writer: Optional[csv.DictWriter] = None
    if mode == "a" and existing_rows:
        # Reuse header that's already in the file
        writer = csv.DictWriter(csv_handle, fieldnames=existing_rows[0].keys())

    total = len(datasets) * len(configs) * len(seeds)
    done_in_grid = sum(
        1 for d in datasets for mth, lr in configs for s in seeds
        if (d, mth, lr, s) in done
    )
    pending = total - done_in_grid
    typer.echo(f"Total configs: {total}  Pending: {pending}  "
               f"(datasets={len(datasets)} × method-LR pairs={len(configs)} × seeds={len(seeds)})")

    n_done = 0
    try:
        for dataset in datasets:
            metric = DATASET_METRIC.get(dataset, "accuracy")
            for method, lr in configs:
                for seed in seeds:
                    key = (dataset, method, lr, seed)
                    if key in done:
                        continue
                    n_done += 1
                    typer.echo(
                        f"\n[{n_done}/{pending}] {dataset} | {method} | lr={lr:g} | seed={seed} | "
                        f"metric={metric}"
                    )
                    row = _train_and_evaluate(
                        method=method,
                        base_name=base_model,
                        dataset_name=dataset,
                        n_per_class=n_per_class,
                        seed=seed,
                        lora_rank=lora_rank,
                        target_modules=target_modules,
                        max_seq_length=max_seq_length,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        body_lr=lr,
                        metric=metric,
                        eval_size=eval_size,
                    )
                    typer.echo(
                        f"  score={row['score']:.4f}  "
                        f"trainable={row['trainable_body_params']:,}  "
                        f"train_s={row['train_seconds']}"
                    )
                    if writer is None:
                        writer = csv.DictWriter(csv_handle, fieldnames=row.keys())
                        writer.writeheader()
                    writer.writerow(row)
                    csv_handle.flush()
                    rows.append(row)
    finally:
        csv_handle.close()

    typer.echo(f"\nSaved {len(rows)} rows total to {out}")
    _summarize(rows)


if __name__ == "__main__":
    app()
