"""Adapter-registration churn microbenchmark.

The question is whether the O(N^2)-vs-O(1) cold-start result translates
to production impact -- "what % of serving time is spent in adapter
registration under realistic churn rates?".  Answering it needs three things
the existing ``lora_serving.benchmark.run --churn`` path does not provide.

**1. Registration has to be the production path.**  The currently reported
number (0.29 ms/adapter) comes from ``adapter_load_total_s / N`` on the
startup preload, i.e. from :meth:`AdapterStore.load_synthetic` -- a
``torch.manual_seed`` + ``nn.init.normal_`` loop that generates random weights
on the GPU with curand.  It reads no file and performs no host-to-device copy,
so it cannot back the sentence "registration is an O(1) memcpy".  The shipped
registration path is :meth:`AdapterStore.load_from_file`, which
``deploy/server/app.py`` names in the comment beside its synthetic stub and
which ``deploy/server/reload.py`` feeds from S3.  That is what ``--registration
-path file`` measures.  ``synthetic`` is kept as an ablation arm so the gap
between the two is reported rather than silently inherited.

**2. Churn rate has to be an independent variable.**  In the older harness the
churn rate was an emergent byproduct of (Zipf alpha, tenant count, capacity),
which meant defending a particular alpha over a particular tenant population as
"realistic".  Here admissions are driven directly at a target rate, so a reader
maps their own workload onto the axis instead of onto our alpha.

Note this makes the request stream's hit rate uninteresting by construction:
requests always draw from the *current* resident pool, so every request hits and
no hit/miss metric is collected.  Zipf alpha shapes only which resident adapters
are popular, and since replacement walks slots round-robin rather than by
popularity, alpha has little influence on the result at all.

The churn model is pool *replacement*, not demand paging: at the target rate a
resident adapter is evicted and a fresh one registered in its place, while the
request stream draws Zipf over the resident pool.  That matches the deployment
in ``deploy/server/reload.py`` -- adapters are hot-reloaded on a Redis
notification when a tenant retrains, they are not faulted in on a request miss.

**3. Eviction and registration have to be billed together.**  ``evict()`` is a
``del``, so it returns the block to the caching allocator and looks nearly
free, while the real cost resurfaces in the next allocation and gets billed to
registration.  Measuring the two separately is what produced a cost curve that
was non-monotonic in the driving variable (72.7 ms/admission at N=20k/B=32
against 7.1 ms at N=20k/B=64, which has twice the misses).  The headline unit
here is therefore one *replacement* = evict + register, with the two also
reported separately for diagnosis.

Run:
    uv run python -m lora_serving.benchmark.churn \\
        --corpus-dir /workspace/adapter_corpus \\
        --registration-paths synthetic file file_pinned \\
        --resident 1000 47000 \\
        --admission-rates 0.1 1 10 100 \\
        --churn-modes blocking background \\
        --seeds 1 2 3 \\
        --out benchmarks/results/churn_registration.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import queue
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import torch

from lora_serving.benchmark.synthetic import (
    make_synthetic_adapters,
    make_synthetic_inputs,
    make_synthetic_lr_weights,
)
from lora_serving.config import LoraServingConfig
from lora_serving.model.encoder import EncoderWithLora
from lora_serving.model.hf_wrapper import HFEncoderWithLora
from lora_serving.weights.batch import BatchAssembler
from lora_serving.weights.store import AdapterStore, LoraWeight

DTYPE_MAP = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}

# Registration arms. "file" is the shipped path; the other two exist to explain
# it -- "synthetic" reproduces the currently reported number, and
# "file_pinned" bounds how much of the file path is torch.load overhead versus
# the host-to-device copy itself.
REGISTRATION_PATHS = ("synthetic", "file", "file_pinned")
CHURN_MODES = ("blocking", "background")

# Cap on admissions owed but not yet performed. A target rate above what the
# arm can sustain is a legitimate cell -- it is how the saturation point gets
# measured -- but it has to degrade into "achieved rate < target rate" rather
# than into unbounded work. Without a cap the blocking drain loop livelocks
# (debt accrues faster than admissions retire, so the deadline check outside
# the loop is never reached) and the background queue grows without bound.
# Shedding past the cap is what a server with backpressure would do, and the
# shed count is reported so the saturation is visible rather than inferred.
MAX_ADMISSION_BACKLOG = 256


# --------------------------------------------------------------------------
# Corpus of real adapter files
# --------------------------------------------------------------------------
def corpus_key_fn(layer_idx: int, module: str) -> tuple[str, str]:
    """Key resolver for the benchmark corpus written by :func:`build_corpus`.

    Matches the (key_A, key_B) contract of :meth:`AdapterStore.load_from_file`.
    """
    return (
        f"encoder.layer.{layer_idx}.attention.self.{module}.lora_A",
        f"encoder.layer.{layer_idx}.attention.self.{module}.lora_B",
    )


def build_corpus(config: LoraServingConfig, out_dir: Path, n_files: int) -> list[Path]:
    """Write ``n_files`` real adapter .bin files, reusing any already present.

    The files hold the same tensor geometry a trained adapter would -- per
    layer, per target module, A of shape (r, H) and B of shape (H, r), which is
    the column-major layout ``load_from_file`` transposes on ingest.  Contents
    are random; only the shapes, dtype and file size affect what is being timed.

    Written once and reused across runs: at r=8 on bge-m3 each file is 1.57 MB,
    so a 2,000-file corpus is ~3.1 GB on disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    H, R, L = config.hidden_size, config.lora_rank, config.num_layers
    paths = []
    made = 0
    for i in range(n_files):
        path = out_dir / f"corpus_{i:06d}.bin"
        paths.append(path)
        if path.exists():
            continue
        state_dict = {}
        for layer in range(L):
            for module in config.target_modules:
                key_a, key_b = corpus_key_fn(layer, module)
                state_dict[key_a] = torch.randn(R, H, dtype=config.dtype) * 0.02
                state_dict[key_b] = torch.zeros(H, R, dtype=config.dtype)
        torch.save(state_dict, path)
        made += 1
    if made:
        size_mb = paths[0].stat().st_size / 1e6
        print(f"  Corpus: wrote {made} new file(s) to {out_dir} "
              f"({size_mb:.2f} MB each, {made * size_mb / 1000:.2f} GB total)")
    else:
        print(f"  Corpus: reusing {n_files} existing file(s) in {out_dir}")
    return paths


def drop_page_cache(path: Path) -> bool:
    """Evict one file from the OS page cache. Returns False if unsupported.

    ``POSIX_FADV_DONTNEED`` works per-file and without root, which is what lets
    the cold arm be measured on a shared benchmark host.  Not available on
    macOS, where the cold arm degrades to a second warm measurement -- hence
    the return value, which the caller records in the CSV rather than assuming.
    """
    if not hasattr(os, "posix_fadvise"):
        return False
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return True


# --------------------------------------------------------------------------
# Registration arms
# --------------------------------------------------------------------------
class PinnedStager:
    """Reusable pinned host buffers for the staged host-to-device arm.

    A pageable-memory H2D copy has to be bounced through a driver-internal
    staging buffer and cannot overlap with compute; copying via a pre-pinned
    buffer can.  The buffers are allocated once and reused, because per-call
    ``pin_memory()`` would charge cudaHostAlloc to every registration and that
    is not what a server would do.
    """

    def __init__(self, config: LoraServingConfig):
        H, R, L = config.hidden_size, config.lora_rank, config.num_layers
        # Held in the *destination* layout so the H2D copy is contiguous: a
        # transposed view would silently fall back to a blocking copy.
        self.a = {
            m: torch.empty(L, H, R, dtype=config.dtype).pin_memory()
            for m in config.target_modules
        }
        self.b = {
            m: torch.empty(L, R, H, dtype=config.dtype).pin_memory()
            for m in config.target_modules
        }


class Registrar:
    """One registration arm: how a single adapter becomes resident."""

    def __init__(
        self,
        store: AdapterStore,
        config: LoraServingConfig,
        corpus: list[Path],
        path_kind: str,
        num_labels: int,
        lr_coefs: dict,
        lr_intercepts: dict,
        cold: bool = False,
    ):
        self._store = store
        self._config = config
        self._corpus = corpus
        self._kind = path_kind
        self._num_labels = num_labels
        self._lr_coefs = lr_coefs
        self._lr_intercepts = lr_intercepts
        self._cold = cold
        self._stager = PinnedStager(config) if path_kind == "file_pinned" else None
        # The tenant's classification head is part of registration -- a tenant is
        # not servable without one -- but it has to be *transferred*, not
        # generated. Synthesising it on device (as make_synthetic_lr_weights does)
        # charges a curand launch to every admission, so the timed region would
        # not be purely file-load + H2D. Built once on pinned host memory here,
        # register() then pays exactly the host-to-device copy a real head would.
        head = torch.randn(1, num_labels, config.hidden_size, dtype=config.dtype)
        intercept = torch.zeros(1, num_labels, dtype=config.dtype)
        if config.device.type == "cuda":
            head, intercept = head.pin_memory(), intercept.pin_memory()
        self._host_coef = head
        self._host_intercept = intercept
        # True only when a cold run actually managed to evict the page cache.
        # A warm run reports False rather than True-by-default, so the CSV
        # cannot be read as "this was a cold measurement" when it was not.
        self.page_cache_dropped = cold

    def register(self, adapter_id: str, corpus_idx: int) -> None:
        cfg = self._config
        if self._kind == "synthetic":
            self._store.load_synthetic(adapter_id, seed=corpus_idx)
        else:
            path = self._corpus[corpus_idx % len(self._corpus)]
            if self._cold and not drop_page_cache(path):
                self.page_cache_dropped = False
            if self._kind == "file":
                self._store.load_from_file(adapter_id, str(path), corpus_key_fn)
            else:
                self._register_pinned(adapter_id, path)

        # Install the tenant's head by copying it from the pinned host buffer,
        # so the timed region is file-load + host-to-device transfer and nothing
        # else. See __init__ for why this is not generated on device.
        self._lr_coefs[adapter_id] = self._host_coef.to(cfg.device, non_blocking=True)
        self._lr_intercepts[adapter_id] = self._host_intercept.to(
            cfg.device, non_blocking=True
        )

    def _register_pinned(self, adapter_id: str, path: Path) -> None:
        """torch.load to host -> pinned staging buffer -> non_blocking H2D."""
        cfg = self._config
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        weight = LoraWeight(cfg)
        assert self._stager is not None
        for module in cfg.target_modules:
            a_layers, b_layers = [], []
            for layer in range(cfg.num_layers):
                key_a, key_b = corpus_key_fn(layer, module)
                a_layers.append(state_dict[key_a])
                b_layers.append(state_dict[key_b])
            # Stack + transpose on the host into the pinned buffer, then a
            # single contiguous copy per module per matrix.
            self._stager.a[module].copy_(torch.stack(a_layers).transpose(1, 2))
            self._stager.b[module].copy_(torch.stack(b_layers).transpose(1, 2))
            weight.wa[module].copy_(self._stager.a[module], non_blocking=True)
            weight.wb[module].copy_(self._stager.b[module], non_blocking=True)
        self._store.insert(adapter_id, weight)

    def evict(self, adapter_id: str) -> None:
        self._store.evict(adapter_id)
        self._lr_coefs.pop(adapter_id, None)
        self._lr_intercepts.pop(adapter_id, None)


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------
def _zipf_probabilities(n: int, alpha: float) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=np.float64)
    p = 1.0 / np.power(ranks, alpha)
    return p / p.sum()


def run_cell(
    model,
    config: LoraServingConfig,
    assembler: BatchAssembler,
    registrar: Registrar,
    resident: list[str],
    lr_coefs: dict,
    lr_intercepts: dict,
    inputs: dict,
    output_lr: torch.Tensor,
    batch_size: int,
    duration_s: float,
    admission_rate: float,
    churn_mode: str,
    zipf_alpha: float,
    rng: np.random.Generator,
    warmup_s: float,
    next_corpus_idx: int,
) -> dict:
    """Replay one (rate, mode) cell for ``duration_s`` and return raw metrics.

    Serving runs continuously; admissions are injected at ``admission_rate``
    against elapsed wall clock, so the rate is an input rather than something
    the Zipf parameters happen to produce.  In ``blocking`` mode the
    replacement happens inline between batches, which is the worst case and the
    bound.  In ``background`` mode it happens on a side CUDA stream in a worker
    thread. NOTE this is a *proposed* design, not the shipped one:
    ``deploy/server/reload.py`` acquires the same inference lock the serving
    path holds before invoking the reload callback, so production registration
    is serialized against inference and corresponds to ``blocking``. Quote the
    blocking numbers as the production figures; background bounds what an
    overlapped implementation could buy, and at what tail cost.
    """
    device = config.device
    probs = _zipf_probabilities(len(resident), zipf_alpha)

    # The pool is a fixed-length slot array, not a list that grows and shrinks:
    # a replacement swaps one slot, so the Zipf popularity distribution stays
    # bound to a stable set of positions and the request stream is unaffected
    # by churn except through which adapter sits in the churned slot.
    pool: list[str] = list(resident)
    pool_lock = threading.Lock()
    # Slots with an admission in flight. A slot must not be re-admitted before
    # the previous admission publishes: the main thread would read the same
    # victim twice (the new id is not visible until the worker publishes it)
    # and the second retirement would evict an adapter already freed, while the
    # first newcomer leaked. Backlogs deeper than the pool make that reachable,
    # so the slot is owned exclusively for the duration of its admission.
    pending_slots: set[int] = set()

    serve_latencies: list[float] = []
    replace_records: list[tuple[float, float, float]] = []
    records_lock = threading.Lock()
    blocked_ms = 0.0
    admissions = 0
    shed = 0
    corpus_idx = next_corpus_idx

    copy_stream = torch.cuda.Stream(device=device) if churn_mode == "background" else None

    def _sync() -> None:
        """Wait for this thread's own GPU work only.

        In background mode a device-wide ``torch.cuda.synchronize`` would block
        on the serving stream too, which both serializes the copy stream it is
        supposed to be overlapping with and charges serving time to the
        admission measurement.
        """
        if copy_stream is not None:
            copy_stream.synchronize()
        else:
            torch.cuda.synchronize(device)

    # Victims waiting to be freed. In background mode the admitting thread must
    # not call evict() itself: the serving thread samples a batch, releases the
    # pool lock, and only then looks the ids up in the store, so an eviction in
    # that window would KeyError on an adapter an in-flight batch still names.
    # Instead the victim is retired here and freed by the serving thread at the
    # top of its next iteration -- by which point the batch that could have
    # named it has already completed, because serving is synchronous.
    retire_q: queue.Queue = queue.Queue()
    deferred_evict_ms: list[float] = []

    def _register_only(slot: int, victim: str, new_id: str, idx: int) -> float:
        """Register the newcomer and publish it into the slot. Returns register_ms.

        Register-before-evict for two reasons. It removes the window in which a
        pool slot names an adapter that is not resident, which is what a hot
        reload has to guarantee. And it forces every registration to allocate
        fresh rather than reusing the block the eviction just returned to the
        caching allocator -- so the allocation cost lands on the operation that
        actually pays it instead of leaking into whichever admission happens
        next. The previous harness measured evict-then-register and reported
        eviction as nearly free (a ``del``) with the cost resurfacing
        misattributed to the following registration.
        """
        _sync()
        t0 = time.perf_counter()
        registrar.register(new_id, idx)
        _sync()
        register_ms = (time.perf_counter() - t0) * 1000
        with pool_lock:
            pool[slot] = new_id
        return register_ms

    def _evict_now(victim: str) -> float:
        t0 = time.perf_counter()
        registrar.evict(victim)
        _sync()
        return (time.perf_counter() - t0) * 1000

    work: queue.Queue = queue.Queue()
    worker: threading.Thread | None = None
    stop = threading.Event()

    if churn_mode == "background":
        def _drain():
            with torch.cuda.stream(copy_stream):
                while not stop.is_set() or not work.empty():
                    try:
                        slot, victim, new_id, idx = work.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    try:
                        register_ms = _register_only(slot, victim, new_id, idx)
                        retire_q.put(victim)
                        with records_lock:
                            replace_records.append((register_ms, 0.0, register_ms))
                    finally:
                        with pool_lock:
                            pending_slots.discard(slot)
                        work.task_done()

        worker = threading.Thread(target=_drain, daemon=True, name="churn-admit")
        worker.start()

    def _drain_retired() -> float:
        """Free adapters retired by the admitting thread. Returns ms spent."""
        spent = 0.0
        while True:
            try:
                victim = retire_q.get_nowait()
            except queue.Empty:
                break
            spent += _evict_now(victim)
        if spent:
            deferred_evict_ms.append(spent)
        return spent

    def _serve_one() -> float:
        idxs = rng.choice(len(pool), size=batch_size, p=probs)
        with pool_lock:
            batch_ids = [pool[int(i)] for i in idxs]
        coefs = [lr_coefs[a] for a in batch_ids]
        intercepts = [lr_intercepts[a] for a in batch_ids]
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        lora_w, lr_w = assembler.assemble(batch_ids, coefs, intercepts)
        with torch.no_grad():
            model(inputs["input_ids"], inputs["attention_mask"], lora_w, lr_w, output_lr)
        torch.cuda.synchronize(device)
        return (time.perf_counter() - t0) * 1000

    # Warm the pool, the allocator and the clocks, then discard.
    warm_end = time.perf_counter() + warmup_s
    while time.perf_counter() < warm_end:
        _serve_one()

    # Churn walks the slots round-robin so that over a long run every slot is
    # replaced equally often, rather than repeatedly recycling the coldest one.
    next_slot = 0
    debt = 0.0
    start = time.perf_counter()
    last = start
    end = start + duration_s
    while True:
        now = time.perf_counter()
        if now >= end:
            break
        debt += (now - last) * admission_rate
        last = now
        # Owing more than the cap means the target rate exceeds what this arm
        # sustains; the excess is shed rather than queued, and counted.
        if debt > MAX_ADMISSION_BACKLOG:
            shed += int(debt - MAX_ADMISSION_BACKLOG)
            debt = float(MAX_ADMISSION_BACKLOG)
        # The deadline is checked inside the drain loop, not only outside it:
        # when an admission costs more than 1/rate seconds the loop would
        # otherwise never terminate.
        while debt >= 1.0 and time.perf_counter() < end:
            if churn_mode == "background" and work.qsize() >= MAX_ADMISSION_BACKLOG:
                shed += 1
                debt -= 1.0
                continue
            debt -= 1.0
            # Claim the next slot that has no admission already in flight.
            slot = None
            with pool_lock:
                for _ in range(len(pool)):
                    cand = next_slot % len(pool)
                    next_slot += 1
                    if cand not in pending_slots:
                        slot = cand
                        pending_slots.add(cand)
                        victim = pool[cand]
                        break
            if slot is None:
                # Every slot is mid-admission; nothing can be admitted now.
                shed += 1
                continue
            new_id = f"churn_{corpus_idx}"
            idx = corpus_idx
            corpus_idx += 1
            if churn_mode == "blocking":
                t0 = time.perf_counter()
                register_ms = _register_only(slot, victim, new_id, idx)
                evict_ms = _evict_now(victim)
                blocked_ms += (time.perf_counter() - t0) * 1000
                with pool_lock:
                    pending_slots.discard(slot)
                replace_records.append((register_ms, evict_ms, register_ms + evict_ms))
            else:
                work.put((slot, victim, new_id, idx))
            admissions += 1
        # Deferred frees are charged to the serving thread, because that is
        # where they actually happen -- background admission moves the copy off
        # the serving path, not the free.
        blocked_ms += _drain_retired()
        serve_latencies.append(_serve_one())

    wall_s = time.perf_counter() - start

    # Requested vs completed matters only in background mode, where the worker
    # can fall behind the target rate. A backlog at the end of the window is
    # the signal that the requested rate exceeds what this arm can sustain --
    # which is the result, not a defect, so it is reported rather than hidden
    # by waiting for the queue and dividing by a longer wall clock.
    # Snapshot the in-window records *and* their count together. Taking only
    # the count here and reading the arrays after the drain would mix windows:
    # the achieved rate would describe in-window completions while the latency
    # totals and work share silently included registrations that finished
    # afterwards. That is exactly the regime where it matters -- at saturation
    # the backlog is large (at 100/s the drain runs ~3 s past the window).
    with records_lock:
        completed_in_window = len(replace_records)
        in_window_records = list(replace_records)
    drain_s = 0.0
    if churn_mode == "background":
        assert worker is not None
        t_drain = time.perf_counter()
        work.join()
        stop.set()
        worker.join(timeout=60.0)
        _drain_retired()
        drain_s = time.perf_counter() - t_drain

    register_latencies = np.asarray([r[0] for r in in_window_records])
    inline_evict = np.asarray([r[1] for r in in_window_records])
    # In blocking mode eviction is inline and per-admission. In background mode
    # it is deferred and batched onto the serving thread, so it is reported as
    # a per-drain figure and the per-admission replacement cost is the
    # registration plus the mean deferred free -- flagged here rather than
    # papered over, because the two are not the same measurement.
    if churn_mode == "blocking":
        evict_latencies = inline_evict
        replace_latencies = register_latencies + inline_evict
    else:
        evict_latencies = np.asarray(deferred_evict_ms)
        mean_evict = float(np.mean(evict_latencies)) if evict_latencies.size else 0.0
        replace_latencies = register_latencies + mean_evict

    return {
        "wall_s": wall_s,
        "serve_latencies": np.asarray(serve_latencies),
        "replace_latencies": replace_latencies,
        "register_latencies": register_latencies,
        "evict_latencies": evict_latencies,
        "blocked_ms": blocked_ms,
        "admissions_accepted": admissions,
        "admissions_completed": completed_in_window,
        "admissions_shed": shed,
        "drain_s": drain_s,
        "next_corpus_idx": corpus_idx,
        "page_cache_dropped": registrar.page_cache_dropped,
    }


def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else float("nan")


def summarize(cell: dict, batch_size: int, admission_rate: float) -> dict:
    """Turn one cell's raw arrays into the reported quantities.

    The headline is *not* a percentage share -- that is denominator-dependent
    and invites argument about what belongs in the denominator.  It is the
    sustainable admission rate: how many admissions per second fit inside a
    fixed fraction of serving wall-clock.  The share is still reported, because
    it is the commonly requested metric, but it is derived from the same two
    measured quantities rather than being the primary claim.
    """
    serve = cell["serve_latencies"]
    replace = cell["replace_latencies"]
    serve_total_ms = float(np.sum(serve))
    replace_total_ms = float(np.sum(replace))
    per_admission_ms = float(np.mean(replace)) if replace.size else float("nan")

    # Blocking cost is what the serving thread actually loses; in background
    # mode that is (by design) near zero even though the replacement itself
    # still costs per_admission_ms on the copy stream.
    blocked_ms = cell["blocked_ms"]
    denom = serve_total_ms + blocked_ms
    share_pct = 100 * blocked_ms / denom if denom else 0.0

    # The same quantity with registration work counted whether or not it landed
    # on the serving thread. Reporting both closes off the argument about which
    # denominator is the honest one: blocking share is what serving loses,
    # work share is what the GPU spent, and in blocking mode they coincide.
    work_denom = serve_total_ms + replace_total_ms
    work_share_pct = 100 * replace_total_ms / work_denom if work_denom else 0.0

    return {
        # Accepted + shed = the demand the target rate actually generated;
        # "accepted" alone was previously named "requested", which read as
        # total arrivals and understated demand whenever anything was shed.
        "admissions_accepted": cell["admissions_accepted"],
        "admissions_arrived": cell["admissions_accepted"] + cell["admissions_shed"],
        "admissions_completed": cell["admissions_completed"],
        "admissions_shed": cell["admissions_shed"],
        "achieved_admission_rate": round(cell["admissions_completed"] / cell["wall_s"], 3),
        "target_admission_rate": admission_rate,
        # Below 1.0 the arm could not keep up with the target rate: the cell is
        # measuring saturation, and the achieved rate is the sustainable one.
        "rate_attainment": round(
            (cell["admissions_completed"] / cell["wall_s"]) / admission_rate, 3
        ) if admission_rate > 0 else "",
        "backlog_drain_s": round(cell["drain_s"], 2),
        "wall_s": round(cell["wall_s"], 2),
        "batches_served": int(serve.size),
        "serve_p50_ms": round(_pct(serve, 50), 3),
        "serve_p95_ms": round(_pct(serve, 95), 3),
        "serve_p99_ms": round(_pct(serve, 99), 3),
        "serve_mean_ms": round(float(np.mean(serve)), 3) if serve.size else "",
        # Two throughputs, because one of them is a trap. The first divides by
        # mean serving latency alone and so describes throughput *while
        # serving* -- it excludes the intervals the loop spent registering. In
        # blocking mode that is not delivered throughput: at 100 admissions/s
        # it reads ~987 samples/s against 18 actually delivered. The second
        # divides by wall-clock and is what the pod delivers under churn.
        "serve_throughput_while_serving": round(
            batch_size / (float(np.mean(serve)) / 1000), 1
        ) if serve.size else "",
        "delivered_throughput_samples_sec": round(
            serve.size * batch_size / cell["wall_s"], 1
        ) if cell["wall_s"] else "",
        "replace_p50_ms": round(_pct(replace, 50), 3),
        "replace_p95_ms": round(_pct(replace, 95), 3),
        "replace_mean_ms": round(per_admission_ms, 3),
        "replace_total_ms": round(replace_total_ms, 3),
        "serve_total_ms": round(serve_total_ms, 3),
        "register_mean_ms": round(float(np.mean(cell["register_latencies"])), 3)
        if cell["register_latencies"].size else "",
        "evict_mean_ms": round(float(np.mean(cell["evict_latencies"])), 3)
        if cell["evict_latencies"].size else "",
        "serving_blocked_ms": round(blocked_ms, 3),
        "registration_share_pct": round(share_pct, 4),
        "registration_work_share_pct": round(work_share_pct, 4),
        # Derived headline: admissions/s that fit in 1% of serving wall-clock,
        # and the saturation rate (100% of wall-clock spent admitting).
        "sustainable_rate_at_1pct": round(0.01 / (per_admission_ms / 1000), 2)
        if per_admission_ms == per_admission_ms and per_admission_ms > 0 else "",
        "saturation_rate": round(1.0 / (per_admission_ms / 1000), 1)
        if per_admission_ms == per_admission_ms and per_admission_ms > 0 else "",
        "page_cache_dropped": cell["page_cache_dropped"],
    }


def ci95(values: list[float]) -> tuple[float, float]:
    """Mean and 95% half-width across seeds (Student-t, small n)."""
    vals = [v for v in values if v == v]
    if len(vals) < 2:
        return (vals[0] if vals else float("nan")), float("nan")
    mean = statistics.fmean(vals)
    half = statistics.stdev(vals) / len(vals) ** 0.5
    # t_{0.975, n-1} for n = 2..6; beyond that the normal approximation is fine.
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(len(vals) - 1, 1.96)
    return mean, half * tcrit


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="BAAI/bge-m3")
    p.add_argument("--dtype", choices=list(DTYPE_MAP), default="fp16")
    p.add_argument("--engine", choices=["custom", "hf"], default="hf")
    p.add_argument("--target-modules", nargs="+", default=["query", "value"])
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-labels", type=int, default=10)
    p.add_argument(
        "--resident", nargs="+", type=int, default=[1000],
        help="Resident pool sizes. Run at moderate occupancy AND at the "
             "ceiling: allocator pressure near the ceiling is a real finding "
             "but a separate one, and mixing the two is what produced the "
             "non-monotonic cost curve in the previous harness.",
    )
    p.add_argument(
        "--admission-rates", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0],
        help="Target admissions/second -- the x-axis. Driven directly so the "
             "churn rate is an input, not a byproduct of (alpha, tenants, capacity).",
    )
    p.add_argument("--registration-paths", nargs="+", default=["synthetic", "file", "file_pinned"],
                   choices=REGISTRATION_PATHS)
    p.add_argument("--churn-modes", nargs="+", default=["blocking", "background"],
                   choices=CHURN_MODES)
    p.add_argument("--cold", action="store_true",
                   help="Drop each adapter file from the page cache before "
                        "loading it (cold-start bound). Records whether the "
                        "drop was actually supported on this platform.")
    p.add_argument("--zipf-alpha", type=float, default=1.1,
                   help="Popularity skew of the request stream over the "
                        "resident pool. Affects the hit pattern, not the "
                        "admission rate, which is set independently.")
    p.add_argument("--duration", type=float, default=30.0, help="Measured seconds per cell.")
    p.add_argument("--warmup", type=float, default=5.0, help="Discarded seconds per cell.")
    p.add_argument("--corpus-dir", default="adapter_corpus")
    p.add_argument("--corpus-size", type=int, default=2000)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--out", default="benchmarks/results/churn_registration.csv")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU detected; this benchmark requires one.")
        return

    dtype = DTYPE_MAP[args.dtype]
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    config = LoraServingConfig(
        model_name=args.model,
        lora_rank=args.lora_rank,
        batch_size=args.batch_size,
        max_seq_len=args.seq_len,
        target_modules=args.target_modules,
        device=device,
        dtype=dtype,
    )

    needs_files = any(k != "synthetic" for k in args.registration_paths)
    corpus: list[Path] = []
    if needs_files:
        corpus = build_corpus(config, Path(args.corpus_dir), args.corpus_size)

    print(f"  Loading base model ({args.model}, {dtype}, engine={args.engine})...")
    model = (
        HFEncoderWithLora.from_pretrained_serving(config)
        if args.engine == "hf"
        else EncoderWithLora.from_pretrained_serving(config)
    )
    model.eval()

    rows: list[dict] = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = None
    writer = None

    cells = [
        (n, kind, mode, rate, seed)
        for n in args.resident
        for kind in args.registration_paths
        for mode in args.churn_modes
        for rate in args.admission_rates
        for seed in args.seeds
    ]
    print(f"Sweep: {len(cells)} cells "
          f"({args.duration + args.warmup:.0f}s each, "
          f"~{len(cells) * (args.duration + args.warmup) / 60:.0f} min of measurement)\n")

    for i, (n_resident, kind, mode, rate, seed) in enumerate(cells, 1):
        print(f"[{i}/{len(cells)}] resident={n_resident} path={kind} mode={mode} "
              f"rate={rate}/s seed={seed}")
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        store = AdapterStore(config)
        t0 = time.perf_counter()
        # The initial resident population only has to establish occupancy and
        # allocator state, so it is filled synthetically even in the file arms:
        # writing 47,000 real files would cost 74 GB of disk to no measurement
        # benefit. Only the *replacements* are timed, and those come from the
        # corpus.
        make_synthetic_adapters(store, n_resident)
        fill_s = time.perf_counter() - t0
        resident = [f"adapter_{j}" for j in range(n_resident)]
        lr_coefs = {}
        lr_intercepts = {}
        for aid in resident:
            coef, intercept = make_synthetic_lr_weights(config, args.num_labels)
            lr_coefs[aid] = coef
            lr_intercepts[aid] = intercept

        assembler = BatchAssembler(store, config)
        inputs = make_synthetic_inputs(config, args.batch_size)
        output_lr = torch.zeros(
            args.batch_size, 1, args.num_labels, dtype=dtype, device=device
        )
        registrar = Registrar(
            store, config, corpus, kind, args.num_labels,
            lr_coefs, lr_intercepts, cold=args.cold,
        )

        torch.cuda.reset_peak_memory_stats(device)
        try:
            cell = run_cell(
                model=model, config=config, assembler=assembler,
                registrar=registrar, resident=resident, lr_coefs=lr_coefs,
                lr_intercepts=lr_intercepts, inputs=inputs, output_lr=output_lr,
                batch_size=args.batch_size, duration_s=args.duration,
                admission_rate=rate, churn_mode=mode, zipf_alpha=args.zipf_alpha,
                rng=rng, warmup_s=args.warmup, next_corpus_idx=0,
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM: {e}")
            del store
            torch.cuda.empty_cache()
            continue

        row = {
            "model": args.model,
            "dtype": str(dtype).replace("torch.", ""),
            "engine": args.engine,
            "resident": n_resident,
            "batch_size": args.batch_size,
            "lora_rank": args.lora_rank,
            "seq_len": args.seq_len,
            "registration_path": kind,
            "churn_mode": mode,
            "cold": args.cold,
            "zipf_alpha": args.zipf_alpha,
            "seed": seed,
            "store_fill_s": round(fill_s, 2),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3),
            "adapter_cache_gb": round(store.memory_gb(), 3),
        }
        row.update(summarize(cell, args.batch_size, rate))
        row["alloc_conf"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default")
        rows.append(row)

        if writer is None:
            csv_file = open(out_path, "w", newline="")
            writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
            writer.writeheader()
        writer.writerow(row)
        csv_file.flush()
        os.fsync(csv_file.fileno())

        print(f"  replace={row['replace_mean_ms']}ms/admission "
              f"(register={row['register_mean_ms']}, evict={row['evict_mean_ms']})  "
              f"serve p50={row['serve_p50_ms']}ms p99={row['serve_p99_ms']}ms  "
              f"blocked_share={row['registration_share_pct']}%  "
              f"sustainable@1%={row['sustainable_rate_at_1pct']}/s  "
              f"achieved={row['achieved_admission_rate']}/s\n")

        del store, assembler, registrar
        torch.cuda.empty_cache()

    if csv_file is not None:
        csv_file.close()
    if not rows:
        print("No cells completed.")
        return

    # Cross-seed rollup: the per-admission cost is the quantity every derived
    # claim rests on, so it is the one reported with an interval.
    print("\n=== Per-admission cost (mean +/- 95% CI across seeds) ===")
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        key = (r["resident"], r["registration_path"], r["churn_mode"], r["target_admission_rate"])
        groups.setdefault(key, []).append(r["replace_mean_ms"])
    for key in sorted(groups):
        mean, half = ci95(groups[key])
        n_res, kind, mode, rate = key
        sustain = 0.01 / (mean / 1000) if mean and mean == mean else float("nan")
        print(f"  resident={n_res:<6} {kind:<12} {mode:<11} rate={rate:<6} "
              f"{mean:7.3f} +/- {half:5.3f} ms  ->  {sustain:8.1f} admissions/s at 1% budget")

    print(f"\nSaved to {out_path}  ({len(rows)}/{len(cells)} cells)")


if __name__ == "__main__":
    main()
