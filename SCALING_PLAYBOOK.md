# Multi-GPU scaling playbook

How to measure training throughput honestly, and a prioritized list of things
to try when scaling disappoints. Companion to `DDP_MULTI_GPU.md`, which covers
correctness; this file is only about speed.

## 0. Measure correctly first — most "results" here are measurement artifacts

`torch.compile` spends **1–2 minutes** building graphs before the first step
(`PyTorchCompileMode` in the opt config). A 2M-position smoke at 12k pos/s runs
for ~170 s, so **startup is the majority of the wall time** and any comparison
between settings is really comparing compile-time variance.

Sample a *running* job instead — immune to startup, takes two minutes:

```bash
scripts/server/throughput.sh <OUTPUTS>/logs/<ID>_launch.log 120
```

If you must use fixed-length smokes, make them **≥ 10M positions** so compile
is a few percent rather than half.

## 1. Diagnose before optimizing: comm-bound or data-bound?

```bash
nvidia-smi --query-gpu=index,utilization.gpu --format=csv -l 5
```

| GPU utilization | Bottleneck | Where to look |
|---|---|---|
| ~90 %+ | Compute — already near the ceiling | Section 4 (compile mode), or accept it |
| 50–80 %, all ranks | **Data loading** (zstd decompress + TPG parse) | Section 3 |
| Low, and worse as ranks are added | **Communication** | Section 2 |

The signature of a communication problem is unmistakable: throughput *per card*
falls as cards are added. Measured on a 4×A100 host with two NVLink pairs
bridged by PCIe:

| Ranks | Total | Per card |
|---|---|---|
| 1 | 6 600 /s | 6 600 |
| 2 (one NVLink pair) | ~12 000 /s | 6 000 |
| 4 (spanning both pairs) | **2 800 /s** | **700** |

Four cards were ~9× slower *per card* than one. The all-reduce crossing the
PCIe bridge dominated everything else.

Confirm the topology, and what NCCL actually chose:

```bash
nvidia-smi topo -m                       # NV# = NVLink, SYS/PHB = across the bridge
NCCL_DEBUG=INFO torchrun ... 2>&1 | grep -E "via|Channel|Ring|Tree|NVL"
```

## 2. Communication-bound: prioritized experiments

Change **one thing at a time** and re-measure with `throughput.sh`.

| # | Change | How | Cost | Changes results? |
|---|---|---|---|---|
| 1 | Drop the unused-parameter scan | `CERES_DDP_FIND_UNUSED=0` | free | no |
| 2 | Tree collective instead of ring | `NCCL_ALGO=Tree` | free | no |
| 3 | bf16 gradient compression | `"DDPBF16Compress": true` | free | negligible¹ |
| 4 | Larger gradient buckets | `"DDPBucketCapMB": 100` (try 200) | free | no |
| 5 | Fewer optimizer steps per position | raise `BatchSizeBackwardPass` to 8192/16384 | needs LR retune | **YES** |

¹ Gradients are produced under bf16 autocast, so the fp32 `.grad` buffers
already hold values that passed through bf16 matmuls. Sending them as bf16
loses little that was not already lost.

**Item 5 is the only one that changes the recipe.** It halves or quarters the
all-reduces per position, because communication volume is set by the
optimization batch — but it makes results incomparable with everything measured
at 4096, and the LR must be scaled to match.

**What does NOT help: gradient accumulation.** With the `no_sync` path active
the all-reduce already fires once per *optimizer* step, so splitting that step
into more micro-steps leaves communication per position unchanged. (It does
matter under `static_graph`, where `no_sync` is disabled — see
`DDP_MULTI_GPU.md`.)

### If none of items 1–4 change anything: check NUMA

Measured on the 4×A100 host, items 1–4 were tried and none moved the number.
That rules out the collective's *execution* and points at process placement.

On a dual-socket box the GPUs are split across sockets — typically 0,1 on
socket 0 and 2,3 on socket 1. An unbound rank can be scheduled on the socket
that does **not** own its GPU, so its dataloader parses into remote memory and
NCCL stages host buffers on the wrong side. Every batch then crosses the
inter-socket link twice. Two ranks on one socket never expose it; four ranks
spanning both do — exactly the "2 GPUs fine, 4 GPUs collapse" signature. It also
explains more workers making things *worse*: unbound workers scatter across both
sockets and add cross-socket traffic rather than parallelism.

No NCCL variable can fix this, because the misplacement happens at allocation
time in both the dataloader and the collective's host buffers.

```bash
nvidia-smi topo -m                     # CPU Affinity / NUMA Affinity columns
lscpu | grep -E "Socket|NUMA"
```

Different NUMA nodes for 0,1 versus 2,3 confirms it. `launch_ddp.sh` then binds
each rank to its own GPU's node automatically via `scripts/server/numa_wrap.sh`;
`CERES_NUMA_BIND=0` disables it so the effect can be measured both ways. Look
for the `[numa] rank N -> gpu N -> NUMA node M` lines at startup.

Note this is placement, not core count: the same host had 40+ cores idle, so
CPU starvation was never the issue.

### Stopping rule

If items 1–4 do not lift a 4-rank job above roughly 10k pos/s, stop optimizing.
Two independent 2-GPU jobs — one per NVLink island — delivered **~24k pos/s
aggregate against 2.8k** for a single 4-GPU job on the same hardware. Reach for
4-rank only when a *single* run must span all cards (e.g. one 2B run). For A/B
experiments the split is strictly better anyway, since the arms must not share
gradients:

```bash
scripts/server/launch_tactical_ab.sh /path/to/OUTPUTS 2   # one arm per NVLink pair
```

## 3. Data-bound: prioritized experiments

The dataloader does zstd decompression and TPG record parsing on CPU, and has
historically been the ceiling on this corpus (~59 % GPU utilization was measured
single-GPU on an earlier run).

| # | Change | How | Note |
|---|---|---|---|
| 1 | More dataloader workers | `CERES_NUM_DATASET_WORKERS=2`, then 4 | **The shard requirement scales with it**: every corpus needs `shards ≥ ranks × workers`, enforced by `preflight.sh` |
| 2 | Faster storage | corpus on local NVMe | all ranks read the same disk |
| 3 | Larger forward batch | `BatchSizeForwardPass` 1024/2048/4096 | fewer, larger kernels; does not reduce the bytes to parse, so it cannot fix a data bound |

**Measured and exhausted on the 4×A100 host (2026-08-18): worker counts 1, 2, 4
and 8 all landed within noise of each other, with 2 marginally best.** A flat
response to worker count means the loader is *not* the binding constraint — a
real supply limit improves when you add parsers. Combined with 6.6k pos/s on a
single card and 40+ idle cores, single-GPU throughput here is compute-bound, and
worker tuning is a dead end on this machine. Do not re-derive this; measure
utilization (section 1) before touching section 3 at all.

## 4. Compute-bound: compile mode

`PyTorchCompileMode` goes straight to `torch.compile(mode=...)` with
`dynamic=False`.

| Mode | When |
|---|---|
| `default` | fast to compile, no kernel autotuning |
| `max-autotune-no-cudagraphs` | **recommended under DDP** — benchmarks Triton configs for the real geometry, without CUDA-graph fragility |
| `max-autotune` | same autotuning plus CUDA graphs |
| `reduce-overhead` | CUDA-graph based; fragile with DDP and with this loop's training-only attribute stashes, and aimed at small batches. Avoid here. |

Autotuning costs minutes of extra compile — irrelevant for a 300M+ run,
dominant in a short smoke. `PyTorchCompileMode` must be `null` if you ever use
the gradient-conflict probe.

## 5. Record what you measure

| Date | Config | Ranks | Workers | Batch f/b | Change under test | pos/s | Notes |
|---|---|---|---|---|---|---|---|
| 2026-08-18 | cfT80_300M | 1 (4090) | 1 | 512/4096 | baseline | 5 046 | desktop reference |
| 2026-08-18 | — | 1 (A100) | 2 | 4096/4096 | baseline | 6 600 | |
| 2026-08-18 | — | 2 (A100, NVLink pair) | 2 | 4096/4096 | baseline | ~12 000 | 1.82× |
| 2026-08-18 | — | 4 (A100, both pairs) | 2 | 4096/4096 | baseline | 2 800 | comm/placement-bound |
| 2026-08-18 | — | 4 | 1 / 2 / 4 / 8 | 4096/4096 | worker count | ~2 800 | flat — loader is not the constraint |
| 2026-08-18 | — | 4 | 2 | 4096/4096 | FIND_UNUSED=0 | no change | |
| 2026-08-18 | — | 4 | 2 | 4096/4096 | NCCL_ALGO=Tree | no change | |
| 2026-08-18 | — | 4 | 2 | 4096/4096 | bf16 compress + buckets | no change | |
| — | — | 4 | 2 | 4096/4096 | **NUMA binding** | *pending* | the one lever not yet tried |
| reference | lc0 (kovax) | 4 | C++ loader | 1024/GPU | — | ~12 000 | same per-GPU batch |
