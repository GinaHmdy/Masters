# Continuing the experiments on the dual-RTX-3090 machine

Everything except the Gowalla work is finished. What remains is **21 PPD v6 training runs plus 2
LightGCN runs**, all on Gowalla — the one dataset that is expensive here (~1.05 h/run on an RTX
3060 Laptop, ≈23 h total).

## Is the move worth it?

Yes, but the honest figure is **3–5 hours, not two.**

| | RTX 3060 Laptop | 2× RTX 3090 |
|---|---|---|
| memory bandwidth | ~336 GB/s | 936 GB/s each |
| per Gowalla v6 run | ~63 min | ~27 min (est.) |
| total GPU-work | ~23 h | ~9.4 GPU-h |
| wall clock | ~23 h | **~4.7 h** at 1 worker/GPU, **~3 h** at 2 workers/GPU |

The workload is sparse-matmul bound, so it scales with memory bandwidth (2.8×) rather than
FLOPs. I have assumed ~2.35× to leave room for CPU-side graph construction, which does not speed
up. The 2-workers-per-GPU figure assumes a single run does not saturate a 3090 — likely, since a
single run only reached 82% utilisation on the much smaller 3060, but **measure it rather than
trusting it** (see step 5).

24 GB per GPU is far more than the ~3 GB a run uses, so memory is not a constraint there — unlike
here, where two concurrent runs would not fit.

---

## Step 1 — Commit first (do not skip)

There are ~288 modified/untracked files and **nothing has been committed this session**. Moving
uncommitted work across machines is how days of it disappear.

```bash
cd ~/98-Downloads/Masters
git add -A
git commit -m "Phase 1-2 complete; eval cache, dedup, portability for multi-GPU migration"
```

If you would rather not commit the 3.3 GB of checkpoints, add them to `.gitignore` first — the
results JSONs, not the weights, are the evidence.

## Step 2 — Build the bundle (~400 MB)

```bash
./make_migration_bundle.sh ~/thesis_migration.tar.zst
```

It carries the code, `data/`, the complete `results/` tree (every metric, per-user dump and
significance test), the feature cache, and the two already-trained Gowalla `a_full` checkpoints
so those runs are not repeated. It deliberately leaves the other ~3 GB of weights behind: every
completed result already lives in `results/*.json`, and no remaining run reads them.

Verify the checksum it prints after transferring.

## Step 3 — Set up on the 3090 machine

```bash
mkdir -p ~/Masters && tar -xf thesis_migration.tar.zst -C ~/Masters
cd ~/Masters/LightGCN/code

conda create -n gMasters python=3.12 -y && conda activate gMasters
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match the local CUDA
pip install -r requirements-frozen.txt                                  # everything else
```

Reference versions from this machine: `python 3.12.13`, `torch 2.12.0+cu130`, `numpy 2.4.5`,
`scipy 1.17.1`, `scikit-learn 1.8.0`, `lightgbm 4.6.0`. Torch must match the 3090 box's CUDA;
the rest should match exactly.

## Step 4 — Sanity-check before spending GPU hours

```bash
export THESIS_PYTHON=$(which python)
python -c "import torch;print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
python check_sanity_floors.py       # must pass on the transferred results
python make_numbers_pack.py         # must regenerate without error
```

`check_sanity_floors.py` exits non-zero if any model has fallen below the Random floor, which
would mean the transfer corrupted something. It should print the same table you saw here.

## Step 5 — Measure one run before committing to a layout

```bash
time THESIS_CKPT_TAG=probe CUDA_VISIBLE_DEVICES=0 \
  python run_phase2_ablation.py Gowalla --arms b_no_spectral --seeds 2024 --out-suffix _probe
```

Watch `nvidia-smi` during it. If a single run sits below ~60% utilisation, use
`WORKERS_PER_GPU=2`; if it saturates, use 1. Then merge or discard the probe shard:

```bash
python merge_shards.py phase2_ablation_gowalla     # keeps the probe result
```

## Step 6 — Run it

```bash
WORKERS_PER_GPU=2 ./run_parallel_gowalla.sh        # or omit for 1 worker/GPU
```

This runs the seed extension, then the six ablation arms in parallel (one per worker), merges the
shards, runs the gate transfer, and regenerates every derived artifact.

**Two safeguards make the parallelism safe, and both matter:**

1. `THESIS_CKPT_TAG` gives each worker a private default checkpoint. Without it every concurrent
   run writes the same `ppd_v6-Gowalla-3-64.pth.tar`, and the driver copies whichever weights
   landed there last — silent cross-arm contamination that the mtime guard cannot catch, because
   the clobbered file is *newer*, not older.
2. Each worker writes its own `_shard<N>.json`; `merge_shards.py` recombines them and **fails
   loudly** if two shards disagree about an arm rather than silently taking one.

## Step 7 — Bring the results back

Only `results/` needs to return:

```bash
tar -czf results_back.tar.gz LightGCN/code/results
```

Then on this machine, extract and regenerate:
`python make_canonical.py && python make_readme_tables.py && python make_numbers_pack.py`

---

## What is left after this

With Gowalla done, every open experimental question closes:

- **Finding Y** — does Gowalla keep its gates under Yahoo's smaller initialization? The
  `yahoo_init` transfer arm decides it. This is the one that determines whether "the mechanisms
  adapt to the data" is a result or a hypothesis.
- **Finding AA** — does Gowalla's arm (f) show the Yahoo pattern (sampler owns AvgPop, mechanisms
  own APLT)? This decides whether the contribution's fairness claim rests on the named mechanisms.
- **Item 5.11** — Gowalla at 5 seeds decides whether the headline Recall@20 margin clears
  across-seed significance (currently p=0.056 at n=3).

Nothing else is outstanding. Phases 1 and 2 are complete on Coat, Yahoo and KuaiRec.

## Note on reproducibility

Gowalla PPD v6 training is **not** bit-reproducible even on identical hardware (audit finding
**AB**: ~2–6% of the seed std, from non-deterministic CUDA reductions). Different GPUs will
differ by at least that much. This does not invalidate anything — the seeds are draws either way
— but do not expect the 3090 numbers to reproduce the 3060 numbers exactly, and do not report a
difference smaller than ~6e-05 on Recall@20 as an effect.
