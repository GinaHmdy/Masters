"""
Validation-only re-tuning of ASTD-PD (two-stage search), dataset-generic.

    python -m src.astdpd.tune --dataset Coat                      # run from thesis/
    python -m src.astdpd.tune --dataset Gowalla --n-random 40 --top-k 4 --workers 2

Selection rule (fixed in advance; test metrics are never read):
  threshold T = for each seed, the lowest VALIDATION AvgPop@20 among the four baselines
  (lightgcn.json and the validation-selected ppd/tsp/dap *_selected.json of that seed),
  averaged over ALL benchmark seeds (2024-2028; "thr5"). The mean over the confirmation seeds
  only ("thr3") is logged as information but not used. Among ASTD-PD configurations, choose the
  one with the highest mean validation NDCG@20 subject to mean validation AvgPop@20 < T. If no
  configuration is feasible, choose the lowest mean validation AvgPop@20 and record that the
  constraint was infeasible.

Search:
  stage 1  the shipped configuration crossed with the inference-penalty grid, plus `n_random`
           random configurations, each trained once on the first seed; ranked by the rule with
           that seed's threshold; the `top_k` best go on (fill: lowest AvgPop if < top_k feasible);
  stage 2  the top-k are trained on the first `confirm_seeds` seeds; the rule is applied to their
           means over those seeds with the matching mean threshold -> the winner;
  final    the winner is also trained on the remaining seeds (into the trial directory); then,
           with --wait only after `run.py all` has finished its shipped-config ASTD-PD runs of the
           dataset (else `run.py astdpd` would pick up the new config), the winner is written to configs/astdpd_selected.yaml (which src.astdpd.train
           .load_config then prefers), the shipped-config outputs of every seed are kept as
           astdpd_shipped.{json,pt,_per_user_test.npz}, and runs/<ds>/seed<s>/astdpd.* are
           produced with the winner (stage-2 runs of the winner are reused; identical call).

Every trial is a full src.astdpd.train.run (same training, early stopping on the validation
hypervolume, same evaluator), so a trial's validation numbers are exactly what the final run
selects with. The log runs/<ds>/astdpd_tune.json holds every configuration with its
VALIDATION metrics only; trial output directories are deleted at the end.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
SELECTED = ROOT / "configs" / "astdpd_selected.yaml"
SEEDS = [2024, 2025, 2026, 2027, 2028]
BASELINE_FILES = {"LightGCN": "lightgcn.json", "PPD": "ppd_selected.json",
                  "TSP": "tsp_selected.json", "DAP": "dap_selected.json"}
VALID_KEYS = ("recall@20", "ndcg@20", "precision@20", "avgpop@20", "aplt@20", "coverage@20",
              "recall@10", "ndcg@10", "avgpop@10")

# Knobs that move popularity. mu (inference penalty) is in score units at temperature 0.2 and
# is rescaled by 0.2 / temperature_acc, since the scores scale with 1 / temperature.
MU_GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]
SPACE = dict(
    inference_pop_penalty=MU_GRID,
    sampler=[(0.5, 0.4, 0.1), (1.0, 0.0, 0.0), (0.6, 0.3, 0.1), (0.4, 0.4, 0.2), (0.3, 0.5, 0.2),
             (0.2, 0.6, 0.2), (0.2, 0.5, 0.3), (0.4, 0.3, 0.3), (0.1, 0.6, 0.3)],
    tail_phi=[0.5, 1.0, 1.5, 2.0, 3.0],
    lambda_deb=[0.1, 0.3, 0.5, 1.0],
    fuse_lambda_min=[0.0, 0.1, 0.2],
    fuse_lambda_max=[0.3, 0.5, 0.8, 1.0],
    alpha_base=[0.05, 0.1, 0.2, 0.4, 0.6],
    topo_pop_damping=[0.05, 0.1, 0.3, 0.5, 1.0],
    lambda_ind=[0.05, 0.1, 0.3],
    highfreq_gamma=[0.0, 0.1, 0.3],
)


# ------------------------------------------------------------------------------ configs
def shipped_config(ds):
    from src.astdpd.train import load_config
    return load_config(ds, selected=False)


def make_config(base, ov):
    """Full config = shipped config with the overrides `ov` applied (model keys / sampler)."""
    cfg = json.loads(json.dumps(base))
    for k, v in ov.items():
        if k == "sampler":
            cfg["sampler"] = dict(uniform_ratio=float(v[0]), matched_ratio=float(v[1]), hard_ratio=float(v[2]))
        else:
            cfg["model"][k] = float(v)
    lo, hi = cfg["model"].get("fuse_lambda_min", 0.0), cfg["model"].get("fuse_lambda_max", 0.5)
    if lo > hi:
        cfg["model"]["fuse_lambda_min"] = hi
    return cfg


def config_id(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def stage1_overrides(base, n_random, seed, mu_grid=None):
    scale = 0.2 / max(1e-4, base["model"].get("temperature_acc", 0.1))
    out = [dict(inference_pop_penalty=round(mu * scale, 4))            # incumbent x mu
           for mu in (MU_GRID if mu_grid is None else mu_grid)]
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        ov = {}
        for k, choices in SPACE.items():
            v = choices[rng.integers(len(choices))]
            ov[k] = round(v * scale, 4) if k == "inference_pop_penalty" else v
        out.append(ov)
    return out


# ------------------------------------------------------------------------------ threshold
def baseline_valid_avgpop(ds, seed):
    """{baseline: validation AvgPop@20} of the seed's selected baseline runs (None if missing/stale)."""
    d = RUNS / ds / f"seed{seed}"
    if not all((d / f).exists() for f in BASELINE_FILES.values()):
        return None
    bb = json.load(open(d / "lightgcn.json"))
    out = {}
    for name, f in BASELINE_FILES.items():
        r = json.load(open(d / f))
        if name != "LightGCN" and r.get("backbone_id") != bb.get("backbone_id"):
            return None
        out[name] = float(r["valid"]["avgpop@20"])
    return out


def threshold(ds, seeds):
    per = {s: baseline_valid_avgpop(ds, s) for s in seeds}
    missing = [s for s, v in per.items() if v is None]
    if missing:
        raise SystemExit(f"{ds}: baseline validation results missing/stale for seeds {missing}")
    mins = {s: min(v.values()) for s, v in per.items()}
    return float(np.mean(list(mins.values()))), dict(
        seeds=list(seeds), per_seed_min=mins,
        per_seed_argmin={s: min(v, key=v.get) for s, v in per.items()}, per_seed_baselines=per)


def select(cands, thr):
    """cands: [(id, ndcg, avgpop)]. Rule above -> (ranked ids, feasible flag of the first)."""
    feas = sorted([c for c in cands if c[2] < thr], key=lambda c: -c[1])
    rest = sorted([c for c in cands if c[2] >= thr], key=lambda c: c[2])
    return [c[0] for c in feas + rest], bool(feas)


# ------------------------------------------------------------------------------ workers
def _init(gpus, threads, counter):
    with counter.get_lock():   # round-robin the pool's workers over `gpus`
        i = counter.value
        counter.value += 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[i % len(gpus)])
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(threads)
    import torch
    torch.set_num_threads(threads)


def _trial(ds, seed, cfg, out_dir):
    import sys
    sys.path.insert(0, str(ROOT))
    from src.astdpd import train as A
    t0 = time.time()
    r = A.run(ds, seed, out_dir=out_dir, cfg=cfg, verbose=False, device="cuda")
    # validation only leaves this function: selection can never see test metrics
    return dict(valid={k: r["valid"][k] for k in VALID_KEYS if k in r["valid"]},
                best_epoch=r["best_epoch"], epochs_run=r["epochs_run"],
                best_valid_hypervolume=r["best_valid_hypervolume"], wall_s=time.time() - t0)


class Tuner:
    def __init__(self, ds, workers, gpus, threads):
        self.ds, self.workers, self.gpus, self.threads = ds, workers, list(gpus), threads
        self.log_path = RUNS / ds / "astdpd_tune.json"
        self.trial_dir = RUNS / ds / "astdpd_tune_runs"
        self.log = json.load(open(self.log_path)) if self.log_path.exists() else dict(configs={}, trials=[])

    def save(self):
        tmp = self.log_path.with_suffix(".tmp")
        json.dump(self.log, open(tmp, "w"), indent=1)
        tmp.replace(self.log_path)

    def result(self, cid, seed):
        for t in self.log["trials"]:
            if t["id"] == cid and t["seed"] == seed:
                return t
        return None

    def pool(self, n):
        ctx = get_context("spawn")
        return ProcessPoolExecutor(n, mp_context=ctx, initializer=_init,
                                   initargs=(self.gpus, self.threads, ctx.Value("i", 0)))

    def run_trials(self, todo, stage):
        """todo: [(cid, seed)]; trains the missing ones in a GPU process pool."""
        todo = [(c, s) for c, s in todo if self.result(c, s) is None]
        if not todo:
            return
        print(f"[{self.ds}] {stage}: {len(todo)} trials on GPUs {self.gpus}, {self.workers} workers", flush=True)
        t0, done = time.time(), 0
        with self.pool(self.workers) as ex:
            futs = {ex.submit(_trial, self.ds, s, self.log["configs"][c]["config"],
                              str(self.trial_dir / c / f"seed{s}")): (c, s) for c, s in todo}
            for f in as_completed(futs):
                c, s = futs[f]
                try:
                    r = f.result()
                except Exception as e:  # keep going; a failed config is logged and never selected
                    print(f"  FAILED {c} s{s}: {e!r}", flush=True)
                    continue
                self.log["trials"].append(dict(id=c, seed=s, stage=stage, **r))
                self.save()
                done += 1
                v = r["valid"]
                print(f"  [{done}/{len(todo)} {time.time() - t0:.0f}s] {c} s{s} valid N@20 {v['ndcg@20']:.4f} "
                      f"R@20 {v['recall@20']:.4f} AvgPop@20 {v['avgpop@20']:.2f}", flush=True)

    def mean_valid(self, cid, seeds):
        rs = [self.result(cid, s) for s in seeds]
        if any(r is None for r in rs):
            return None
        return {k: float(np.mean([r["valid"][k] for r in rs])) for k in VALID_KEYS if k in rs[0]["valid"]}


# ------------------------------------------------------------------------------ main flow
RULE = ("max mean valid NDCG@20 (over the confirmation seeds) s.t. mean valid AvgPop@20 < thr5 = mean "
        "over all 5 seeds of the per-seed lowest baseline valid AvgPop@20; else min mean valid AvgPop@20")


def stage2_select(T, means, conf, thr_all, thr_conf):
    """Stage-2 rule against thr5 (thr_all); thr3 (thr_conf) is recorded for information only."""
    ranked, feasible = select([(c, m["ndcg@20"], m["avgpop@20"]) for c, m in means.items()], thr_all)
    return ranked[0], dict(seeds=conf, threshold=thr_all, threshold_kind="thr5 (mean over all 5 seeds)",
                           threshold_confirm_seeds_info=thr_conf, feasible=feasible,
                           candidates={c: means[c] for c in ranked})


def reselect(ds):
    """Re-apply the stage-2 rule to an existing tune log (no training). Returns the winner."""
    T = Tuner(ds, 1, [0], 1)
    sel = T.log["selection"]
    top, conf = sel["stage1"]["top_k"], sel["stage2"]["seeds"]
    thr_all, thr_meta = threshold(ds, SEEDS)
    thr_conf = float(np.mean([thr_meta["per_seed_min"][s] for s in conf]))
    means = {c: T.mean_valid(c, conf) for c in top}
    means = {c: m for c, m in means.items() if m}
    old = sel["winner"]
    win, st2 = stage2_select(T, means, conf, thr_all, thr_conf)
    sel.update(rule=RULE, stage2=st2, threshold_all_seeds=thr_all, baselines=thr_meta, winner=win,
               winner_overrides=T.log["configs"][win]["overrides"], winner_valid_mean=means[win],
               reselected=dict(at=time.strftime("%Y-%m-%d %H:%M:%S"), previous_winner=old))
    T.save()
    print(f"[{ds}] thr5 {thr_all:.2f} (thr3 {thr_conf:.2f}, info): winner {win} "
          f"({'feasible' if st2['feasible'] else 'INFEASIBLE'}), previous {old}", flush=True)
    for c in st2["candidates"]:
        m = means[c]
        print(f"   {c} valid N@20 {m['ndcg@20']:.4f} AvgPop@20 {m['avgpop@20']:.2f}"
              f"{' <' if m['avgpop@20'] < thr_all else ''}", flush=True)
    return win


def tune(ds, n_random=160, top_k=8, confirm_seeds=3, workers=4, gpus=None, threads=2, rng_seed=0,
         final=True, wait=False, mu_grid=None):
    gpus = [pick_gpu()] if gpus is None else gpus
    T = Tuner(ds, workers, gpus, threads)
    base = shipped_config(ds)
    if wait:
        wait_baselines(ds)
    s1, conf = SEEDS[0], SEEDS[:confirm_seeds]
    thr_all, thr_meta = threshold(ds, SEEDS)
    thr_s1 = thr_meta["per_seed_min"][s1]
    thr_conf = float(np.mean([thr_meta["per_seed_min"][s] for s in conf]))
    print(f"[{ds}] baseline min validation AvgPop@20: seed {s1} {thr_s1:.2f}, "
          f"seeds {conf} {thr_conf:.2f}, all seeds {thr_all:.2f}", flush=True)

    # stage 1
    ids = []
    for ov in stage1_overrides(base, n_random, rng_seed, mu_grid):
        cfg = make_config(base, ov)
        cid = config_id(cfg)
        if cid not in ids:
            ids.append(cid)
            T.log["configs"].setdefault(cid, dict(overrides={k: list(v) if isinstance(v, tuple) else v
                                                             for k, v in ov.items()}, config=cfg))
    T.save()
    T.run_trials([(c, s1) for c in ids], "stage1")
    ok = [c for c in ids if T.result(c, s1)]
    ranked, _ = select([(c, T.result(c, s1)["valid"]["ndcg@20"], T.result(c, s1)["valid"]["avgpop@20"])
                        for c in ok], thr_s1)
    top = ranked[:top_k]
    for c in ok:   # free disk: only the stage-2 candidates' outputs can be reused later
        if c not in top:
            shutil.rmtree(T.trial_dir / c, ignore_errors=True)

    # stage 2
    T.run_trials([(c, s) for c in top for s in conf], "stage2")
    means = {c: T.mean_valid(c, conf) for c in top}
    means = {c: m for c, m in means.items() if m}
    win, st2 = stage2_select(T, means, conf, thr_all, thr_conf)
    T.log["selection"] = dict(
        rule=RULE,
        stage1=dict(seed=s1, threshold=thr_s1, n_configs=len(ok), top_k=top),
        stage2=st2, threshold_all_seeds=thr_all, baselines=thr_meta,
        winner=win, winner_overrides=T.log["configs"][win]["overrides"], winner_valid_mean=means[win],
        search_space={k: [list(x) if isinstance(x, tuple) else x for x in v] for k, v in SPACE.items()},
        budget=dict(n_random=n_random, incumbent_mu_grid=MU_GRID if mu_grid is None else mu_grid, top_k=top_k,
                    confirm_seeds=conf, n_trials=len(T.log["trials"])),
        generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    T.save()
    for c in top:   # free disk: only the winner's outputs are reused
        if c != win:
            shutil.rmtree(T.trial_dir / c, ignore_errors=True)
    feasible = st2["feasible"]
    print(f"[{ds}] winner {win} ({'feasible' if feasible else 'INFEASIBLE: lowest AvgPop'}) "
          f"overrides {T.log['configs'][win]['overrides']} valid mean {means[win]}", flush=True)
    # the winner on the remaining seeds (into the trial directory; finalize() only copies files)
    T.run_trials([(win, s) for s in SEEDS if s not in conf], "final")
    T.log["selection"]["budget"]["n_trials"] = len(T.log["trials"])
    T.save()
    if final:
        if wait:
            wait_shipped(T)
        finalize(T, win)
    return win


def finalize_only(ds, workers=2, gpus=None, threads=2, wait=False):
    """Final step of an earlier `--no-final` search: the winner recorded in the tune log."""
    T = Tuner(ds, workers, [pick_gpu()] if gpus is None else gpus, threads)
    win = T.log["selection"]["winner"]
    if wait:
        wait_shipped(T)
    finalize(T, win)


def finalize(T, win):
    ds, cfg = T.ds, T.log["configs"][win]["config"]
    sel = (yaml.safe_load(open(SELECTED)) or {}) if SELECTED.exists() else {}
    sel[ds] = dict(config=cfg, tune_log=f"runs/{ds}/astdpd_tune.json", config_id=win,
                   overrides=T.log["configs"][win]["overrides"],
                   feasible=T.log["selection"]["stage2"]["feasible"],
                   valid_mean_confirm_seeds=T.log["selection"]["winner_valid_mean"])
    head = ("# Validation-retuned ASTD-PD configurations, written by `python -m src.astdpd.tune`.\n"
            "# src/astdpd/train.py:load_config prefers <dataset>.config over configs/astdpd.yaml.\n"
            "# Selection used validation metrics only; see runs/<dataset>/astdpd_tune.json.\n")
    SELECTED.write_text(head + yaml.safe_dump(sel, sort_keys=False, default_flow_style=None))

    stems = ("astdpd.json", "astdpd.pt", "astdpd_per_user_test.npz")
    shipped = shipped_config(ds)
    for s in SEEDS:   # keep the shipped-config results under astdpd_shipped.*
        d = RUNS / ds / f"seed{s}"
        f = d / "astdpd.json"
        if f.exists() and not (d / "astdpd_shipped.json").exists() and json.load(open(f)).get("config") == shipped:
            for n in stems:
                if (d / n).exists():
                    (d / n).rename(d / n.replace("astdpd", "astdpd_shipped"))
    todo = []
    for s in SEEDS:
        d, src = RUNS / ds / f"seed{s}", T.trial_dir / win / f"seed{s}"
        if (d / "astdpd.json").exists() and json.load(open(d / "astdpd.json")).get("config") == cfg:
            continue
        if (src / "astdpd.json").exists():   # identical call already made in stage 1/2
            for n in stems:
                shutil.copy2(src / n, d / n)
        else:
            todo.append(s)
    if todo:
        print(f"[{ds}] final runs of {win} for seeds {todo}", flush=True)
        with T.pool(min(T.workers, len(todo))) as ex:
            for f in [ex.submit(_trial, ds, s, cfg, str(RUNS / ds / f"seed{s}")) for s in todo]:
                f.result()
    shutil.rmtree(T.trial_dir, ignore_errors=True)
    print(f"[{ds}] runs/{ds}/seed*/astdpd.json now use the retuned config; "
          f"shipped results in astdpd_shipped.json", flush=True)


# ------------------------------------------------------------------ coexisting with `run.py all`
def run_all_alive():
    """True while the `run.py all` scheduler is running."""
    r = subprocess.run(["pgrep", "-f", "run.py all"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def wait_baselines(ds, poll=120):
    """Block until every seed's baseline selections exist (the threshold needs them)."""
    while True:
        missing = [s for s in SEEDS if baseline_valid_avgpop(ds, s) is None]
        if not missing:
            return
        if not run_all_alive():
            raise SystemExit(f"{ds}: baselines missing for seeds {missing} and `run.py all` is not running")
        print(f"[{ds}] waiting for baseline selections of seeds {missing}", flush=True)
        time.sleep(poll)


def _has_shipped(ds, s, shipped):
    d = RUNS / ds / f"seed{s}"
    if (d / "astdpd_shipped.json").exists():
        return True
    f = d / "astdpd.json"
    if not f.exists():
        return False
    r = json.load(open(f))
    return r.get("config") == shipped and "wall_time_s" in r


def wait_shipped(T, poll=120):
    """Before writing configs/astdpd_selected.yaml (which `run.py astdpd` would then use), let
    `run.py all` finish its shipped-config ASTD-PD runs of this dataset; if the scheduler is gone
    and some are missing, train them here (identical call: shipped config, same seed)."""
    ds, shipped = T.ds, shipped_config(T.ds)
    while True:
        missing = [s for s in SEEDS if not _has_shipped(ds, s, shipped)]
        if not missing:
            return
        if not run_all_alive():
            break
        print(f"[{ds}] waiting for run.py all's shipped-config ASTD-PD runs of seeds {missing}", flush=True)
        time.sleep(poll)
    print(f"[{ds}] `run.py all` not running; training shipped config for seeds {missing}", flush=True)
    with T.pool(min(T.workers, len(missing))) as ex:
        for f in [ex.submit(_trial, ds, s, shipped, str(RUNS / ds / f"seed{s}")) for s in missing]:
            f.result()


def pick_gpu():
    """The GPU with the most free memory."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        rows = [tuple(int(x) for x in l.split(",")) for l in out.strip().splitlines()]
        return max(rows, key=lambda r: r[1])[0]
    except Exception:
        return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-random", type=int, default=160)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--confirm-seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpu", type=int, nargs="+", default=None,
                    help="GPU(s); workers are spread round-robin (default: the one with most free memory)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-final", action="store_true", help="search only; do not write/run the winner")
    ap.add_argument("--mu-grid", type=float, nargs="+", default=None,
                    help="inference-penalty values crossed with the shipped config in stage 1 "
                         "(at temperature 0.2; default MU_GRID)")
    ap.add_argument("--final-only", action="store_true",
                    help="only the final step for the winner already recorded in runs/<ds>/astdpd_tune.json")
    ap.add_argument("--reselect", action="store_true",
                    help="only re-apply the stage-2 selection rule to the existing tune log (no training)")
    ap.add_argument("--wait", action="store_true", help="wait for the baselines before searching and for "
                    "`run.py all`'s shipped-config ASTD-PD runs before the final step")
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.reselect:
        reselect(a.dataset)
    elif a.final_only:
        finalize_only(a.dataset, a.workers, a.gpu, a.threads, wait=a.wait)
    else:
        tune(a.dataset, a.n_random, a.top_k, a.confirm_seeds, a.workers, a.gpu, a.threads,
             final=not a.no_final, wait=a.wait, mu_grid=a.mu_grid)
