"""Single entry point of the benchmark (run from anywhere; paths are relative to this file).

    python run.py backbone  [--datasets ...] [--seeds ...] [--gpu N]   # (a) LightGCN backbone
    python run.py baselines [--methods ppd tsp dap] [...]             # (b-d) post-hoc, tuned on valid
    python run.py astdpd    [...]                                     # (e) ASTD-PD
    python run.py report                                              # results/{results,significance}.json, tables.txt
    python run.py all       [--gpus 0 1] [--gpu-slots 3] [--cpu-jobs 18]   # everything (scripts/run_all.sh)

Every step is idempotent: finished outputs (for the current backbone checkpoint / config) are
skipped. `--gpu -1` runs on CPU. PPD's official loops are CPU-only and single-threaded, so `all`
runs them in a CPU process pool: per backbone one job computes the popularity scores p, then one
job per beta computes b(beta) and scores both phi values, then a final job selects on validation
(`baselines --methods ppd --prep` / `--only beta=<b>` / plain).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

THESIS = Path(__file__).resolve().parent
RUNS, LOGS, RESULTS = THESIS / "runs", THESIS / "logs", THESIS / "results"
DATASETS = ["Coat", "Yahoo", "KuaiRec", "Gowalla"]          # small first
SEEDS = [2024, 2025, 2026, 2027, 2028]
BASELINES = ["ppd", "tsp", "dap"]


def _setup_device(gpu: int | None, threads: int | None):
    """Must run before torch is imported."""
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "" if gpu < 0 else str(gpu)
    if threads:
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[v] = str(threads)
    sys.path.insert(0, str(THESIS))
    os.chdir(THESIS)
    import torch
    if threads:
        torch.set_num_threads(threads)


# ----------------------------------------------------------------------------------- steps
def step_backbone(datasets, seeds):
    from src import backbone as B
    for ds in datasets:
        for s in seeds:
            if not B.is_trained(ds, s):
                B.train(ds, s)
            f = RUNS / ds / f"seed{s}" / "lightgcn.json"
            if not f.exists() or json.load(open(f)).get("backbone_id") != B.checkpoint_id(ds, s):
                B.evaluate_backbone(ds, s, "cuda" if _cuda() else "cpu")
            else:
                print(f"[{ds} s{s}] backbone done, skipped", flush=True)


def step_baselines(datasets, seeds, methods, only=None, prep=False):
    from src.methods import parse_hparams, ppd, tune as T
    for ds in datasets:
        for s in seeds:
            for m in methods:
                if T.is_done(ds, s, m):
                    print(f"[{m} {ds} s{s}] done, skipped", flush=True)
                    continue
                if prep and m != "ppd":
                    continue
                if m == "ppd":
                    ppd.CACHE_DIR = RUNS / ds / f"seed{s}" / "ppd_cache"
                if prep:  # PPD only: the backbone's global popularity scores p
                    from src.backbone import load_backbone
                    t0 = time.time()
                    ppd.prepare(load_backbone(ds, s, "cpu")[0])
                    print(f"[ppd {ds} s{s}] popularity scores ready ({time.time() - t0:.0f}s)", flush=True)
                    continue
                T.tune(ds, s, m, only=parse_hparams(only) if only else None)


def step_astdpd(datasets, seeds):
    from src.astdpd import train as A
    for ds in datasets:
        for s in seeds:
            f = RUNS / ds / f"seed{s}" / "astdpd.json"
            if f.exists():
                r = json.load(open(f))
                if r.get("config") == A.load_config(ds) and "wall_time_s" in r:
                    print(f"[astdpd {ds} s{s}] done, skipped", flush=True)
                    continue
            A.run(ds, s)


def _cuda():
    import torch
    return torch.cuda.is_available()


# ---------------------------------------------------------------------------------- report
METHOD_FILES = {"LightGCN": "lightgcn", "PPD": "ppd", "TSP": "tsp", "DAP": "dap", "ASTD-PD": "astdpd",
                # shipped-config ASTD-PD, kept when src/astdpd/tune.py re-tuned the dataset (reference
                # row only: not part of the popularity race)
                "ASTD-PD (shipped)": "astdpd_shipped"}
SIG_METRICS = ("recall@20", "ndcg@20", "precision@20", "avgpop@20")


def _load_runs(ds, seeds):
    """{method: {seed: result json}}; post-hoc results must match the seed's backbone."""
    import numpy as np
    out, stale = {}, []
    for name, stem in METHOD_FILES.items():
        for s in seeds:
            d = RUNS / ds / f"seed{s}"
            f = d / (f"{stem}.json" if stem in ("lightgcn", "astdpd", "astdpd_shipped") else f"{stem}_selected.json")
            if not f.exists():
                continue
            r = json.load(open(f))
            if stem in BASELINES:
                bb = d / "lightgcn.json"
                if not bb.exists() or json.load(open(bb))["backbone_id"] != r.get("backbone_id"):
                    stale.append(f"{ds}/seed{s}/{f.name}")
                    continue
            pu = d / f"{stem}_per_user_test.npz"
            if pu.exists():
                r["test"]["per_user"] = dict(np.load(pu))
            out.setdefault(name, {})[s] = r
    return out, stale


def _wall(r):
    return r.get("wall_time_s") or r.get("grid_time_s") or r.get("train_wall_time_s")


def step_report(datasets, seeds):
    import numpy as np
    from src import significance as S
    RESULTS.mkdir(exist_ok=True)
    results, sig, lines, stale_all, blocks = {}, {}, [], [], {}
    fmt = lambda m, k: f"{m['mean'][k]:.4f} ± {m['std'][k]:.4f}" if k in m["mean"] else "-"
    fmtp = lambda m, k: f"{m['mean'][k]:.1f} ± {m['std'][k]:.1f}" if k in m["mean"] else "-"
    for ds in datasets:
        runs, stale = _load_runs(ds, seeds)
        stale_all += stale
        if not runs:
            continue
        results[ds] = {}
        for name, by_seed in runs.items():
            keys = [k for k in next(iter(by_seed.values()))["test"] if k.endswith(("@10", "@20"))]
            vals = {k: np.array([by_seed[s]["test"][k] for s in by_seed]) for k in keys}
            results[ds][name] = dict(
                seeds=sorted(by_seed), n_seeds=len(by_seed),
                mean={k: float(v.mean()) for k, v in vals.items()},
                std={k: float(v.std(ddof=1)) if len(v) > 1 else 0.0 for k, v in vals.items()},
                per_seed_test={s: {k: by_seed[s]["test"][k] for k in keys} for s in sorted(by_seed)},
                selected_hparams={s: by_seed[s].get("selected_hparams") for s in sorted(by_seed)}
                if name in ("PPD", "TSP", "DAP") else None,
                best_epoch={s: by_seed[s].get("best_epoch") for s in sorted(by_seed)}
                if name in ("LightGCN", "ASTD-PD", "ASTD-PD (shipped)") else None,
                wall_time_s={s: _wall(by_seed[s]) for s in sorted(by_seed)})
        if "ASTD-PD" in runs:
            sig[ds] = {}
            for name in runs:
                if name == "ASTD-PD":
                    continue
                base = {s: r["test"] for s, r in runs[name].items()}
                cand = {s: r["test"] for s, r in runs["ASTD-PD"].items()}
                sig[ds][f"ASTD-PD vs {name}"] = S.compare(base, cand, SIG_METRICS)

        res = results[ds]
        start = len(lines)
        lines += [f"### {ds}", "",
                  "| Method | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 ↓ | APLT@20 | Coverage@20 | TailRecall@20 | Recall@10 | NDCG@10 | AvgPop@10 ↓ | seeds | time/seed (s) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name in METHOD_FILES:
            if name not in res:
                continue
            m = res[name]
            wt = [t for t in m["wall_time_s"].values() if t]
            wt = f"{np.mean(wt):.0f}" if wt else "-"
            lines.append(f"| {name} | {fmt(m, 'recall@20')} | {fmt(m, 'ndcg@20')} | {fmt(m, 'precision@20')} | "
                         f"{fmtp(m, 'avgpop@20')} | {fmt(m, 'aplt@20')} | {fmt(m, 'coverage@20')} | "
                         f"{fmt(m, 'tail_recall@20')} | {fmt(m, 'recall@10')} | {fmt(m, 'ndcg@10')} | "
                         f"{fmtp(m, 'avgpop@10')} | {m['n_seeds']} | {wt} |")
        if ds in sig:
            lines += ["", "ASTD-PD minus baseline on test @20 (mean over seeds); Holm-adjusted p: "
                      "paired t-test over seeds / per-user Wilcoxon (pooled over seeds).", "",
                      "| Comparison | ΔRecall | p (seed / user) | ΔNDCG | p (seed / user) | ΔPrecision | p (seed / user) | ΔAvgPop | p (seed / user) |",
                      "|---|---|---|---|---|---|---|---|---|"]
            pf = lambda e: "-" if e is None or e.get("p_holm") is None else f"{e['p_holm']:.2g}"
            for comp, rep in sig[ds].items():
                cells = []
                for k in SIG_METRICS:
                    a = rep["across_seed"][k]
                    cells += [f"{a['mean_diff']:+.4f}" if k != "avgpop@20" else f"{a['mean_diff']:+.1f}",
                              f"{pf(a)} / {pf(rep['per_user'].get(k))}"]
                lines.append(f"| {comp} | " + " | ".join(cells) + " |")
        pops = {n: m["mean"]["avgpop@20"] for n, m in res.items() if n != "ASTD-PD (shipped)"}
        winner = min(pops, key=pops.get)
        race = (f"**Popularity race ({ds}):** lowest test AvgPop@20 = {winner} ({pops[winner]:.1f}); "
                + ("ASTD-PD wins." if winner == "ASTD-PD" else
                   f"ASTD-PD does not win ({pops['ASTD-PD']:.1f})." if "ASTD-PD" in pops else "ASTD-PD missing."))
        results[ds]["popularity_race"] = dict(lowest_avgpop20=winner, astdpd_wins=winner == "ASTD-PD",
                                              mean_avgpop20=pops)
        lines += ["", race, ""]
        blocks[ds] = lines[start:]
    meta = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), seeds=seeds,
                std="sample std over seeds (ddof=1)", stale_results_ignored=stale_all,
                selection={"LightGCN": "valid Recall@20 every 5 epochs, patience 10 checks",
                           "PPD/TSP/DAP": "grid on valid Recall@20 (ties NDCG@20), configs/grids.yaml",
                           "ASTD-PD": "valid Pareto hypervolume, src/astdpd/train.py; hyperparameters re-tuned on "
                                       "validation (src/astdpd/tune.py) where configs/astdpd_selected.yaml lists the dataset"})
    json.dump({"meta": meta, **results}, open(RESULTS / "results.json", "w"), indent=2, default=_np)
    json.dump({"meta": {**meta, "metrics": SIG_METRICS, "holm": "across the 4 metrics, per level"}, **sig},
              open(RESULTS / "significance.json", "w"), indent=2, default=_np)
    head = ["<!-- generated by `python run.py report`; do not edit -->",
            f"Test metrics, mean ± sample std over seeds {seeds}. ↓ = lower is better.", ""]
    if stale_all:
        head += [f"Ignored (computed on an older backbone): {', '.join(stale_all)}", ""]
    (RESULTS / "tables.txt").write_text("\n".join(head + lines) + "\n")
    _update_readme(results, sig, blocks, datasets, seeds, meta["generated"])
    print("\n".join(lines))
    print(f"-> {RESULTS}/results.json, significance.json, tables.txt")


# ---------------------------------------------------------------- README results (markers)
README = THESIS / "README.md"
RACE = ("LightGCN", "PPD", "TSP", "DAP", "ASTD-PD")


def _retune_pending(ds, seeds):
    """True while an ASTD-PD validation re-tune of `ds` is planned or running but not installed:
    the dataset has a tune log or a `search <ds>` line in scripts/astdpd_tune_all.sh, and
    configs/astdpd_selected.yaml + every seed's astdpd.json / astdpd_shipped.json do not yet
    reflect the winner."""
    import re
    import yaml
    sh = THESIS / "scripts" / "astdpd_tune_all.sh"
    planned = (RUNS / ds / "astdpd_tune.json").exists() or (
        sh.exists() and re.search(rf"^search {ds}\b", sh.read_text(), re.M) is not None)
    if not planned:
        return False
    f = THESIS / "configs" / "astdpd_selected.yaml"
    sel = (yaml.safe_load(open(f)) or {}) if f.exists() else {}
    if ds not in sel:
        return True
    for s in seeds:
        d = RUNS / ds / f"seed{s}"
        if not (d / "astdpd_shipped.json").exists() or not (d / "astdpd.json").exists() \
                or json.load(open(d / "astdpd.json")).get("config") != sel[ds]["config"]:
            return True
    return False


def _update_readme(results, sig, blocks, datasets, seeds, generated):
    """Rewrite the README regions between <!-- RESULTS:BEGIN/END --> and <!-- HEADLINE:BEGIN/END -->
    (only datasets whose five methods are complete on every seed; others show "running").
    Does nothing if README.md or a marker pair is missing."""
    if not README.exists():
        return
    n, rows, head_lines, body = len(seeds), [], [], []
    for ds in datasets:
        res = results.get(ds, {})
        have = {m: res[m]["n_seeds"] if m in res else 0 for m in RACE}
        pending = _retune_pending(ds, seeds)
        if any(v < n for v in have.values()) or pending:
            prog = ", ".join(f"{m} {v}/{n}" for m, v in have.items())
            note = "; ASTD-PD validation re-tune not yet installed" if pending else ""
            rows.append(f"| {ds} | running | - | - | - | - | - |")
            body += [f"### {ds}", "", f"*Running.* Finished seeds: {prog}{note}. "
                     "Partial numbers: `results/tables.txt`.", ""]
            continue
        mean = {m: res[m]["mean"] for m in res if m != "popularity_race"}
        base = min(RACE[:4], key=lambda m: mean[m]["avgpop@20"])
        a, b, l = mean["ASTD-PD"], mean[base], mean["LightGCN"]
        rel = lambda x, y: f"{100 * (x - y) / y:+.1f}%"
        pa = (sig.get(ds, {}).get(f"ASTD-PD vs {base}") or {})
        pp = lambda e: "-" if not e or e.get("p_holm") is None else f"{e['p_holm']:.2g}"
        p_txt = (f"Holm p = {pp(pa.get('across_seed', {}).get('avgpop@20'))} across seeds / "
                 f"{pp(pa.get('per_user', {}).get('avgpop@20'))} per user")
        win = res["popularity_race"]["lowest_avgpop20"]
        rows.append(f"| {ds} | complete | {win} | {a['avgpop@20']:.1f} | {base} {b['avgpop@20']:.1f} "
                    f"({rel(a['avgpop@20'], b['avgpop@20'])}) | {a['ndcg@20']:.4f} vs {l['ndcg@20']:.4f} "
                    f"({rel(a['ndcg@20'], l['ndcg@20'])}) | {a['recall@20']:.4f} vs {l['recall@20']:.4f} "
                    f"({rel(a['recall@20'], l['recall@20'])}) |")
        head_lines.append(
            f"- **{ds}**: lowest test AvgPop@20 = {win}. ASTD-PD {a['avgpop@20']:.1f} vs the least "
            f"popular baseline ({base}) {b['avgpop@20']:.1f} ({rel(a['avgpop@20'], b['avgpop@20'])}, {p_txt}); "
            f"NDCG@20 {rel(a['ndcg@20'], l['ndcg@20'])} and Recall@20 {rel(a['recall@20'], l['recall@20'])} "
            f"vs LightGCN.")
        body += blocks.get(ds, [])
    summary = [f"*Generated by `python run.py report` on {generated} from `results/results.json`; "
               f"do not edit by hand. Test split, mean ± sample std over seeds {seeds}; ↓ = lower is better.*",
               "", "| Dataset | status | lowest AvgPop@20 | ASTD-PD AvgPop@20 ↓ | least-popular baseline "
               "AvgPop@20 (ASTD-PD change) | ASTD-PD vs LightGCN NDCG@20 | ASTD-PD vs LightGCN Recall@20 |",
               "|---|---|---|---|---|---|---|", *rows, ""]
    headline = head_lines or ["- No dataset is complete yet; see Results."]
    running = [ds for ds in datasets if f"| {ds} | running" in "\n".join(rows)]
    if running:
        headline.append(f"- Still running: {', '.join(running)}.")
    text = README.read_text()
    for tag, content in (("RESULTS", summary + body), ("HEADLINE", headline)):
        b, e = f"<!-- {tag}:BEGIN -->", f"<!-- {tag}:END -->"
        i, j = text.find(b), text.find(e)
        if i < 0 or j < i:
            continue
        text = text[:i + len(b)] + "\n" + "\n".join(content).rstrip() + "\n" + text[j:]
    README.write_text(text)


def _np(o):
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ------------------------------------------------------------------------------- scheduler
def step_all(datasets, seeds, gpus, gpu_slots, cpu_jobs, ppd_betas=None):
    """Dependency-aware job runner: GPU slots (gpu_slots per GPU) and a CPU pool for PPD."""
    import yaml
    LOGS.mkdir(exist_ok=True)
    betas = ppd_betas or yaml.safe_load(open(THESIS / "configs" / "grids.yaml"))["ppd"]["grid"]["beta"]
    py, me = sys.executable, str(THESIS / "run.py")
    jobs = {}

    def add(name, kind, args, deps=(), prio=0):
        jobs[name] = dict(name=name, kind=kind, args=args, deps=list(deps), prio=prio, state="pending")

    for di, ds in enumerate(datasets):
        for s in seeds:
            a, tag = ["--datasets", ds, "--seeds", str(s)], f"{ds}/s{s}"
            add(f"{tag}/backbone", "gpu", ["backbone", *a], prio=(0, di, s))
            for m in ("tsp", "dap"):
                add(f"{tag}/{m}", "gpu", ["baselines", "--methods", m, *a], [f"{tag}/backbone"], (1, di, s))
            add(f"{tag}/ppd-prep", "cpu", ["baselines", "--methods", "ppd", "--prep", *a],
                [f"{tag}/backbone"], (0, di, s))
            for b in betas:
                add(f"{tag}/ppd-beta{b}", "cpu", ["baselines", "--methods", "ppd", "--only", f"beta={b}", *a],
                    [f"{tag}/ppd-prep"], (1, di, s))
            add(f"{tag}/ppd", "cpu", ["baselines", "--methods", "ppd", *a],
                [f"{tag}/ppd-beta{b}" for b in betas], (2, di, s))
            add(f"{tag}/astdpd", "gpu", ["astdpd", *a], prio=(2, di, s))
    # the report always covers every dataset/seed with results, not only the ones scheduled here
    add("report", "cpu", ["report"],
        [n for n in jobs], (9, 0, 0))

    slots = {f"gpu{g}.{k}": g for g in gpus for k in range(gpu_slots)}
    free_gpu, cpu_running = list(slots), 0
    running, t_start = {}, time.time()
    jlog = open(LOGS / "jobs.jsonl", "a")
    print(f"[all] {len(jobs)} jobs, GPUs {gpus} x {gpu_slots} slots, CPU pool {cpu_jobs}", flush=True)

    def write_status():
        n = {st: sum(j["state"] == st for j in jobs.values()) for st in ("done", "running", "pending", "failed", "skipped")}
        rows = [f"{time.strftime('%Y-%m-%d %H:%M:%S')}  elapsed {(time.time() - t_start) / 3600:.2f} h  "
                + "  ".join(f"{k}={v}" for k, v in n.items())]
        rows += [f"  running {j['name']:28s} {j.get('slot', 'cpu'):8s} {(time.time() - j['t0']) / 60:7.1f} min"
                 for j in sorted(running.values(), key=lambda j: j["t0"])]
        rows += [f"  FAILED  {j['name']} (see {j['log']})" for j in jobs.values() if j["state"] == "failed"]
        (LOGS / "status.txt").write_text("\n".join(rows) + "\n")

    while True:
        for name, j in list(running.items()):
            rc = j["proc"].poll()
            if rc is None:
                continue
            j["state"] = "done" if rc == 0 else "failed"
            wall = time.time() - j["t0"]
            jlog.write(json.dumps(dict(job=name, state=j["state"], rc=rc, slot=j.get("slot", "cpu"),
                                       start=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(j["t0"])),
                                       wall_s=round(wall, 1), log=j["log"])) + "\n")
            jlog.flush()
            print(f"[all] {j['state']:6s} {name} ({wall / 60:.1f} min)", flush=True)
            del running[name]
            if j["kind"] == "gpu":
                free_gpu.append(j["slot"])
            else:
                cpu_running -= 1
        changed = True
        while changed:  # propagate failures
            changed = False
            for j in jobs.values():
                if j["state"] == "pending" and j["name"] != "report" and any(
                        jobs[d]["state"] in ("failed", "skipped") for d in j["deps"]):
                    j["state"], changed = "skipped", True
        ready = sorted((j for j in jobs.values() if j["state"] == "pending" and all(
            jobs[d]["state"] in ("done", "failed", "skipped") if j["name"] == "report"
            else jobs[d]["state"] == "done" for d in j["deps"])), key=lambda j: j["prio"])
        for j in ready:
            if j["kind"] == "gpu":
                if not free_gpu:
                    continue
                free_gpu.sort()
                # spread over GPUs: take a slot on the least-loaded GPU
                load = {g: sum(1 for r in running.values() if r.get("gpu") == g) for g in gpus}
                slot = min(free_gpu, key=lambda sl: (load[slots[sl]], sl))
                free_gpu.remove(slot)
                j.update(slot=slot, gpu=slots[slot])
                extra, threads = ["--gpu", str(slots[slot])], "4"
            else:
                if cpu_running >= cpu_jobs:
                    continue
                cpu_running += 1
                extra, threads = ["--gpu", "-1"], "1"
            log = LOGS / (j["name"].replace("/", "_") + ".log")
            j.update(state="running", t0=time.time(), log=str(log))
            j["proc"] = subprocess.Popen([py, me, *j["args"], *extra, "--threads", threads], cwd=THESIS,
                                         stdout=open(log, "a"), stderr=subprocess.STDOUT)
            running[j["name"]] = j
            print(f"[all] start  {j['name']} on {j.get('slot', 'cpu')}", flush=True)
        write_status()
        if not running and not any(j["state"] == "pending" for j in jobs.values()):
            break
        time.sleep(5)
    failed = [n for n, j in jobs.items() if j["state"] in ("failed", "skipped")]
    print(f"[all] finished in {(time.time() - t_start) / 3600:.2f} h; failed/skipped: {failed or 'none'}",
          flush=True)
    return 1 if failed else 0


# -------------------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["backbone", "baselines", "astdpd", "report", "all"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--gpu", type=int, default=None, help="GPU index (sets CUDA_VISIBLE_DEVICES); -1 = CPU")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads of this process")
    ap.add_argument("--methods", nargs="+", default=BASELINES, choices=BASELINES)
    ap.add_argument("--only", nargs="*", default=None, help="baselines: evaluate only grid points "
                    "matching k=v (no selection; rows are cached for the final call)")
    ap.add_argument("--prep", action="store_true", help="baselines (ppd): only compute popularity scores p")
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1], help="all: GPUs to use")
    ap.add_argument("--gpu-slots", type=int, default=3, help="all: concurrent jobs per GPU")
    ap.add_argument("--cpu-jobs", type=int, default=18, help="all: concurrent CPU (PPD) jobs")
    a = ap.parse_args()
    datasets = [d for d in DATASETS if d in a.datasets]  # canonical order: small first
    if a.step == "all":
        sys.exit(step_all(datasets, a.seeds, a.gpus, a.gpu_slots, a.cpu_jobs))
    _setup_device(a.gpu, a.threads)
    t0 = time.time()
    if a.step == "backbone":
        step_backbone(datasets, a.seeds)
    elif a.step == "baselines":
        step_baselines(datasets, a.seeds, a.methods, a.only, a.prep)
    elif a.step == "astdpd":
        step_astdpd(datasets, a.seeds)
    else:
        step_report(datasets, a.seeds)
    print(f"[{a.step}] wall time {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
