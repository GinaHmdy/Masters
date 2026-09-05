"""
Phase 5: build the manuscript numbers pack.

The manuscript is drafted outside this repository, so the deliverable of the experimental work
is not prose -- it is a set of paste-ready tables and figure-ready CSVs whose every cell can be
traced to a run. This script writes them into `results/numbers_pack/`, regenerating from the
same `results/*.json` that RESULTS_CANONICAL.md reads, so the two can never disagree.

Outputs:
    numbers_pack/README.md              what each file is, and the caveats that must travel with it
    numbers_pack/table1_main.md         headline cross-dataset comparison (markdown)
    numbers_pack/table1_main.tex        the same, as a LaTeX booktabs table
    numbers_pack/table2_significance.md paired tests vs LightGCN
    numbers_pack/table3_ablation.md     mechanism-isolation arms, per dataset
    numbers_pack/table4_gates.md        final alpha/theta/lambda vs their configured starts
    numbers_pack/*.csv                  one CSV per table, for plotting
"""
import csv
import json
import os

CODE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(CODE, "results")
OUT = os.path.join(RESULTS, "numbers_pack")

DATASETS = ["Gowalla", "Yahoo", "KuaiRec", "Coat"]
ORDER = ["LightGCN", "TSP", "PPD v3", "PPD v4", "PPD v6"]
COLS = [("recall@20", "Recall@20", 6), ("ndcg@20", "NDCG@20", 6),
        ("precision@20", "Precision@20", 6), ("avgpop@20", "AvgPop@20", 2),
        ("coverage@20", "Coverage@20", 4), ("aplt@20", "APLT@20", 4),
        ("gini@20", "Gini@20", 4)]
LOWER_BETTER = {"avgpop@20", "gini@20"}
ARM_LABEL = {
    "a_full": "(a) full v6",
    "b_no_spectral": "(b) no spectral projection",
    "c_no_simplex": "(c) no simplex stream",
    "d_no_tail": "(d) no tail reweighting",
    "e_recipe_only": "(e) recipe only (no debiasing)",
}
# Caveats that must not be separated from the numbers they qualify.
CAVEAT = {
    "Gowalla": "Test users are 100% of all users; unaffected by either protocol fix.",
    "Yahoo": "Only 16.6% of users are in the test split -- every Yahoo number describes that minority.",
    "KuaiRec": "Candidates restricted to the 3,247-item test universe; AvgPop scoped to test users.",
    "Coat": "TSP builds zero edges on Coat and never runs, so its rows equal LightGCN by construction.",
}


def load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def trivial(ds):
    """MostPopular / Random reference rows. Finding Z: these belong IN the table, not a footnote --
    on KuaiRec every method except PPD v6 loses to MostPopular on NDCG@20."""
    d = load("trivial_baselines.json")
    if not d:
        return []
    ref = d.get("results", {}).get(ds)
    if not ref:
        return []
    rows = []
    if "MostPopular" in ref:
        rows.append(("MostPopular *(reference)*", ref["MostPopular"], None))
    if "Random" in ref:
        rows.append(("Random *(reference)*", ref["Random"]["mean"], ref["Random"].get("std", {})))
    return rows


def phase1(ds):
    data = {}
    for suffix in ("", "_v6"):
        d = load(f"phase1_{ds.lower()}{suffix}.json")
        if d:
            data.update(d)
    return data


def fmt(mean, std, key, prec):
    if key not in mean:
        return "--"
    sd = std.get(key)
    if sd is None:
        # Deterministic single-run reference (MostPopular): "± 0.000000" would imply a
        # measured spread of zero rather than "no repeats were run".
        return f"{mean[key]:.{prec}f}"
    return f"{mean[key]:.{prec}f} ± {sd:.{prec}f}"


def degenerate(data, ds, method):
    """True if `method` loses to MostPopular on ALL THREE accuracy metrics.

    Finding Z: such a row posts excellent fairness numbers bought by sub-trivial accuracy --
    a degenerate solution, not a favourable trade. Its fairness cells must not be bolded as
    'best in column', or the table recommends the worst model in it.
    """
    triv = load("trivial_baselines.json")
    if not triv:
        return False
    mp = triv.get("results", {}).get(ds, {}).get("MostPopular")
    if not mp or method not in data:
        return False
    mean = data[method]["mean"]
    acc = ("recall@20", "ndcg@20", "precision@20")
    return all(mean.get(k, 0) <= mp.get(k, 0) for k in acc)


def write(name, lines):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [{name}]")


def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  [{name}]")


def table1():
    md = ["# Table 1 - Main cross-dataset comparison", "",
          "3 seeds (2024, 2025, 2026), mean ± population std. Best per column in **bold**.",
          "Higher is better except AvgPop@20 and Gini@20.", ""]
    csv_rows = []
    for ds in DATASETS:
        data = phase1(ds)
        if not data:
            continue
        rows = [m for m in ORDER if m in data]
        md += [f"## {ds}", "", f"*{CAVEAT[ds]}*", "",
               "| Method | " + " | ".join(c[1] for c in COLS) + " |",
               "|---|" + "---|" * len(COLS)]
        degen = {m for m in rows if degenerate(data, ds, m)}
        best = {}
        for key, _, _ in COLS:
            # A degenerate row's fairness numbers are an artifact of it not working.
            vals = [data[m]["mean"][key] for m in rows
                    if key in data[m]["mean"] and m not in degen]
            if vals:
                best[key] = min(vals) if key in LOWER_BETTER else max(vals)
        for m in rows:
            mean, std = data[m]["mean"], data[m]["std"]
            cells = []
            for key, _, prec in COLS:
                txt = fmt(mean, std, key, prec)
                if key in best and key in mean and abs(mean[key] - best[key]) < 1e-12:
                    txt = f"**{txt}**"
                cells.append(txt)
            label = f"{m} ⚠" if m in degen else m
            md.append(f"| {label} | " + " | ".join(cells) + " |")
            csv_rows.append([ds, m] + [mean.get(k, "") for k, _, _ in COLS]
                                    + [std.get(k, "") for k, _, _ in COLS])
        for label, mean, std in trivial(ds):
            cells = [fmt(mean, std if std is not None else {k: None}, k, p) if k in mean else "--"
                     for k, _, p in COLS]
            md.append(f"| {label} | " + " | ".join(cells) + " |")
            csv_rows.append([ds, label.replace(" *(reference)*", "")]
                            + [mean.get(k, "") for k, _, _ in COLS]
                            + [(std or {}).get(k, "") for k, _, _ in COLS])
        if degen:
            md += ["",
                   "> ⚠ " + ", ".join(sorted(degen)) + " loses to **MostPopular** on all three "
                   "accuracy metrics. Its fairness figures are a symptom of the model not "
                   "working, not a favourable trade, and are excluded from the best-in-column "
                   "bolding. See `THESIS_AUDIT.md` finding **Z**."]
        md.append("")
    write("table1_main.md", md)
    write_csv("table1_main.csv",
              ["dataset", "method"] + [k for k, _, _ in COLS]
              + [f"{k}_std" for k, _, _ in COLS], csv_rows)

    tex = [r"% Table 1 - main comparison. Requires \usepackage{booktabs}.",
           r"\begin{tabular}{ll" + "r" * len(COLS) + "}", r"\toprule",
           "Dataset & Method & " + " & ".join(c[1].replace("@", r"@") for c in COLS) + r" \\",
           r"\midrule"]
    for ds in DATASETS:
        data = phase1(ds)
        if not data:
            continue
        for i, m in enumerate([x for x in ORDER if x in data]):
            mean, std = data[m]["mean"], data[m]["std"]
            cells = [f"{mean[k]:.{p}f}" if k in mean else "--" for k, _, p in COLS]
            tex.append(f"{ds if i == 0 else ''} & {m} & " + " & ".join(cells) + r" \\")
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    tex += [r"\end{tabular}"]
    write("table1_main.tex", tex)


def table2():
    md = ["# Table 2 - Paired significance, PPD v6 vs LightGCN", "",
          "Two tests, and a result should clear both:",
          "",
          "- **per-user**: Wilcoxon signed-rank over per-user metric vectors. High power",
          "  (n = test users) but holds the trained model fixed, so it says nothing about seed variance.",
          "- **across-seed**: paired t-test over seed means. Low power (n = 3) but this is the one",
          "  that answers whether the margin survives retraining.",
          "",
          "p-values are Holm-corrected across the three metrics. *sigma* is the mean difference",
          "expressed in baseline seed standard deviations.", "",
          "| Dataset | Metric | Δ mean | p (per-user) | p (across-seed) | σ | Verdict |",
          "|---|---|---|---|---|---|---|"]
    rows = []
    warned = False
    for ds in DATASETS:
        d = load(f"significance_{ds.lower()}_v6.json")
        if not d:
            continue
        pu_all, as_all = d.get("per_user", {}), d.get("across_seed", {})
        for metric in ("recall", "ndcg", "precision"):
            pu, asd = pu_all.get(metric, {}), as_all.get(metric, {})
            if not pu and not asd:
                continue
            diff = asd.get("mean_diff", pu.get("mean_diff", 0.0))
            p_pu = pu.get("p_holm", pu.get("p_value"))
            p_as = asd.get("p_holm", asd.get("p_value"))
            sigma = asd.get("sigma_of_baseline")
            if asd.get("low_n_warning"):
                warned = True
            both = (p_pu is not None and p_pu < 0.05) and (p_as is not None and p_as < 0.05)
            if both:
                verdict = "**significant (both)**"
            elif p_pu is not None and p_pu < 0.05:
                verdict = "per-user only"
            else:
                verdict = "n.s."
            if diff < 0 and both:
                verdict = "**significant LOSS**"
            def pf(v):
                # Wilcoxon p-values on ~90k users underflow to exactly 0.0 in float64;
                # printing "0" would claim infinite certainty.
                if v is None:
                    return "--"
                return "<1e-300" if v == 0.0 else f"{v:.3g}"

            md.append(f"| {ds} | {metric}@20 | {diff:+.6f} | "
                      f"{pf(p_pu)} | "
                      f"{pf(p_as)} | "
                      f"{'--' if sigma is None else f'{sigma:+.2f}'} | {verdict} |")
            rows.append([ds, f"{metric}@20", diff, p_pu, p_as, sigma, verdict.replace("*", "")])
    md += ["", "**Caution on n=3.** With three seeds a paired t-test has almost no power, and a",
           "near-constant difference across seeds inflates the statistic rather than evidencing",
           "the effect -- which is why Coat precision reports p ~ 1e-30 on a -2.4 sigma difference.",
           "Read sigma as the evidence and the p-value as a sanity check, never the reverse."]
    if warned:
        md += ["", "*(`significance.py` emitted a low-n warning on at least one row above.)*"]
    write("table2_significance.md", md)
    write_csv("table2_significance.csv",
              ["dataset", "metric", "diff", "p_per_user_holm", "p_across_seed_holm",
               "sigma_of_baseline", "verdict"], rows)


def table3():
    md = ["# Table 3 - Mechanism-isolation ablation", "",
          "Arm (e) is the confound control: PPD v6 changes the loss, the sampler and the",
          "parameter count alongside the architecture, so (e) strips all three debiasing",
          "mechanisms while keeping the training recipe. The gap between (a) and (e) is what",
          "the debiasing itself buys.", ""]
    rows = []
    any_data = False
    for ds in DATASETS:
        d = load(f"phase2_ablation_{ds.lower()}.json")
        if not d:
            md += [f"## {ds}", "", "*not yet run*", ""]
            continue
        any_data = True
        md += [f"## {ds}", "",
               "| Arm | Recall@20 | NDCG@20 | AvgPop@20 | APLT@20 | Coverage@20 | α | θ | λ |",
               "|---|---|---|---|---|---|---|---|---|"]
        for arm, v in d.items():
            m, s = v["mean"], v["std"]
            md.append(f"| {ARM_LABEL.get(arm, arm)} | "
                      f"{fmt(m, s, 'recall@20', 6)} | {fmt(m, s, 'ndcg@20', 6)} | "
                      f"{fmt(m, s, 'avgpop@20', 2)} | {fmt(m, s, 'aplt@20', 4)} | "
                      f"{fmt(m, s, 'coverage@20', 4)} | "
                      f"{m.get('mean_alpha_final', float('nan')):.4f} | "
                      f"{m.get('mean_theta_final', float('nan')):.4f} | "
                      f"{m.get('mean_lambda_final', float('nan')):.4f} |")
            rows.append([ds, arm, m.get("recall@20"), m.get("ndcg@20"), m.get("avgpop@20"),
                         m.get("aplt@20"), m.get("coverage@20"), m.get("mean_alpha_final"),
                         m.get("mean_theta_final"), m.get("mean_lambda_final")])
        md.append("")
    write("table3_ablation.md", md)
    if any_data:
        write_csv("table3_ablation.csv",
                  ["dataset", "arm", "recall@20", "ndcg@20", "avgpop@20", "aplt@20",
                   "coverage@20", "alpha_final", "theta_final", "lambda_final"], rows)


def table4():
    """Gate values against the configuration that produced them -- the finding Y evidence."""
    try:
        from experiment_configs import DATASET_CONFIGS
    except Exception:
        return
    md = ["# Table 4 - Learned gate values vs their configured starting points", "",
          "The evidence for audit finding **Y**. Two things to read here:", "",
          "1. **λ saturates at its ceiling on every dataset** -- it is a hyperparameter, not a",
          "   learned per-user gate. The earlier reading that it 'sits at its 0.4998",
          "   initialization' was an artifact of Yahoo's init happening to equal its ceiling.",
          "2. **α and θ retain value only where PPD v6 wins.** This is confounded with the",
          "   starting value (Gowalla starts α four times higher); `run_phase3_gate_transfer.py`",
          "   swaps the knobs across datasets to separate the two.", "",
          "| Dataset | α start | α final | % kept | θ start | θ final | % kept | λ ceiling | λ final | % of ceiling |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for ds in DATASETS:
        d = load(f"phase2_ablation_{ds.lower()}.json")
        if not d or "a_full" not in d:
            continue
        m = d["a_full"]["mean"]
        p = DATASET_CONFIGS[ds]["ppd_v6_params"]
        a0 = float(p.get("astd_alpha_base", 0) or 0)
        t0 = float(p.get("astd_topo_gate_init", 0) or 0)
        l0 = float(p.get("astd_fuse_lambda_max", 0) or 0)
        af = m.get("mean_alpha_final", float("nan"))
        tf = m.get("mean_theta_final", float("nan"))
        lf = m.get("mean_lambda_final", float("nan"))
        pa = f"{af / a0 * 100:.1f}%" if a0 else "--"
        pt = f"{tf / t0 * 100:.1f}%" if t0 else "--"
        pl = f"{lf / l0 * 100:.1f}%" if l0 else "--"
        md.append(f"| {ds} | {a0:.2f} | {af:.4f} | {pa} | {t0:.2f} | {tf:.4f} | {pt} | "
                  f"{l0:.2f} | {lf:.4f} | **{pl}** |")
        rows.append([ds, a0, af, t0, tf, l0, lf])
    write("table4_gates.md", md)
    write_csv("table4_gates.csv",
              ["dataset", "alpha_start", "alpha_final", "theta_start", "theta_final",
               "lambda_ceiling", "lambda_final"], rows)


def table5():
    """Gate-transfer arms: does the collapse follow the data or the initialization?"""
    try:
        from experiment_configs import DATASET_CONFIGS
    except Exception:
        return
    STARTS = {"gowalla_init": (0.40, 0.40), "yahoo_init": (0.10, 0.75)}
    md = ["# Table 5 - Gate transfer: adaptation vs initialization", "",
          "Finding **Y** observed that alpha and theta retain value on Gowalla and collapse",
          "elsewhere -- but Gowalla also *starts* alpha four times higher, so retention and",
          "initialization were confounded. These arms swap `astd_alpha_base` / `astd_topo_gate_init`",
          "across datasets, holding everything else at the dataset's tuned configuration.", "",
          "Read the **% kept** columns, not the absolute values: if the collapse tracked the",
          "starting value, a larger start would retain proportionally more.", ""]
    rows = []
    for ds in DATASETS:
        d = load(f"phase3_gate_transfer_{ds.lower()}.json")
        if not d:
            continue
        native = DATASET_CONFIGS[ds]["ppd_v6_params"]
        n0 = (float(native.get("astd_alpha_base", 0) or 0),
              float(native.get("astd_topo_gate_init", 0) or 0))
        md += [f"## {ds}", "", f"*Native configuration: alpha_base={n0[0]}, topo_gate_init={n0[1]}*", "",
               "| Arm | alpha start | alpha final | % kept | theta start | theta final | % kept | lambda | Recall@20 |",
               "|---|---|---|---|---|---|---|---|---|"]
        for arm in ("native", "gowalla_init", "yahoo_init"):
            if arm not in d:
                continue
            m = d[arm]["mean"]
            a0, t0 = STARTS.get(arm, n0)
            af = m.get("mean_alpha_final", float("nan"))
            tf = m.get("mean_theta_final", float("nan"))
            alias = d[arm].get("alias_of")
            label = f"{arm} *(= {alias})*" if alias else arm
            md.append(f"| {label} | {a0:.2f} | {af:.4f} | {af/a0*100:.1f}% | {t0:.2f} | "
                      f"{tf:.4f} | {tf/t0*100:.1f}% | {m.get('mean_lambda_final', float('nan')):.4f} | "
                      f"{m['recall@20']:.6f} |")
            rows.append([ds, arm, alias or "", a0, af, t0, tf,
                         m.get("mean_lambda_final"), m.get("recall@20")])
        aliased = [a for a in d if d[a].get("alias_of")]
        if aliased:
            md += ["", "> Arms marked *(= X)* are the **same configuration** as arm X on this "
                       "dataset -- the override happens to equal the dataset's own value -- so "
                       "they reuse its result and are **not independent replicates**. See "
                       "`THESIS_AUDIT.md` finding **AC**."]
        md.append("")
    write("table5_gate_transfer.md", md)
    if rows:
        write_csv("table5_gate_transfer.csv",
                  ["dataset", "arm", "alias_of", "alpha_start", "alpha_final",
                   "theta_start", "theta_final", "lambda_final", "recall@20"], rows)


def table6():
    """Inference-penalty frontier: the four-metric goal, tested against its own bar."""
    d = load("phase4_inference_penalty_gowalla.json")
    if not d:
        return
    base = load("phase1_gowalla.json") or {}
    bar = {}
    for k in ("recall@20", "ndcg@20", "precision@20"):
        vals = [m["mean"][k] for m in base.values() if k in m.get("mean", {})]
        if vals:
            bar[k] = max(vals)
    pops = [m["mean"]["avgpop@20"] for m in base.values() if "avgpop@20" in m.get("mean", {})]
    if pops:
        bar["avgpop@20"] = min(pops)

    md = ["# Table 6 - Inference-penalty frontier (Gowalla)", "",
          "`--astd_inference_pop_penalty mu` subtracts `mu * p_i` at ranking time, so this sweep",
          "re-ranks the five trained checkpoints rather than retraining anything. **mu is selected",
          "on validation**, by the same 4-D Pareto hypervolume `main.py` uses for checkpoint",
          "selection; the test columns are reported, never used to choose.", ""]
    if bar:
        md += ["The bar a four-metric win must clear is the *strongest baseline on each axis*: "
               + ", ".join(f"{k} {v:.6f}" if k != "avgpop@20" else f"{k} {v:.2f}"
                           for k, v in bar.items()) + ".", ""]
    md += ["| mu | validation HV | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | APLT@20 | clears all four? |",
           "|---|---|---|---|---|---|---|---|"]
    rows = []
    best_mu = max(d.values(), key=lambda v: v["valid_mean"]["hypervolume"])["mu"]
    for key in sorted(d, key=lambda k: d[k]["mu"]):
        v = d[key]; m = v["mean"]; hv = v["valid_mean"]["hypervolume"]
        clears = all([m["recall@20"] > bar.get("recall@20", 9e9),
                      m["ndcg@20"] > bar.get("ndcg@20", 9e9),
                      m["precision@20"] > bar.get("precision@20", 9e9),
                      m["avgpop@20"] < bar.get("avgpop@20", -1)]) if bar else False
        sel = " **<- selected**" if v["mu"] == best_mu else ""
        md.append(f"| {v['mu']} | {hv:.8f}{sel} | {m['recall@20']:.6f} | {m['ndcg@20']:.6f} | "
                  f"{m['precision@20']:.6f} | {m['avgpop@20']:.2f} | {m['aplt@20']:.4f} | "
                  f"{'yes' if clears else 'no'} |")
        rows.append([v["mu"], hv, m["recall@20"], m["ndcg@20"], m["precision@20"],
                     m["avgpop@20"], m["aplt@20"], int(clears)])
    md += ["",
           "> **The window that clears all four is a tie, not a win.** Across-seed paired t-tests",
           "> at the crossover: Recall vs LightGCN p = 0.68 (mu=1.6) and 0.96 (mu=1.75); AvgPop vs",
           "> PPD v4 p = 0.96 and 0.85. Validation hypervolume selects mu = 0 regardless. Quoting a",
           "> row from that window as the goal achieved would be test-set selection at 0.2 sigma.",
           "> See `THESIS_AUDIT.md` finding **AG**.", ""]
    write("table6_penalty_frontier.md", md)
    write_csv("table6_penalty_frontier.csv",
              ["mu", "valid_hypervolume", "recall@20", "ndcg@20", "precision@20",
               "avgpop@20", "aplt@20", "clears_all_four"], rows)


def table7():
    """Validation-selected hyperparameters vs the shipped ones (item 5.7 / finding AH)."""
    rows, md = [], [
        "# Table 7 - Hyperparameters re-selected on validation", "",
        "The shipped configurations were influenced by test metrics: `refine_yahoo_ppd_v6.py`",
        "sorts candidates by seed-2024 **test** results. `run_phase5_hpo.py` re-selects over 96",
        "configurations reading the `[Best Valid Results]` block only, scored by the same 4-D",
        "Pareto hypervolume used for checkpoint selection; `run_phase5_final.py` then retrains",
        "each winner at 5 seeds. Dead knobs are not swept, and each incumbent is a grid point.", ""]
    md += ["## Search outcome", "",
           "| Dataset | configs | incumbent rank | winner's validation-HV gain |",
           "|---|---|---|---|"]
    for ds in DATASETS:
        d = load(f"phase5_hpo_{ds.lower()}.json")
        if not d:
            continue
        ranked = sorted(d.values(), key=lambda v: -v["valid_hypervolume"])
        inc = next((v for v in ranked if v.get("is_incumbent")), None)
        if not inc:
            continue
        gain = 100 * (ranked[0]["valid_hypervolume"] / inc["valid_hypervolume"] - 1)
        md.append(f"| {ds} | {len(d)} | {ranked.index(inc) + 1}th | +{gain:.2f}% |")

    md += ["", "## Effect on the reported numbers, 5 seeds, paired by seed", "",
           "| Dataset | metric | shipped | validation-selected | difference |",
           "|---|---|---|---|---|"]
    for ds in DATASETS:
        v = load(f"phase5_final_{ds.lower()}.json")
        s = load(f"phase1_{ds.lower()}_v6.json")
        if not v or not s:
            continue
        s = s.get("PPD v6", {})
        for key, prec in (("recall@20", 6), ("ndcg@20", 6), ("avgpop@20", 2), ("aplt@20", 4)):
            a, b = v["mean"].get(key), s.get("mean", {}).get(key)
            if a is None or b is None:
                continue
            md.append(f"| {ds} | {key} | {b:.{prec}f} | {a:.{prec}f} | {a - b:+.{prec}f} |")
            rows.append([ds, key, b, a, a - b])
    md += ["",
           "> **The re-selection changed the configuration on every dataset and the results on",
           "> none.** All differences land within 0.4%, no conclusion flips, and Gowalla's",
           "> headline margin over LightGCN slightly strengthens (Recall p = 0.0115 -> 0.00582).",
           "> The shipped configurations remain in `experiment_configs.py` so earlier results keep",
           "> their meaning; these sit beside them. See `THESIS_AUDIT.md` finding **AH**.", ""]
    write("table7_validation_selection.md", md)
    if rows:
        write_csv("table7_validation_selection.csv",
                  ["dataset", "metric", "shipped", "validation_selected", "difference"], rows)


def readme():
    write("README.md", [
        "# Numbers pack", "",
        "Generated by `LightGCN/code/make_numbers_pack.py` from `results/*.json`.",
        "**Do not hand-edit** -- re-run the generator. Every cell traces to a run.", "",
        "| File | Contents |",
        "|---|---|",
        "| `table1_main.{md,tex,csv}` | Headline cross-dataset comparison, 7 metrics, 5 methods |",
        "| `table2_significance.{md,csv}` | Paired tests, PPD v6 vs LightGCN |",
        "| `table3_ablation.{md,csv}` | Mechanism-isolation arms (a)-(e) |",
        "| `table4_gates.{md,csv}` | Learned gates vs configured starts (finding Y) |",
        "| `table5_gate_transfer.{md,csv}` | Gate transfer: adaptation vs initialization (findings Y, AC, AF) |",
        "| `table6_penalty_frontier.{md,csv}` | Inference-penalty frontier; the four-metric goal tested (finding AG) |",
        "| `table7_validation_selection.{md,csv}` | Hyperparameters re-selected on validation (finding AH) |",
        "",
        "## Caveats that must travel with these numbers", "",
        "1. **Five seeds on all four datasets** (2024-2028) as of 2026-08-30; the ablation and",
        "   gate-transfer arms remain at 3. The across-seed test still has limited power, so quote",
        "   sigma (effect size in baseline seed stds) alongside any p-value, and never quote an",
        "   extreme p-value from a handful of near-constant differences as strong evidence.",
        "2. **Test-user coverage varies enormously**: Gowalla 100%, Coat 96.9%, KuaiRec 19.6%,",
        "   Yahoo 16.6%. Yahoo and KuaiRec results describe a minority of each user base.",
        "3. **AvgPop@20 cannot carry the fairness claim alone** -- a Random ranker wins it on",
        "   every dataset. Report Coverage@20 and APLT@20 beside it.",
        "4. **TSP does not run on Coat** (zero edges above the cosine threshold); its Coat rows",
        "   are identical to LightGCN by construction, not by coincidence.",
        "5. **The headline result is Gowalla.** It is the only dataset where PPD v6 achieves a",
        "   Pareto improvement, and the only one whose test users are the entire user base --",
        "   which is why neither protocol fix moved it. On KuaiRec v6 wins accuracy and loses",
        "   fairness outright (AvgPop +41%, Coverage -80%); that is a finding, not an omission.",
        "6. **The credit does not go to the named mechanisms.** The negative sampler accounts for",
        "   79% (Gowalla) to 96% (Yahoo) of the AvgPop reduction and the spectral projection is",
        "   inert on every dataset; the named mechanisms own tail exposure (74% of the APLT gain",
        "   on Gowalla). Any sentence attributing the popularity reduction to the spectral or",
        "   simplex streams is contradicted by table 3.",
        "7. **The four-metric goal is not met anywhere, and on Gowalla it is unreachable** rather",
        "   than merely unreached -- see table 6.",
        "",
        "See `THESIS_AUDIT.md` findings **L, P, S, T, U, V, W, X, Y, AA, AE, AF, AG, AH** for the full provenance.",
    ])


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"writing numbers pack -> {OUT}")
    table1(); table2(); table3(); table4(); table5(); table6(); table7(); readme()
    print("done")


if __name__ == "__main__":
    main()
