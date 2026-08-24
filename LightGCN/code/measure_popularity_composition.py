"""
measure_popularity_composition.py

Quantifies, per dataset, how much of an item's training popularity is EARNED
(driven by genuine quality) versus AMPLIFIED (driven by exposure / self-selection).

Why this exists
---------------
The thesis goal treats Average Popularity@20 as a quantity to minimise. That is
only correct when popularity is an artifact of exposure. Where popularity is
genuine quality signal, minimising it is minimising accuracy. Zhao et al.
(IEEE TKDE 2022, arXiv:2109.07946) call these the harmful and benign components
of popularity bias. This script measures the split instead of assuming it.

Method
------
Coat, Yahoo!R3 and KuaiRec all ship a test split collected under (near-)uniform
exposure -- Coat and Yahoo by randomly sampling items to rate, KuaiRec by being a
fully observed user-item matrix. Under uniform exposure an item's positive rate
is an unbiased estimate of its quality, with no exposure confound.

So:  quality_i   = P(positive | randomly shown)      from the unbiased split
     degree_i    = training interactions             from the biased log
     earned      = R^2 of  log(1+degree) ~ quality
     amplified   = 1 - earned

Gowalla has no random-exposure split and therefore cannot be measured; it is
reported as unknown rather than guessed.

Run:  python measure_popularity_composition.py
Out:  results/popularity_composition.json
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                   "popularity_composition.json")

REL_THRESHOLD = 4  # Coat / Yahoo ratings are 1-5; >= 4 counts as a positive


def _read_lists(path):
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            out[int(parts[0])] = [int(x) for x in parts[1:]]
    return out


def _degrees(lists, n_items):
    deg = np.zeros(n_items)
    for items in lists.values():
        for i in items:
            if 0 <= i < n_items:
                deg[i] += 1
    return deg


def _fit(quality, degree, mask):
    """R^2 of log(1+degree) regressed on quality, over the masked items."""
    y = np.log1p(degree[mask])
    x = quality[mask]
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    fit = A @ coef
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ((y - fit) ** 2).sum() / ss_tot if ss_tot > 0 else 0.0
    return float(r2), float(spearmanr(x, y)[0])


def measure_from_ascii(name, train_ascii, test_ascii, min_exposures=3):
    """Coat-style: two dense rating matrices, test collected under uniform exposure."""
    tr = np.loadtxt(train_ascii)
    te = np.loadtxt(test_ascii)
    degree = (tr > 0).sum(0).astype(float)
    shown = (te > 0).sum(0).astype(float)
    liked = (te >= REL_THRESHOLD).sum(0).astype(float)
    quality = np.divide(liked, shown, out=np.zeros_like(liked), where=shown > 0)
    mask = (degree > 0) & (shown >= min_exposures)
    r2, rho = _fit(quality, degree, mask)
    return dict(dataset=name, unbiased_source=os.path.basename(test_ascii),
                items_scored=int(mask.sum()), earned_r2=r2, spearman_quality_degree=rho)


def measure_from_lists(name, train_txt, test_txt, n_items, min_exposures=1):
    """KuaiRec-style: the test split is (near-)fully observed, so an item's
    positive count over the test population estimates its quality."""
    tr = _read_lists(train_txt)
    te = _read_lists(test_txt)
    degree = _degrees(tr, n_items)
    positives = _degrees(te, n_items)
    n_test_users = max(1, len(te))
    quality = positives / n_test_users
    mask = (degree > 0) & (positives >= min_exposures)
    r2, rho = _fit(quality, degree, mask)
    return dict(dataset=name, unbiased_source=os.path.basename(test_txt),
                items_scored=int(mask.sum()), earned_r2=r2, spearman_quality_degree=rho)


def main():
    rows = []

    rows.append(measure_from_ascii(
        "Coat",
        os.path.join(DATA, "Coat", "main", "train.ascii"),
        os.path.join(DATA, "Coat", "main", "test.ascii")))

    rows.append(measure_from_lists(
        "KuaiRec",
        os.path.join(DATA, "KuaiRec", "train.txt"),
        os.path.join(DATA, "KuaiRec", "test.txt"),
        n_items=10728))

    rows.append(measure_from_lists(
        "Yahoo",
        os.path.join(DATA, "Yahoo", "train.txt"),
        os.path.join(DATA, "Yahoo", "test.txt"),
        n_items=1000))

    rows.append(dict(dataset="Gowalla", unbiased_source=None, items_scored=0,
                     earned_r2=None, spearman_quality_degree=None,
                     note="no random-exposure split ships with Gowalla; "
                          "the composition cannot be measured and is not guessed"))

    print(f"{'dataset':10s} {'items':>7s} {'earned R^2':>11s} {'rho(q,deg)':>11s}  reading")
    for r in rows:
        if r["earned_r2"] is None:
            print(f"{r['dataset']:10s} {'-':>7s} {'unmeasurable':>11s} {'-':>11s}  {r['note']}")
            continue
        share = r["earned_r2"] * 100
        reading = ("popularity is largely EARNED -- debiasing must cost accuracy"
                   if share >= 25 else
                   "popularity is largely AMPLIFIED -- debiasing has room to help")
        print(f"{r['dataset']:10s} {r['items_scored']:7d} {share:10.1f}% "
              f"{r['spearman_quality_degree']:+11.4f}  {reading}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"relevance_threshold": REL_THRESHOLD, "measurements": rows}, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
