"""
rev_analyse.py -- statistics for the revision experiments (nf_revision.py output).

For every 2x2 (head x sampler) found in the file it reports, per metric:
  sampler effect, reward/weight effect, weight added to sampler, interaction,
  each with paired-t 95% CI and the exact two-sided sign-flip randomisation p.
Metrics: argmax operating point, threshold-free, and validation-recalibrated.
Also: unique flows seen per class, and (with --preds) the agreement between
lambda-rescaled plain-reward predictions and the frequency-reward model (R2 test).

Usage:  python src/rev_analyse.py results/rev_nf_unsw_mse.json [--preds results/preds]
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np
from scipy import stats

FAM = ["Normal", "DoS", "Probe", "Access", "Malware", "Other"]
METRICS = [
    ("supported_macro_recall", lambda r: r["supported_macro_recall"]),
    ("macro_f1", lambda r: r["macro_f1_supported"]),
    ("detection_f1", lambda r: r["detection_f1"]),
    ("FPR", lambda r: r["false_positive_rate"]),
    ("macro_AUPRC", lambda r: r["threshold_free"]["macro_auprc"]),
    ("binary_AUPRC", lambda r: r["threshold_free"]["binary_auprc"]),
    ("binary_AUROC", lambda r: r["threshold_free"]["binary_auroc"]),
    ("recal_macro_recall", lambda r: r["recalibrated"]["supported_macro_recall"]),
    ("recal_macro_f1", lambda r: r["recalibrated"]["macro_f1_supported"]),
    ("recal_FPR", lambda r: r["recalibrated"]["false_positive_rate"]),
    ("recall@FPR=1e-3", lambda r: r["threshold_free"]["recall_at_fpr_0.001"]),
    ("recall@FPR=1e-2", lambda r: r["threshold_free"]["recall_at_fpr_0.01"]),
]


def signflip_p(d):
    """Exact two-sided sign-flip randomisation test on paired differences."""
    d = np.asarray(d, float); n = len(d)
    obs = abs(d.mean())
    if n > 20:                                   # Monte Carlo beyond 2^20
        rs = np.random.RandomState(0)
        s = rs.choice([-1, 1], size=(200000, n))
        return float((np.abs((s * d).mean(1)) >= obs - 1e-12).mean())
    tot = 0; hit = 0
    for signs in itertools.product([-1, 1], repeat=n):
        tot += 1; hit += abs((np.array(signs) * d).mean()) >= obs - 1e-12
    return hit / tot


def line(label, d):
    d = np.asarray(d, float); n = len(d); m = d.mean()
    if n < 2:
        print(f"    {label:34s} {m:+.4f}  (n={n})"); return
    se = d.std(ddof=1) / np.sqrt(n); tc = stats.t.ppf(0.975, n - 1)
    p = stats.ttest_1samp(d, 0).pvalue
    print(f"    {label:34s} {m:+.4f}  CI [{m-tc*se:+.4f}, {m+tc*se:+.4f}]  "
          f"p_t={p:.4g}  p_exact={signflip_p(d):.4g}  +{int((d>0).sum())}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--preds", default=None)
    ap.add_argument("--processed", default=None, help="folder with <dataset>_test.y.npy (bootstrap)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--overlap", nargs=2, default=None, metavar=("PREFIX_A", "PREFIX_B"),
                    help="report source-row overlap between two prepared datasets")
    a = ap.parse_args()

    if a.overlap:
        ra = np.concatenate([np.load(f"{a.overlap[0]}_{sp}.rows.npy") for sp in ("train", "test")])
        rb = np.concatenate([np.load(f"{a.overlap[1]}_{sp}.rows.npy") for sp in ("train", "test")])
        inter = np.intersect1d(ra, rb).size
        print(f"source-row overlap: {inter} rows = {100*inter/len(rb):.1f}% of {a.overlap[1]} "
              f"({len(rb)}) and {100*inter/len(ra):.1f}% of {a.overlap[0]} ({len(ra)})")

    for path in a.files:
        recs = json.load(open(path))
        print(f"\n==================== {os.path.basename(path)} ====================")
        by = {}
        for r in recs:
            by.setdefault(r["method"], {})[r["seed"]] = r
        ds = recs[0]["dataset"]; loss = recs[0].get("loss")

        for head in ("val", "ce", "bandit"):
            for samp in ("frozen", "epoch", "iw"):
                names = {k: f"{head}-{w}-{s}" for k, w, s in
                         [("PU", "plain", "none"), ("RU", "freq", "none"),
                          ("PB", "plain", samp), ("RB", "freq", samp)]}
                if not all(n in by for n in names.values()):
                    continue
                seeds = sorted(set.intersection(*[set(by[n]) for n in names.values()]))
                print(f"\n--- {head} head, balanced = {samp}  (n={len(seeds)} paired seeds) ---")
                for mname, get in METRICS:
                    try:
                        V = {k: np.array([get(by[n][s]) for s in seeds]) for k, n in names.items()}
                    except (KeyError, TypeError):
                        continue
                    print(f"  {mname}:  PU {V['PU'].mean():.4f}  RU {V['RU'].mean():.4f}  "
                          f"PB {V['PB'].mean():.4f}  RB {V['RB'].mean():.4f}")
                    line("sampler alone (PB-PU)", V["PB"] - V["PU"])
                    line("weight alone (RU-PU)", V["RU"] - V["PU"])
                    line("weight on sampler (RB-PB)", V["RB"] - V["PB"])
                    line("interaction", (V["RB"] - V["PB"]) - (V["RU"] - V["PU"]))

        # unique flows seen
        print("\n--- unique training flows seen per class (mean over seeds) ---")
        print("    " + " ".join(f"{f:>9s}" for f in FAM))
        for m, d in sorted(by.items()):
            u = [r.get("unique_seen_per_class") for r in d.values()]
            if u and u[0]:
                tot = np.mean([r["train_per_class"] for r in d.values()], 0)
                um = np.mean(u, 0)
                print(f"  {m:18s} " + " ".join(f"{x:9.0f}" for x in um) +
                      "   (% of class: " + " ".join(f"{100*x/t:.0f}" for x, t in zip(um, tot)) + ")")

        # baselines
        for m in ("ce-plain-none", "extra_trees", "rf", "xgboost"):
            if m in by:
                v = [r["supported_macro_recall"] for r in by[m].values()]
                f = [r["macro_f1_supported"] for r in by[m].values()]
                extra = ""
                la = [r["logit_adjusted"]["supported_macro_recall"] for r in by[m].values() if "logit_adjusted" in r]
                if la:
                    extra = f"  | logit-adjusted supMacro {np.mean(la):.4f}"
                print(f"  {m:14s} supMacro {np.mean(v):.4f}  macroF1 {np.mean(f):.4f}{extra}")

        # prior-corrected frequency-reward models (reviewer Mo3)
        for m in sorted(by):
            pcs = [r["prior_corrected"] for r in by[m].values() if "prior_corrected" in r]
            if pcs:
                base = [r for r in by[m].values() if "prior_corrected" in r]
                print(f"  prior-corrected {m:20s} supMacro {np.mean([b['supported_macro_recall'] for b in base]):.4f}"
                      f" -> {np.mean([p['supported_macro_recall'] for p in pcs]):.4f} | macroF1 "
                      f"{np.mean([b['macro_f1_supported'] for b in base]):.4f} -> "
                      f"{np.mean([p['macro_f1_supported'] for p in pcs]):.4f} | FPR "
                      f"{np.mean([b['false_positive_rate'] for b in base]):.4f} -> "
                      f"{np.mean([p['false_positive_rate'] for p in pcs]):.4f}")

        # test-set bootstrap over the 2x2 contrasts (reviewer M5)
        if a.preds and a.processed:
            yte = np.load(f"{a.processed}/{ds}_test.y.npy")
            for head in ("val", "ce", "bandit"):
                for samp in ("frozen", "epoch", "iw"):
                    names = {"PU": f"{head}-plain-none", "RU": f"{head}-freq-none",
                             "PB": f"{head}-plain-{samp}", "RB": f"{head}-freq-{samp}"}
                    P = {}
                    for k, n in names.items():
                        fs = sorted(glob.glob(f"{a.preds}/{ds}_{n}_{loss}_s*.npy"))
                        fs = [f for f in fs if not f.endswith(("_rescaled.npy", "_priorcorr.npy"))]
                        P[k] = {f.split("_s")[-1][:-4]: np.load(f) for f in fs}
                    seeds = sorted(set.intersection(*[set(v) for v in P.values()])) if all(P.values()) else []
                    if len(seeds) < 2:
                        continue
                    print(f"\n--- test-set bootstrap (B={a.boot}), {head} head, balanced={samp}, n_seeds={len(seeds)} ---")
                    rs = np.random.RandomState(0)
                    def mrec(y, p, w):
                        cm = np.bincount(y * 6 + p, weights=w, minlength=36).reshape(6, 6)
                        sup = cm.sum(1) > 0
                        return float(np.mean(np.diag(cm)[sup] / cm.sum(1)[sup]))
                    out = {"sampler": [], "weight": [], "weight_on_sampler": [], "interaction": []}
                    for _ in range(a.boot):
                        w = rs.poisson(1.0, len(yte)).astype(float)
                        v = {k: np.mean([mrec(yte, P[k][s].astype(int), w) for s in seeds]) for k in P}
                        out["sampler"].append(v["PB"] - v["PU"]); out["weight"].append(v["RU"] - v["PU"])
                        out["weight_on_sampler"].append(v["RB"] - v["PB"])
                        out["interaction"].append((v["RB"] - v["PB"]) - (v["RU"] - v["PU"]))
                    for k, d in out.items():
                        d = np.array(d)
                        print(f"    supported macro recall, {k:18s} mean {d.mean():+.4f}  "
                              f"95% bootstrap CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")

        # R2 mechanism test: lambda-rescaled plain vs freq predictions
        if a.preds:
            print("\n--- lambda-rescaling test (agreement with frequency-reward model) ---")
            for samp in ("none", "frozen", "epoch"):
                ag, base = [], []
                for f in glob.glob(f"{a.preds}/{ds}_val-plain-{samp}_{loss}_s*_rescaled.npy"):
                    s = f.split("_s")[-1].split("_")[0]
                    g = f"{a.preds}/{ds}_val-freq-{samp}_{loss}_s{s}.npy"
                    p0 = f"{a.preds}/{ds}_val-plain-{samp}_{loss}_s{s}.npy"
                    if os.path.exists(g) and os.path.exists(p0):
                        R, Fq, P0 = np.load(f), np.load(g), np.load(p0)
                        ag.append((R == Fq).mean()); base.append((P0 == Fq).mean())
                if ag:
                    print(f"  sampler={samp:6s} rescaled-plain vs freq: {np.mean(ag):.4f}   "
                          f"(unrescaled plain vs freq: {np.mean(base):.4f}, n={len(ag)})")


if __name__ == "__main__":
    main()
