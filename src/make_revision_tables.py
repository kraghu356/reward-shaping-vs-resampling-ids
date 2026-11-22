"""
make_revision_tables.py -- turn the revision result files into LaTeX.

Reads results/rev_*.json (written by nf_revision.py) and, when available, the saved
test predictions and test labels, and writes into --out:

  revnums.tex          \\RV{<spec>}{<quantity>} values used in the running text
  tab_robust.tex       the factorial contrasts under every specification
  tab_thrfree.tex      threshold-free, recalibrated and prior-corrected metrics
  tab_baselines.tex    tuned supervised and tree baselines
  tab_unique.tex       unique training flows per class under each sampler, and run time

Missing results render as a red placeholder, so the manuscript always compiles.

Usage:  python src/make_revision_tables.py --results results --out manuscript/gen \\
            [--preds results/preds --processed data/processed]
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np
from scipy import stats

FAM = ["Normal", "DoS", "Probe", "Access", "Malware", "Other"]
TBD = r"\TBD{}"

# specification key -> (file, head, balanced-sampler level, label, dataset)
SPECS = [
    ("huber1-frozen", "rev_nf_unsw_huber1.json", "val", "frozen", r"Huber $\kappa{=}1$, frozen bootstrap (original)"),
    ("huber1-epoch", "rev_nf_unsw_huber1.json", "val", "epoch", r"Huber $\kappa{=}1$, per-epoch redraw"),
    ("mse-frozen", "rev_nf_unsw_mse.json", "val", "frozen", r"Squared loss, frozen bootstrap"),
    ("mse-epoch", "rev_nf_unsw_mse.json", "val", "epoch", r"Squared loss, per-epoch redraw"),
    ("mse-iw", "rev_nf_unsw_mse.json", "val", "iw", r"Squared loss, importance weights"),
    ("ce-frozen", "rev_nf_unsw_ce.json", "ce", "frozen", r"Weighted cross-entropy, frozen bootstrap"),
    ("ce-epoch", "rev_nf_unsw_ce.json", "ce", "epoch", r"Weighted cross-entropy, per-epoch redraw"),
    ("ce-iw", "rev_nf_unsw_ce.json", "ce", "iw", r"Weighted cross-entropy, importance weights"),
    ("bandit-frozen", "rev_nf_unsw_bandit.json", "bandit", "frozen", r"$\varepsilon$-greedy bandit + replay, frozen"),
    ("bandit-epoch", "rev_nf_unsw_bandit.json", "bandit", "epoch", r"$\varepsilon$-greedy bandit + replay, per-epoch"),
] + [
    (f"alpha{a}", f"rev_sweep_alpha{a}.json", "val", "epoch", rf"Squared loss, $\alpha={a}$ (fixed grid)")
    for a in ("1", "2", "5", "10")
] + [
    (f"kappa{k}", f"rev_sweep_kappa{k}.json", "val", "epoch",
     (r"Huber, $\kappa=(\alpha{+}1)\lambda_{\max}$" if k == "auto" else rf"Huber, $\kappa={k}$") + " (fixed grid)")
    for k in ("0.5", "1", "2", "5", "10", "auto")
] + [
    ("ton-frozen", "rev_nf_ton_mse.json", "val", "frozen", r"NF-ToN-IoT-v2, squared loss, frozen"),
    ("ton-epoch", "rev_nf_ton_mse.json", "val", "epoch", r"NF-ToN-IoT-v2, squared loss, per-epoch"),
    ("bot-frozen", "rev_nf_bot_mse.json", "val", "frozen", r"NF-BoT-IoT-v2, squared loss, frozen"),
    ("bot-epoch", "rev_nf_bot_mse.json", "val", "epoch", r"NF-BoT-IoT-v2, squared loss, per-epoch"),
    ("small2-frozen", "rev_nf_unsw_small2_mse.json", "val", "frozen", r"100\,k draw, distinct seed, frozen"),
    ("small2-epoch", "rev_nf_unsw_small2_mse.json", "val", "epoch", r"100\,k draw, distinct seed, per-epoch"),
    ("trueprior-frozen", "rev_nf_unsw_trueprior_mse.json", "val", "frozen", r"True corpus prior, frozen"),
    ("trueprior-epoch", "rev_nf_unsw_trueprior_mse.json", "val", "epoch", r"True corpus prior, per-epoch"),
    ("widegrid", "rev_nf_unsw_widegrid.json", "val", "epoch", r"Wider grid, squared loss, per-epoch (PB, RB only)"),
    ("widegrid-frozen", "rev_nf_unsw_widegrid.json", "val", "frozen", r"Wider grid, squared loss, frozen (PB, RB only)"),
]
GROUPS = [  # (heading, spec keys) for the robustness table
    ("Loss and clipping (NF-UNSW-NB15-v2, 10 seeds)", ["huber1-frozen", "huber1-epoch", "mse-frozen", "mse-epoch", "mse-iw"]),
    ("Supervised control (10 seeds)", ["ce-frozen", "ce-epoch", "ce-iw"]),
    ("Exploration-based agent (10 seeds)", ["bandit-frozen", "bandit-epoch"]),
    (r"Reward multiplier $\alpha$ and Huber threshold $\kappa$ (5 seeds)",
     [f"alpha{a}" for a in ("1", "2", "5", "10")] + [f"kappa{k}" for k in ("0.5", "1", "2", "5", "10", "auto")]),
    ("Further corpora and draws (10 seeds)",
     ["ton-frozen", "ton-epoch", "bot-frozen", "bot-epoch", "small2-frozen", "small2-epoch",
      "trueprior-frozen", "trueprior-epoch"]),
    ("Wider hyperparameter grid (5 seeds)", ["widegrid-frozen", "widegrid"]),
]


def fnum(x, d=3, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return TBD
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def fp(p):
    if p is None or not np.isfinite(p):
        return TBD
    return r"$<10^{-4}$" if p < 1e-4 else (f"{p:.4f}" if p < 0.01 else f"{p:.3f}")


def signflip(d):
    d = np.asarray(d, float); n = len(d); obs = abs(d.mean())
    if n > 16:
        rs = np.random.RandomState(0); s = rs.choice([-1, 1], size=(200000, n))
        return float((np.abs((s * d).mean(1)) >= obs - 1e-12).mean())
    hits = sum(abs((np.array(sg) * d).mean()) >= obs - 1e-12 for sg in itertools.product([-1, 1], repeat=n))
    return hits / 2 ** n


def ci(d):
    d = np.asarray(d, float); n = len(d)
    if n < 2:
        return (np.nan, np.nan, np.nan)
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(n); tc = stats.t.ppf(0.975, n - 1)
    return m, m - tc * se, m + tc * se


def get(rec, path):
    """Nested lookup; keys may themselves contain dots (e.g. recall_at_fpr_0.001)."""
    cur, parts = rec, path.split(".")
    i = 0
    while i < len(parts):
        for j in range(len(parts), i, -1):          # longest key first
            k = ".".join(parts[i:j])
            if isinstance(cur, dict) and k in cur:
                cur, i = cur[k], j
                break
        else:
            return None
    return cur


class Results:
    def __init__(self, res_dir):
        self.files = {}
        for f in glob.glob(os.path.join(res_dir, "rev_*.json")):
            by = {}
            for r in json.load(open(f)):
                by.setdefault(r["method"], {})[r["seed"]] = r
            self.files[os.path.basename(f)] = by

    def cells(self, fname, head, samp):
        by = self.files.get(fname)
        if not by:
            return None, []
        names = {"PU": f"{head}-plain-none", "RU": f"{head}-freq-none",
                 "PB": f"{head}-plain-{samp}", "RB": f"{head}-freq-{samp}"}
        avail = {k: by.get(n, {}) for k, n in names.items()}
        return avail, names


def contrasts(avail, path="supported_macro_recall"):
    out = {}
    full = [k for k in ("PU", "RU", "PB", "RB") if avail.get(k)]
    if {"PB", "RB"} <= set(full):
        seeds = sorted(set(avail["PB"]) & set(avail["RB"]))
        out["n_pbrb"] = len(seeds)
        d = [get(avail["RB"][s], path) - get(avail["PB"][s], path) for s in seeds]
        out["rbpb"] = ci(d); out["rbpb_pe"] = signflip(d) if len(d) > 1 else np.nan
        out["rbpb_t"] = stats.ttest_1samp(d, 0).pvalue if len(d) > 1 else np.nan
    if len(full) == 4:
        seeds = sorted(set.intersection(*[set(avail[k]) for k in full]))
        out["n"] = len(seeds)
        v = {k: np.array([get(avail[k][s], path) for s in seeds]) for k in full}
        out["means"] = {k: float(v[k].mean()) for k in full}
        out["sampler"] = ci(v["PB"] - v["PU"]); out["reward"] = ci(v["RU"] - v["PU"])
        inter = (v["RB"] - v["PB"]) - (v["RU"] - v["PU"])
        out["inter"] = ci(inter); out["inter_pe"] = signflip(inter)
        out["inter_t"] = stats.ttest_1samp(inter, 0).pvalue
        out["inter_pos"] = int((inter < 0).sum())
    return out


def bootstrap_inter(preds, processed, ds, names, loss, B=1000, tag=""):
    """Test-set Poisson bootstrap CI for the interaction on supported macro recall."""
    if not (preds and processed):
        return None
    yp = os.path.join(processed, f"{ds}_test.y.npy")
    if not os.path.exists(yp):
        return None
    yte = np.load(yp)
    P = {}
    for k, n in names.items():
        fs = [f for f in glob.glob(os.path.join(preds, f"{ds}_{n}_{loss}{tag}_s*.npy"))
              if not f.endswith(("_rescaled.npy", "_priorcorr.npy"))]
        P[k] = {os.path.basename(f).rsplit("_s", 1)[1][:-4]: np.load(f).astype(int) for f in fs}
    if not all(P.values()):
        return None
    seeds = sorted(set.intersection(*[set(v) for v in P.values()]))
    if len(seeds) < 2:
        return None
    rs = np.random.RandomState(0)

    def mrec(y, p, w):
        cm = np.bincount(y * 6 + p, weights=w, minlength=36).reshape(6, 6)
        sup = cm.sum(1) > 0
        return float(np.mean(np.diag(cm)[sup] / cm.sum(1)[sup]))
    vals = []
    for _ in range(B):
        w = rs.poisson(1.0, len(yte)).astype(float)
        m = {k: np.mean([mrec(yte, P[k][s], w) for s in seeds]) for k in P}
        vals.append((m["RB"] - m["PB"]) - (m["RU"] - m["PU"]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preds", default=None)
    ap.add_argument("--processed", default=None)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = Results(a.results)
    nums = {}

    def setnum(spec, key, val):
        nums[f"{spec}@{key}"] = val

    # ---------------- robustness table ----------------
    rows = []
    for heading, keys in GROUPS:
        rows.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{heading}}}}}\\")
        for key in keys:
            spec = next(s for s in SPECS if s[0] == key)
            _, fname, head, samp, label = spec
            avail, names = R.cells(fname, head, samp)
            c = contrasts(avail) if avail else {}
            f1 = contrasts(avail, "macro_f1_supported") if avail else {}
            fpr = contrasts(avail, "false_positive_rate") if avail else {}
            loss = "huber" if key.startswith(("huber", "kappa")) else "mse"
            boot = None
            if key in ("huber1-frozen", "mse-frozen", "mse-epoch", "mse-iw", "ce-epoch", "bandit-epoch") and avail:
                ds = "nf_unsw"
                boot = bootstrap_inter(a.preds, a.processed, ds, names, loss, a.boot)
            n = c.get("n", c.get("n_pbrb"))
            s_ = c.get("sampler", (None,) * 3); r_ = c.get("reward", (None,) * 3)
            rb = c.get("rbpb", (None,) * 3); it = c.get("inter", (None,) * 3)
            fpr_ratio = None
            if avail and avail.get("PB") and avail.get("RB"):
                pbv = np.mean([get(r, "false_positive_rate") for r in avail["PB"].values()])
                rbv = np.mean([get(r, "false_positive_rate") for r in avail["RB"].values()])
                fpr_ratio = rbv / pbv if pbv else None
            pbrb_only = key.startswith("widegrid")
            dash = lambda v: "--" if (pbrb_only and v == TBD) else v
            cells = [
                label,
                str(n) if n else TBD,
                dash(fnum(s_[0])), dash(fnum(r_[0])),
                (fnum(rb[0]) + (rf" [{fnum(rb[1])},{fnum(rb[2])}]" if rb[0] is not None else "")),
                (dash(fnum(it[0])) + (rf" [{fnum(it[1])},{fnum(it[2])}]" if it[0] is not None else "")
                 + (rf"$^{{b}}$[{fnum(boot[0])},{fnum(boot[1])}]" if boot else "")),
                fp(c.get("inter_pe")) if "inter_pe" in c else ("--" if pbrb_only else TBD),
                fnum(f1.get("rbpb", (None,))[0]) + " / " + (f"{fpr_ratio:.2f}" if fpr_ratio else TBD),
            ]
            rows.append(" & ".join(cells) + r" \\")
            for k, v in (("sampler", s_[0]), ("reward", r_[0]), ("rbpb", rb[0]), ("inter", it[0]),
                         ("interlo", it[1]), ("interhi", it[2]), ("rbpblo", rb[1]), ("rbpbhi", rb[2])):
                setnum(key, k, fnum(v))
            setnum(key, "pexact", fp(c.get("inter_pe")) if "inter_pe" in c else TBD)
            setnum(key, "pt", fp(c.get("inter_t")) if "inter_t" in c else TBD)
            setnum(key, "n", str(n) if n else TBD)
            setnum(key, "neg", f"{c['inter_pos']}/{c['n']}" if "inter_pos" in c else TBD)
            setnum(key, "f1rbpb", fnum(f1.get("rbpb", (None,))[0]))
            setnum(key, "fprratio", f"{fpr_ratio:.2f}" if fpr_ratio else TBD)
            if boot:
                setnum(key, "bootlo", fnum(boot[0])); setnum(key, "boothi", fnum(boot[1]))
            if "means" in c:
                for k2, v2 in c["means"].items():
                    setnum(key, f"mr{k2}", fnum(v2, 4, False))
    with open(os.path.join(a.out, "tab_robust.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")
    # split into the manuscript's three tables: confirmatory, exploratory, sensitivity
    groups, cur = {}, None
    for r in rows:
        if r.startswith("\\multicolumn"):
            cur = r; groups[cur] = [r]
        elif cur:
            groups[cur].append(r.replace("100\\,k draw, distinct seed", "100\\,k draw, separate seed")
                                 .replace("(original)", "(baseline)"))
    parts = {"tab_robustA.tex": ("Loss and clipping",),
             "tab_robustB.tex": ("Supervised", "Exploration", "Further corpora"),
             "tab_sens.tex": ("Reward multiplier", "Wider")}
    for fn, keys in parts.items():
        with open(os.path.join(a.out, fn), "w") as f:
            f.write("\n".join(l for k, v in groups.items() if any(s in k for s in keys) for l in v) + "\n")

    # ---------------- threshold-free / recalibrated / prior-corrected ----------------
    rows = []
    for key in ("huber1-frozen", "mse-frozen", "mse-epoch"):
        spec = next(s for s in SPECS if s[0] == key)
        _, fname, head, samp, label = spec
        avail, names = R.cells(fname, head, samp)
        rows.append(rf"\multicolumn{{9}}{{l}}{{\emph{{{label}}}}}\\")
        for cond in ("PU", "RU", "PB", "RB"):
            recs = list((avail or {}).get(cond, {}).values())

            def m(path, d=3):
                v = [get(r, path) for r in recs]
                v = [x for x in v if x is not None]
                return fnum(float(np.mean(v)), d, False) if v else TBD
            sq = not key.startswith("huber")
            pc = m("prior_corrected.macro_f1_supported") if (cond in ("RU", "RB") and sq) else "--"
            pcf = m("prior_corrected.false_positive_rate", 4) if (cond in ("RU", "RB") and sq) else "--"
            row = [cond, m("threshold_free.macro_auprc"), m("threshold_free.binary_auprc"),
                   m("threshold_free.recall_at_fpr_0.001"), m("threshold_free.recall_at_fpr_0.01"),
                   m("recalibrated.macro_f1_supported"), m("recalibrated.false_positive_rate", 4), pc, pcf]
            rows.append(" & ".join(row) + r" \\")
            for k, v in zip(("mauprc", "bauprc", "r1e3", "r1e2", "recalf1", "recalfpr", "pcf1", "pcfpr"), row[1:]):
                setnum(f"{key}-{cond}", k, v)
        # differences RB-PB on threshold-free metrics, paired
        for path, k in (("threshold_free.macro_auprc", "dmauprc"), ("recalibrated.macro_f1_supported", "drecalf1"),
                        ("threshold_free.binary_auprc", "dbauprc")):
            if avail and avail.get("PB") and avail.get("RB"):
                seeds = sorted(set(avail["PB"]) & set(avail["RB"]))
                d = [get(avail["RB"][s], path) - get(avail["PB"][s], path) for s in seeds
                     if get(avail["RB"][s], path) is not None]
                if len(d) > 1:
                    m_, lo, hi = ci(d)
                    setnum(key, k, fnum(m_, 4)); setnum(key, k + "ci", f"[{fnum(lo, 4)},{fnum(hi, 4)}]")
                    setnum(key, k + "pe", fp(signflip(d)))
    with open(os.path.join(a.out, "tab_thrfree.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")

    # ---------------- lambda-rescaling agreement (population-rule test) ----------------
    if a.preds:
        for samp in ("frozen", "epoch"):
            ag, base = [], []
            for f in glob.glob(os.path.join(a.preds, f"nf_unsw_val-plain-{samp}_mse_s*_rescaled.npy")):
                s = os.path.basename(f).split("_s")[-1].split("_")[0]
                g = os.path.join(a.preds, f"nf_unsw_val-freq-{samp}_mse_s{s}.npy")
                p0 = os.path.join(a.preds, f"nf_unsw_val-plain-{samp}_mse_s{s}.npy")
                if os.path.exists(g) and os.path.exists(p0):
                    Rr, Fq, P0 = np.load(f), np.load(g), np.load(p0)
                    ag.append((Rr == Fq).mean()); base.append((P0 == Fq).mean())
            if ag:
                setnum(f"mse-{samp}", "agree", f"{100*np.mean(ag):.1f}\\%")
                setnum(f"mse-{samp}", "agreebase", f"{100*np.mean(base):.1f}\\%")

    # ---------------- baselines ----------------
    rows = []
    base = [
        ("PB, squared loss, per-epoch", "rev_nf_unsw_mse.json", "val-plain-epoch", None),
        ("Cross-entropy (unweighted)", "rev_nf_unsw_ce.json", "ce-plain-none", None),
        ("Cross-entropy + post-hoc logit adjustment", "rev_nf_unsw_ce.json", "ce-plain-none", "logit_adjusted"),
        (r"Class-weighted cross-entropy ($-\log\pi_c$)", "rev_nf_unsw_ce.json", "ce-freq-none", None),
        ("Cross-entropy + balanced sampler (per-epoch)", "rev_nf_unsw_ce.json", "ce-plain-epoch", None),
        ("Extra Trees (tuned)", "rev_nf_unsw_ce.json", "extra_trees", None),
        ("Random Forest (tuned)", "rev_nf_unsw_ce.json", "rf", None),
        ("Gradient-boosted trees (tuned)", "rev_nf_unsw_ce.json", "xgboost", None),
    ]
    for label, fname, meth, sub in base:
        recs = list(R.files.get(fname, {}).get(meth, {}).values())

        def m(path):
            v = [get(r, (sub + "." if sub else "") + path) for r in recs]
            v = [x for x in v if x is not None]
            return (f"{np.mean(v):.4f} $\\pm$ {np.std(v, ddof=1):.4f}" if len(v) > 1 else TBD)
        rows.append(" & ".join([label, str(len(recs)) if recs else TBD, m("supported_macro_recall"),
                                m("macro_f1_supported"), m("detection_f1"), m("false_positive_rate")]) + r" \\")
    with open(os.path.join(a.out, "tab_baselines.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")

    # ---------------- unique flows and run time ----------------
    rows = []
    for label, fname, meth in [("natural", "rev_nf_unsw_mse.json", "val-plain-none"),
                               ("frozen", "rev_nf_unsw_mse.json", "val-plain-frozen"),
                               ("per-epoch", "rev_nf_unsw_mse.json", "val-plain-epoch"),
                               ("imp. weights", "rev_nf_unsw_mse.json", "val-plain-iw"),
                               ("bandit", "rev_nf_unsw_bandit.json", "bandit-plain-epoch")]:
        recs = list(R.files.get(fname, {}).get(meth, {}).values())
        if recs and recs[0].get("unique_seen_per_class"):
            u = np.mean([r["unique_seen_per_class"] for r in recs], 0)
            t = np.mean([r["train_per_class"] for r in recs], 0)
            sec = np.mean([r.get("seconds", np.nan) for r in recs])
            cells = [label] + [f"{100*x/y:.0f}" for x, y in zip(u, t)] + [f"{sec/60:.1f}"]
            setnum(meth, "normalseen", f"{100*u[0]/t[0]:.0f}\\%")
        else:
            cells = [label] + [TBD] * 7
        rows.append(" & ".join(cells) + r" \\")
    with open(os.path.join(a.out, "tab_unique.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")

    # ---------------- replication overlap ----------------
    if a.processed:
        fa = [os.path.join(a.processed, f"nf_unsw_regen_{sp}.rows.npy") for sp in ("train", "test")]
        fb = [os.path.join(a.processed, f"nf_unsw_small2_{sp}.rows.npy") for sp in ("train", "test")]
        if all(os.path.exists(f) for f in fa + fb):
            ra = np.concatenate([np.load(f) for f in fa]); rb = np.concatenate([np.load(f) for f in fb])
            setnum("overlap", "pct", f"{100*np.intersect1d(ra, rb).size/len(rb):.1f}\\%")
        meta = os.path.join(a.processed, "nf_unsw_regen_train.meta.json")
        if os.path.exists(meta):
            cc = json.load(open(meta)).get("corpus_counts", {})
            if cc.get("Malware"):
                setnum("corpus", "ratio", f"{cc['Normal']/cc['Malware']:,.0f}".replace(",", "\\,"))

    # ---------------- numbers file ----------------
    with open(os.path.join(a.out, "revnums.tex"), "w") as f:
        f.write("% generated by make_revision_tables.py -- do not edit\n")
        for k, v in sorted(nums.items()):
            if v != TBD:
                f.write(f"\\expandafter\\def\\csname rv@{k}\\endcsname{{{v}}}\n")
    print(f"wrote tables to {a.out}; {sum(v != TBD for v in nums.values())} values filled")


if __name__ == "__main__":
    main()
