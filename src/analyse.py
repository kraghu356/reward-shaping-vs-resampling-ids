"""
analyse.py -- regenerate every statistic reported in the paper from the
per-seed result files.  Nothing here is hard-coded: all numbers are read from
results/*.json, so a reader can check the published tables against a fresh run.

Usage:  python src/analyse.py
"""
import json
import os

import numpy as np
from scipy import stats

FAM = ["Normal", "DoS", "Probe", "Access", "Malware", "Other"]
COND = {"dqn_plain_uniform": "PU", "dqn_reward_uniform": "RU",
        "dqn_plain_balanced": "PB", "dqn_reward_balanced": "RB",
        "extra_trees": "ET"}
RES = "results"


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        print(f"  [skip] {p} not found")
        return None
    return json.load(open(p))


def by_cond(recs, field, seeds=None):
    """{condition: array of per-seed values, ordered by seed}"""
    out = {}
    for r in recs:
        c = COND.get(r["method"], r["method"])
        out.setdefault(c, {})[r["seed"]] = r[field]
    ordered = {}
    for c, d in out.items():
        ks = sorted(d) if seeds is None else [s for s in seeds if s in d]
        ordered[c] = np.array([d[k] for k in ks])
    return ordered


def paired(a, b, label):
    d = a - b
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    t, p = stats.ttest_rel(a, b) if n > 1 else (float("nan"),) * 2
    tc = stats.t.ppf(0.975, n - 1) if n > 1 else float("nan")
    print(f"  {label:38s} {m:+.4f}  95% CI [{m-tc*se:+.4f}, {m+tc*se:+.4f}]  "
          f"t={t:+.2f} p={p:.5f}  n={n}")
    return m


def factorial(recs, field, title):
    print(f"\n--- {title} ---")
    D = by_cond(recs, field)
    need = ("PU", "RU", "PB", "RB")
    if not all(k in D for k in need):
        print("  incomplete conditions:", sorted(D)); return
    for c in need + ("ET",):
        if c in D:
            print(f"  {c:3s} {D[c].mean():.4f} +- {D[c].std(ddof=1):.4f}   "
                  f"per-seed {np.round(D[c],4)}")
    print(f"\n  2x2:            uniform  balanced")
    print(f"    plain reward  {D['PU'].mean():.4f}   {D['PB'].mean():.4f}")
    print(f"    freq  reward  {D['RU'].mean():.4f}   {D['RB'].mean():.4f}")
    print()
    paired(D["PB"], D["PU"], "sampler alone (PB-PU)")
    paired(D["RU"], D["PU"], "reward alone (RU-PU)")
    paired(D["RB"], D["PB"], "reward added to sampler (RB-PB)")
    inter = (D["RB"] - D["PB"]) - (D["RU"] - D["PU"])
    n = len(inter); m = inter.mean()
    se = inter.std(ddof=1) / np.sqrt(n); tc = stats.t.ppf(0.975, n - 1)
    t, p = stats.ttest_1samp(inter, 0)
    print(f"  {'interaction (RB-PB)-(RU-PU)':38s} {m:+.4f}  "
          f"95% CI [{m-tc*se:+.4f}, {m+tc*se:+.4f}]  t={t:+.2f} p={p:.5f}  n={n}")
    if "ET" in D:
        print()
        paired(D["ET"], D["PB"], "Extra Trees - PB")


def effective_weights(meta_path):
    """Table 2: effective per-class weight from the actual class counts."""
    if not os.path.exists(meta_path):
        print(f"\n  [skip] {meta_path} not found; run prepare_nf.py first")
        return
    meta = json.load(open(meta_path))
    counts = np.array([meta["class_counts"][f] for f in FAM], float)
    keep = counts > 0
    N = counts.sum()
    lam = np.where(keep, np.log(np.divide(N, counts, out=np.ones_like(counts), where=keep)), 0.0)
    pi = counts / N
    K = int(keep.sum())
    print("\n--- Table 2: effective per-class weight (relative to Normal) ---")
    print(f"  {'family':9s} {'N_c':>9s} {'pi_c':>9s} {'lambda_c':>9s}")
    for i, f in enumerate(FAM):
        print(f"  {f:9s} {int(counts[i]):9d} {pi[i]:9.6f} {lam[i]:9.4f}")
    cfg = {"PU (natural, plain)": pi,
           "RU (natural, frequency)": pi * lam,
           "PB (balanced, plain)": np.where(keep, 1.0 / K, 0.0),
           "RB (balanced, frequency)": np.where(keep, lam / K, 0.0)}
    print(f"\n  {'condition':26s} " + " ".join(f"{f:>9s}" for f in FAM))
    for k, w in cfg.items():
        r = w / w[0] if w[0] else w
        print(f"  {k:26s} " + " ".join(f"{v:9.2f}" for v in r))
    u = np.where(keep, 1.0 / K, 0.0)
    ru = pi * lam; ru = ru / ru.sum()
    print(f"\n  RU total-variation distance from uniform: "
          f"{0.5*np.abs(ru-u).sum():.5f}")
    print(f"  RB upper-bound emphasis, rarest vs majority: "
          f"{(lam[keep].max()/lam[0]):.1f}x  (unclipped targets)")


def per_family(recs, title):
    print(f"\n--- {title}: per-family recall and F1 (mean over seeds) ---")
    for m in sorted({r["method"] for r in recs}):
        rs = [r for r in recs if r["method"] == m]
        rec = np.nanmean([[np.nan if v is None else v for v in r["per_class_recall"]] for r in rs], 0)
        f1 = np.nanmean([[np.nan if v is None else v for v in r["per_class_f1"]] for r in rs], 0)
        print(f"  {COND.get(m,m):3s} rec " + " ".join(f"{v:.3f}" for v in rec))
        print(f"      f1  " + " ".join(f"{v:.3f}" for v in f1))


def main():
    print("=" * 78)
    print("Reward Shaping or Resampling? -- all reported statistics")
    print("=" * 78)

    effective_weights("data/processed/nf_unsw_train.meta.json")

    tuned = load("nf_unsw_tuned.json")
    if tuned:
        factorial(tuned, "supported_macro_recall",
                  "Tables 6-7: PRIMARY endpoint, validation-tuned, 400k")
        factorial(tuned, "macro_f1_supported", "Secondary: macro F1")
        factorial(tuned, "detection_precision", "Secondary: detection precision")
        factorial(tuned, "detection_recall", "Secondary: detection recall")
        factorial(tuned, "false_positive_rate", "Secondary: false-positive rate")
        per_family(tuned, "Table 9")

    full = load("nf_unsw_full.json")
    if full:
        factorial(full, "supported_macro_recall",
                  "Tables 5/13: fixed-budget pass, 400k")

    small = load("nf_unsw_small_tuned.json")
    if small:
        factorial(small, "supported_macro_recall",
                  "Table 10: independent 100k replication")
        factorial(small, "detection_precision", "Table 10: 100k precision")

    print("\nAll contrasts above are computed from per-seed values in results/.")


if __name__ == "__main__":
    main()
