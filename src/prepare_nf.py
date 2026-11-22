"""
prepare_nf.py -- preprocessing for the standardised NetFlow (NF-v2) datasets.

WHY THIS EXISTS
The original pipeline projected three heterogeneous benchmarks onto a
hand-built 10-feature schema.  Two problems followed: features had to be
zero-filled where a dataset lacked them, and a mapping mistake (UNSW 'rate'
is a packet rate, 'synack' is a timing interval) aligned semantically
different columns across datasets.  Both manufacture domain shift, which is
fatal for a cross-dataset study.

The NF-v2 datasets of Sarhan, Layeghy & Portmann ("Towards a Standard
Feature Set for NIDS Datasets", Mobile Networks and Applications, 2022)
solve this at the source: NF-UNSW-NB15-v2, NF-ToN-IoT-v2, NF-BoT-IoT-v2 and
NF-CSE-CIC-IDS2018-v2 all share the SAME 43 extended NetFlow features.
No projection, no zero-fill, no mapping to invent.

LEAKAGE GUARD
Sarhan et al. drop the TTL-based features (sttl, dttl, ct_state_ttl,
MIN_TTL, MAX_TTL) because of their extreme correlation with the labels, and
drop flow identifiers (IPs, ports, timestamps) to stop the model keying on
attacker/victim endpoints.  We do the same, by default and explicitly.
Reported accuracies that keep these features are not comparable to ours.

Usage
    python prepare_nf.py --input NF-UNSW-NB15-v2.csv --dataset nf_unsw \
        --out data/processed/nf_unsw --cap 400000 --seed 42
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Coarse six-family taxonomy, shared across every NF dataset.
# Keys are lower-cased NF 'Attack' values with spaces/hyphens stripped.
# --------------------------------------------------------------------------
CANONICAL_LABELS = ["Normal", "DoS", "Probe", "Access", "Malware", "Other"]

NF_LABEL_MAP = {
    "benign": "Normal",

    "dos": "DoS",
    "ddos": "DoS",

    "reconnaissance": "Probe",
    "scanning": "Probe",
    "analysis": "Probe",

    "exploits": "Access",
    "backdoor": "Access",
    "injection": "Access",
    "password": "Access",
    "bruteforce": "Access",
    "bruteforce web": "Access",
    "xss": "Access",
    "mitm": "Access",
    "webattacks": "Access",
    "infilteration": "Access",
    "infiltration": "Access",

    "bot": "Malware",
    "worms": "Malware",
    "ransomware": "Malware",
    "shellcode": "Malware",
    "theft": "Malware",

    "fuzzers": "Other",
    "generic": "Other",
}

# Flow identifiers: keying on these learns the testbed, not the attack.
ID_COLS = [
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT",
    "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS",
    "Dataset", "dataset",
]
# Label columns (never features).
LABEL_COLS = ["Label", "Attack", "label", "attack"]
# Label-leaking features per Sarhan et al.
LEAKY_COLS = ["MIN_TTL", "MAX_TTL", "sttl", "dttl", "ct_state_ttl", "TTL"]


def _norm(s):
    return str(s).strip().lower().replace("-", "").replace("_", "").replace("  ", " ")


def map_labels(attack_series):
    out, unmapped = [], {}
    for raw in attack_series.astype(str):
        key = _norm(raw)
        fam = NF_LABEL_MAP.get(key)
        if fam is None:
            # tolerate variants such as 'DoS attacks-Hulk'
            for k, v in NF_LABEL_MAP.items():
                if key.startswith(k):
                    fam = v
                    break
        if fam is None:
            fam = "Other"
            unmapped[raw] = unmapped.get(raw, 0) + 1
        out.append(CANONICAL_LABELS.index(fam))
    return np.asarray(out, dtype=np.int64), unmapped


def stratified_subsample(df, label_col, cap, floor, seed):
    """Cap total rows while never dropping a class below `floor` (or its full
    size if smaller).  This is subsampling only -- no oversampling, so no
    duplicate can leak across the later split."""
    if cap is None or len(df) <= cap:
        return df
    rng = np.random.RandomState(seed)
    groups = {k: g for k, g in df.groupby(label_col)}
    kept, reserved = [], 0
    for k, g in groups.items():
        take = min(floor, len(g))
        kept.append(g.sample(n=take, random_state=rng.randint(1 << 31)))
        reserved += take
    budget = max(cap - reserved, 0)
    left = {k: len(g) - min(floor, len(g)) for k, g in groups.items()}
    tot = sum(left.values())
    if budget > 0 and tot > 0:
        for i, (k, g) in enumerate(groups.items()):
            if left[k] <= 0:
                continue
            extra = min(int(round(budget * left[k] / tot)), left[k])
            if extra > 0:
                pool = g.drop(index=kept[i].index)
                kept.append(pool.sample(n=extra, random_state=rng.randint(1 << 31)))
    return pd.concat(kept).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def count_classes(path, chunksize):
    """Pass 1: read ONLY the label column to get exact class counts cheaply."""
    attack_col = None
    counts, total = {}, 0
    head = pd.read_csv(path, nrows=1)
    head.columns = [c.strip() for c in head.columns]
    attack_col = next((c for c in ("Attack", "attack") if c in head.columns), None)
    if attack_col is None:
        raise KeyError(f"No 'Attack' column. Found: {list(head.columns)[:12]}")
    for chunk in pd.read_csv(path, usecols=[attack_col], chunksize=chunksize):
        chunk.columns = [c.strip() for c in chunk.columns]
        fam, _ = map_labels(chunk[attack_col])
        for f, n in zip(*np.unique(fam, return_counts=True)):
            counts[int(f)] = counts.get(int(f), 0) + int(n)
        total += len(chunk)
    return counts, total, attack_col


def plan_quota(counts, cap, floor):
    """How many rows to keep per class.

    Preserves the NATURAL class imbalance (the phenomenon under study) while
    guaranteeing a floor for rare families.  A flat per-class cap would
    flatten an 88%-benign corpus into a near-balanced one and silently
    destroy the imbalance the paper is about.
    """
    if cap is None:
        return {c: n for c, n in counts.items()}
    quota = {c: min(floor, n) for c, n in counts.items()}
    reserved = sum(quota.values())
    budget = max(cap - reserved, 0)
    left = {c: counts[c] - quota[c] for c in counts}
    tot = sum(left.values())
    if budget > 0 and tot > 0:
        for c in counts:
            quota[c] += min(int(round(budget * left[c] / tot)), left[c])
    return quota


PLAN = {}


def read_and_subsample(path, cap, floor, seed, chunksize=1_000_000):
    """Memory-safe two-pass read.

    NF-CSE-CIC-IDS2018-v2 has ~18.9M rows and NF-ToN-IoT-v2 ~16.9M; a plain
    read_csv of either exhausts Colab's ~12 GB.

    Pass 1 counts classes from the label column alone.  Pass 2 streams the
    full rows and keeps each class at its planned quota, sampling each chunk
    at the per-class acceptance rate so the retained sample is an unbiased
    draw that PRESERVES the original imbalance.  Peak memory is one chunk
    plus the retained sample.
    """
    rng = np.random.RandomState(seed)
    print("  pass 1/2: counting classes ...", flush=True)
    counts, total, attack_col = count_classes(path, chunksize)
    named = {CANONICAL_LABELS[c]: n for c, n in sorted(counts.items())}
    print(f"    {total:,} rows, classes: {named}")

    quota = plan_quota(counts, cap, floor)
    PLAN.update({"corpus_counts": {CANONICAL_LABELS[c]: int(n) for c, n in sorted(counts.items())},
                 "planned_quota": {CANONICAL_LABELS[c]: int(quota[c]) for c in sorted(quota)},
                 "quota_rule": "quota_c = min(floor, n_c) + round((cap - sum_j min(floor, n_j)) * "
                               "(n_c - min(floor, n_c)) / sum_j (n_j - min(floor, n_j)))"})
    rates = {c: (quota[c] / counts[c] if counts[c] else 0.0) for c in counts}
    print(f"    quota: { {CANONICAL_LABELS[c]: quota[c] for c in sorted(quota)} }")

    kept = {c: [] for c in counts}
    got = {c: 0 for c in counts}
    seen_rows = 0
    print("  pass 2/2: streaming rows ...", flush=True)
    for chunk in pd.read_csv(path, low_memory=False, chunksize=chunksize):
        chunk.columns = [c.strip() for c in chunk.columns]
        fam, _ = map_labels(chunk[attack_col])
        chunk = chunk.assign(_family=fam, _row=np.arange(seen_rows, seen_rows + len(chunk)))
        seen_rows += len(chunk)
        for c, grp in chunk.groupby("_family"):
            need = quota[c] - got[c]
            if need <= 0:
                continue
            # accept at the class rate, but never exceed the remaining need
            n_take = min(need, rng.binomial(len(grp), min(1.0, rates[c])))
            if n_take <= 0 and need > 0 and len(grp) and rates[c] >= 1.0:
                n_take = min(need, len(grp))
            if n_take > 0:
                kept[c].append(grp.sample(n=int(n_take),
                                          random_state=rng.randint(1 << 31)))
                got[c] += int(n_take)
        print(f"    {seen_rows:,}/{total:,} rows", flush=True)

    df = pd.concat([pd.concat(v) for v in kept.values() if v], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"  retained {len(df):,} rows, imbalance preserved: "
          f"{ {CANONICAL_LABELS[c]: got[c] for c in sorted(got) if got[c]} }")
    return df, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--dataset", required=True, help="short name, e.g. nf_unsw")
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--cap", type=int, default=400000, help="0 = keep all rows")
    ap.add_argument("--floor", type=int, default=1000)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-ttl", action="store_true",
                    help="keep TTL features (NOT recommended: they leak the label)")
    ap.add_argument("--stream", action="store_true",
                    help="chunked read + reservoir sampling; use for files "
                         "over ~5M rows (NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2)")
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cap = None if a.cap == 0 else a.cap

    if a.stream:
        print(f"streaming {a.input} (chunksize={a.chunksize:,}) ...")
        df, total_rows = read_and_subsample(a.input, cap, a.floor, a.seed, a.chunksize)
        print(f"  {len(df)} rows x {len(df.columns)} columns retained")
        unmapped = {}
    else:
        print(f"reading {a.input} ...")
        df = pd.read_csv(a.input, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        total_rows = len(df)
        print(f"  {len(df)} rows x {len(df.columns)} columns")
        attack_col = next((c for c in ("Attack", "attack") if c in df.columns), None)
        if attack_col is None:
            raise KeyError(f"No 'Attack' column. Found: {list(df.columns)[:12]}")
        y0, unmapped = map_labels(df[attack_col])
        df["_family"] = y0
        df["_row"] = np.arange(len(df))
        df = stratified_subsample(df, "_family", cap, a.floor, a.seed)

    if unmapped:
        print("  WARNING unmapped attack labels -> 'Other':", unmapped)

    drop = set(ID_COLS) | set(LABEL_COLS)
    if not a.keep_ttl:
        drop |= set(LEAKY_COLS)
    feat_cols = [c for c in df.columns
                 if c not in drop and c not in ("_family", "_row")
                 and pd.api.types.is_numeric_dtype(df[c])]
    dropped_present = sorted((drop & set(df.columns)))
    print(f"  dropped ({len(dropped_present)}): {dropped_present}")
    print(f"  using {len(feat_cols)} numeric NetFlow features")

    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.values.astype(np.float32)
    y = df["_family"].values.astype(np.int64)
    rows = df["_row"].values.astype(np.int64)

    # stratified split (subsample-only above, so no duplicate can cross the split)
    rng = np.random.RandomState(a.seed)
    tr_idx, te_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_te = max(1, int(round(len(idx) * a.test_frac))) if len(idx) > 1 else 0
        te_idx.extend(idx[:n_te])
        tr_idx.extend(idx[n_te:])
    tr_idx, te_idx = np.array(tr_idx), np.array(te_idx)

    for split, idx in (("train", tr_idx), ("test", te_idx)):
        np.save(f"{a.out}_{split}.X.npy", X[idx])
        np.save(f"{a.out}_{split}.y.npy", y[idx])
        np.save(f"{a.out}_{split}.rows.npy", rows[idx])   # source-row ids (overlap audit)
        counts = {CANONICAL_LABELS[i]: int(n)
                  for i, n in enumerate(np.bincount(y[idx], minlength=6))}
        meta = {"dataset": a.dataset, "split": split,
                "n_samples": int(len(idx)), "n_features": int(X.shape[1]),
                "features": feat_cols, "dropped_columns": dropped_present,
                "ttl_features_dropped": not a.keep_ttl,
                "label_order": CANONICAL_LABELS, "class_counts": counts,
                "cap": a.cap, "floor": a.floor, "seed": a.seed,
                "source_total_rows": int(total_rows), "streamed": bool(a.stream),
                "zero_filled_features": [], **PLAN}   # NF-v2 shares all features: none
        with open(f"{a.out}_{split}.meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [{split}] {len(idx)} rows -> {counts}")

    print(f"\nsupported families: "
          f"{[CANONICAL_LABELS[i] for i, n in enumerate(np.bincount(y, minlength=6)) if n > 0]}")


if __name__ == "__main__":
    main()
