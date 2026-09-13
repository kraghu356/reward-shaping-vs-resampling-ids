"""
nf_matrix.py -- cross-dataset matrix over the NF-v2 datasets.

Differences from the original runner, each addressing a review finding:

 * Shared 43-feature NetFlow schema, so no zero-fill and no invented mapping.
 * Strong tabular baselines (Extra Trees, Random Forest, optional XGBoost).
   Sarhan et al. used Extra Trees on this feature set; on tabular data a tree
   ensemble is the reference a reviewer will expect, not a courtesy addition.
 * The 2x2 reward x sampler factorial, so the class-frequency reward is
   separated from class-balanced sampling instead of confounded with it.
 * TRAIN ONCE PER SOURCE, then evaluate that frozen model on every target.
   The old runner retrained per (source,target) pair, which is not what
   "transferring a trained model" means.
 * Metrics report supported-family macro recall (primary) and fixed-taxonomy
   macro recall (secondary), each against its own chance line, plus the
   best-constant-family baseline so "collapse" has a precise reference.

Usage
    python nf_matrix.py --processed data/processed \
        --datasets nf_unsw nf_ton nf_cic2018 \
        --methods extra_trees rf ce focal dqn_plain_uniform dqn_reward_uniform \
                  dqn_plain_balanced dqn_reward_balanced \
        --seeds 42 7 13 99 123 --out results/nf_matrix.json
"""
import argparse
import json
import os

import numpy as np

LABELS = ["Normal", "DoS", "Probe", "Access", "Malware", "Other"]
K = len(LABELS)


# ---------------------------------------------------------------- metrics
def per_class_recall(y_true, y_pred):
    """None where the class has no test support (never silently 0.0)."""
    out = []
    for c in range(K):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        out.append(tp / (tp + fn) if (tp + fn) else None)
    return out


def per_class_f1(y_true, y_pred):
    out = []
    for c in range(K):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        if (tp + fn) == 0:
            out.append(None)
            continue
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn)
        out.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return out


def metrics_block(y_true, y_pred):
    rec = per_class_recall(y_true, y_pred)
    f1 = per_class_f1(y_true, y_pred)
    sup_idx = [i for i, r in enumerate(rec) if r is not None]
    n_sup = len(sup_idx)
    sup_macro = float(np.mean([rec[i] for i in sup_idx])) if sup_idx else 0.0
    fixed_macro = float(np.mean([0.0 if r is None else r for r in rec]))
    yb, pb = (y_true != 0).astype(int), (y_pred != 0).astype(int)
    tp = int(np.sum((pb == 1) & (yb == 1)))
    fp = int(np.sum((pb == 1) & (yb == 0)))
    fn = int(np.sum((pb == 0) & (yb == 1)))
    tn = int(np.sum((pb == 0) & (yb == 0)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    recl = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "supported_macro_recall": sup_macro,          # PRIMARY
        "fixed_taxonomy_macro": fixed_macro,          # SECONDARY, named
        "n_supported_families": n_sup,
        "chance_supported": 1.0 / K,                  # uniform K-way guessing
        "chance_fixed_taxonomy": n_sup / float(K) / K,
        "best_constant_family": 1.0 / n_sup if n_sup else 0.0,
        "macro_f1_supported": float(np.mean([f1[i] for i in sup_idx])) if sup_idx else 0.0,
        "per_class_recall": rec,
        "per_class_f1": f1,
        "accuracy": float(np.mean(y_true == y_pred)),
        "detection_precision": prec,
        "detection_recall": recl,
        "detection_f1": 2 * prec * recl / (prec + recl) if (prec + recl) else 0.0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
    }


# ---------------------------------------------------------------- models
def fit_model(method, Xtr, ytr, seed):
    """Return a fitted object exposing .predict(X). Trained ONCE per source."""
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    if method in ("extra_trees", "rf"):
        cls = ExtraTreesClassifier if method == "extra_trees" else RandomForestClassifier
        m = cls(n_estimators=100, n_jobs=-1, random_state=seed, class_weight="balanced")
        m.fit(Xtr, ytr)
        return ("sk", m, None)

    if method == "xgboost":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                              subsample=0.8, tree_method="hist",
                              random_state=seed, num_class=K,
                              objective="multi:softmax", verbosity=0)
        m.fit(Xtr, ytr)
        return ("sk", m, None)

    # neural methods share one scaler fitted on SOURCE TRAIN ONLY
    sc = StandardScaler().fit(Xtr)
    net = _fit_net(method, sc.transform(Xtr), ytr, seed)
    return ("net", net, sc)


def _fit_net(method, Xtr, ytr, seed, epochs=8, batch=256):
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(seed)
    d = Xtr.shape[1]

    # reward-aware / sampler-aware factorial conditions
    FACT = {
        "dqn_plain_uniform":   (False, False),
        "dqn_reward_uniform":  (True,  False),
        "dqn_plain_balanced":  (False, True),
        "dqn_reward_balanced": (True,  True),
        "ce":                  (None,  False),
        "focal":               (None,  False),
        "ce_balanced":         (None,  True),
        "focal_balanced":      (None,  True),
    }
    use_reward, use_balanced = FACT[method]

    counts = np.bincount(ytr, minlength=K).astype(float)
    N = counts.sum()
    lam = np.where(counts > 0, np.log(np.divide(N, counts, out=np.ones(K), where=counts > 0)), 1.0)

    idx = np.arange(len(ytr))
    if use_balanced:
        w = np.where(counts[ytr] > 0, 1.0 / counts[ytr], 0.0)
        w = w / w.sum()
        rs = np.random.RandomState(seed)
        idx = rs.choice(idx, size=len(idx), replace=True, p=w)
    Xf, yf = Xtr[idx], ytr[idx]

    model = keras.Sequential([
        keras.layers.Input(shape=(d,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(K),
    ])

    if use_reward is None:               # supervised heads
        if method.startswith("focal"):
            loss = keras.losses.CategoricalFocalCrossentropy(gamma=2.0, from_logits=True)
            yy = keras.utils.to_categorical(yf, K)
        else:
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            yy = yf
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss)
        model.fit(Xf, yy, epochs=epochs, batch_size=batch, verbose=0)
    else:                                # value head regressed to reward
        alpha = 5.0
        lam_use = lam if use_reward else np.ones(K)
        # target Q for the true class; wrong actions cost -lambda_c
        R = np.full((K, K), 0.0, dtype=np.float32)
        for c in range(K):
            R[c, :] = -lam_use[c]
            R[c, c] = alpha * lam_use[c]
        target = R[yf]                    # (n, K) full-row regression target
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(delta=1.0))
        model.fit(Xf, target, epochs=epochs, batch_size=batch, verbose=0)
    return model


def predict(kind_model_scaler, X):
    kind, m, sc = kind_model_scaler
    if kind == "sk":
        return m.predict(X).astype(np.int64)
    return np.argmax(m.predict(sc.transform(X), verbose=0), axis=1).astype(np.int64)


# ---------------------------------------------------------------- runner
def load(proc, ds, split):
    return (np.load(f"{proc}/{ds}_{split}.X.npy"),
            np.load(f"{proc}/{ds}_{split}.y.npy"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 13, 99, 123])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = {}
    if os.path.exists(a.out):
        for r in json.load(open(a.out)):
            done[(r["train"], r["test"], r["method"], r["seed"])] = r
        print(f"resuming: {len(done)} cells already complete")

    results = list(done.values())
    for src in a.datasets:
        Xtr, ytr = load(a.processed, src, "train")
        for method in a.methods:
            for seed in a.seeds:
                need = [t for t in a.datasets
                        if (src, t, method, seed) not in done]
                if not need:
                    continue
                print(f"train {method} on {src} seed={seed} ...", flush=True)
                fitted = fit_model(method, Xtr, ytr, seed)      # ONCE per source
                for tgt in need:
                    Xte, yte = load(a.processed, tgt, "test")
                    yp = predict(fitted, Xte)
                    rec = {"train": src, "test": tgt, "method": method,
                           "seed": seed, "in_domain": src == tgt}
                    rec.update(metrics_block(yte, yp))
                    results.append(rec)
                    done[(src, tgt, method, seed)] = rec
                    print(f"   -> {tgt:12s} supMacro={rec['supported_macro_recall']:.3f} "
                          f"(chance {rec['chance_supported']:.3f}, "
                          f"const {rec['best_constant_family']:.3f})  "
                          f"detF1={rec['detection_f1']:.3f}", flush=True)
                    with open(a.out, "w") as f:      # flush after every cell
                        json.dump(results, f, indent=1)
    print(f"\nwrote {len(results)} cells to {a.out}")


if __name__ == "__main__":
    main()
