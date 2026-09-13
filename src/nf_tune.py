"""
nf_tune.py -- honest performance tuning for the NF pipeline.

METHODOLOGY (this is the part reviewers check)
Every hyperparameter is selected on a VALIDATION split carved out of the
TRAINING data.  The test split is scored once, with the configuration already
chosen.  Selecting a configuration by its test score inflates the reported
number and is not reproducible by a reviewer, which is a far bigger threat to
acceptance than a modest honest result.

Improvements over the first pass, each with a stated reason:

1. Categorical encoding.  L7_PROTO and PROTOCOL are integer CODES, not
   magnitudes.  Feeding them as continuous features tells the network that
   protocol 17 is "more" than protocol 6, which is meaningless.  We one-hot
   the top-k levels.  Trees are indifferent; the neural methods are not.

2. Log1p scaling for heavy-tailed counters.  Byte/packet counters span many
   orders of magnitude; StandardScaler on a raw heavy tail leaves almost all
   mass in a narrow band.  log1p first, then standardise.

3. Early stopping on validation macro recall, with a generous epoch budget,
   instead of a fixed 8 epochs.

4. Class weighting for the supervised baselines.  Without it, CE is being
   compared at a disadvantage -- and an unfairly weak baseline is the first
   thing a reviewer attacks.  We want our objective to beat a STRONG CE.

5. A small hyperparameter grid per method, selected on validation.

Usage
    python nf_tune.py --processed data/processed --dataset nf_unsw \\
        --methods extra_trees ce focal dqn_reward_balanced \\
        --seeds 42 7 13 99 123 --out results/nf_tuned.json
"""
import argparse
import json
import os

import numpy as np

LABELS = ["Normal", "DoS", "Probe", "Access", "Malware", "Other"]
K = len(LABELS)
# Column indices are resolved from the meta.json feature list at runtime.
CATEGORICAL_NAMES = ["L7_PROTO", "PROTOCOL", "TCP_FLAGS",
                     "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "ICMP_TYPE",
                     "ICMP_IPV4_TYPE", "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE"]
# Heavy-tailed counters that benefit from log1p before standardisation.
LOGSCALE_HINTS = ["BYTES", "PKTS", "DURATION", "SECOND", "THROUGHPUT", "LEN"]


# ------------------------------------------------------------------ features
def build_transform(Xtr, feat_names, top_k=12):
    """Fit a feature transform on TRAIN ONLY. Returns (fn, out_dim, report)."""
    cat_idx = [i for i, n in enumerate(feat_names) if n in CATEGORICAL_NAMES]
    log_idx = [i for i, n in enumerate(feat_names)
               if i not in cat_idx and any(h in n.upper() for h in LOGSCALE_HINTS)]
    num_idx = [i for i in range(len(feat_names)) if i not in cat_idx]

    # retain the top_k most frequent levels per categorical column
    levels = {}
    for i in cat_idx:
        vals, counts = np.unique(Xtr[:, i], return_counts=True)
        levels[i] = vals[np.argsort(-counts)][:top_k]

    Z = Xtr[:, num_idx].astype(np.float64).copy()
    logmask = np.array([c in log_idx for c in num_idx])
    if logmask.any():
        Z[:, logmask] = np.log1p(np.clip(Z[:, logmask], 0, None))
    mu, sd = Z.mean(axis=0), Z.std(axis=0)
    sd[sd < 1e-8] = 1.0

    def fn(X):
        Zn = X[:, num_idx].astype(np.float64).copy()
        if logmask.any():
            Zn[:, logmask] = np.log1p(np.clip(Zn[:, logmask], 0, None))
        Zn = (Zn - mu) / sd
        parts = [Zn]
        for i in cat_idx:
            col = X[:, i]
            oh = np.zeros((len(X), len(levels[i]) + 1), dtype=np.float64)
            for j, lv in enumerate(levels[i]):
                oh[:, j] = (col == lv)
            oh[:, -1] = ~np.isin(col, levels[i])       # "other" bucket
            parts.append(oh)
        return np.hstack(parts).astype(np.float32)

    out_dim = fn(Xtr[:5]).shape[1]
    report = {"n_raw": len(feat_names), "n_transformed": int(out_dim),
              "categorical": [feat_names[i] for i in cat_idx],
              "log_scaled": [feat_names[i] for i in log_idx]}
    return fn, out_dim, report


# ------------------------------------------------------------------ metrics
def macro_recall_supported(y_true, y_pred):
    vals = []
    for c in range(K):
        m = (y_true == c)
        if m.sum():
            vals.append(float((y_pred[m] == c).mean()))
    return float(np.mean(vals)) if vals else 0.0


def full_metrics(y_true, y_pred):
    rec, f1 = [], []
    for c in range(K):
        m = (y_true == c)
        if not m.sum():
            rec.append(None); f1.append(None); continue
        tp = int(((y_pred == c) & m).sum())
        fp = int(((y_pred == c) & ~m).sum())
        fn = int(((y_pred != c) & m).sum())
        r = tp / (tp + fn)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        rec.append(r); f1.append(2 * p * r / (p + r) if (p + r) else 0.0)
    sup = [i for i, v in enumerate(rec) if v is not None]
    yb, pb = (y_true != 0).astype(int), (y_pred != 0).astype(int)
    tp = int(((pb == 1) & (yb == 1)).sum()); fp = int(((pb == 1) & (yb == 0)).sum())
    fn = int(((pb == 0) & (yb == 1)).sum()); tn = int(((pb == 0) & (yb == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return {"supported_macro_recall": float(np.mean([rec[i] for i in sup])),
            "macro_f1_supported": float(np.mean([f1[i] for i in sup])),
            "per_class_recall": rec, "per_class_f1": f1,
            "n_supported_families": len(sup),
            "chance_supported": 1.0 / K,
            "best_constant_family": 1.0 / len(sup) if sup else 0.0,
            "accuracy": float((y_true == y_pred).mean()),
            "detection_f1": 2 * p * r / (p + r) if (p + r) else 0.0,
            "detection_precision": p, "detection_recall": r,
            "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0}


# ------------------------------------------------------------------ grids
GRIDS = {
    "extra_trees": [{"n_estimators": n, "min_samples_leaf": l, "class_weight": w}
                    for n in (300,) for l in (1, 3) for w in ("balanced", "balanced_subsample")],
    "rf":          [{"n_estimators": n, "min_samples_leaf": l, "class_weight": "balanced"}
                    for n in (300,) for l in (1, 3)],
    "xgboost":     [{"max_depth": d, "learning_rate": lr, "n_estimators": 400}
                    for d in (6, 10) for lr in (0.1, 0.2)],
    # neural: width x lr, with class weighting for the supervised heads
    "_net":        [{"width": w, "lr": lr} for w in (256, 512) for lr in (1e-3, 3e-4)],
}


def fit_tree(method, cfg, Xtr, ytr, seed):
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    if method == "extra_trees":
        m = ExtraTreesClassifier(random_state=seed, n_jobs=-1, **cfg)
    elif method == "rf":
        m = RandomForestClassifier(random_state=seed, n_jobs=-1, **cfg)
    else:
        import xgboost as xgb
        m = xgb.XGBClassifier(objective="multi:softmax", num_class=K,
                              tree_method="hist", subsample=0.8,
                              random_state=seed, verbosity=0, n_jobs=-1, **cfg)
    m.fit(Xtr, ytr)
    return m


def fit_net(method, cfg, Xtr, ytr, Xva, yva, seed, max_epochs=60):
    import tensorflow as tf
    from tensorflow import keras
    tf.keras.utils.set_random_seed(seed)

    FACT = {"dqn_plain_uniform": (False, False), "dqn_reward_uniform": (True, False),
            "dqn_plain_balanced": (False, True), "dqn_reward_balanced": (True, True),
            "ce": (None, False), "focal": (None, False)}
    use_reward, use_balanced = FACT[method]

    counts = np.bincount(ytr, minlength=K).astype(float)
    N = counts.sum()
    lam = np.where(counts > 0, np.log(np.divide(N, counts, out=np.ones(K), where=counts > 0)), 1.0)

    idx = np.arange(len(ytr))
    if use_balanced:
        w = np.where(counts[ytr] > 0, 1.0 / counts[ytr], 0.0); w /= w.sum()
        idx = np.random.RandomState(seed).choice(idx, len(idx), replace=True, p=w)
    Xf, yf = Xtr[idx], ytr[idx]

    net = keras.Sequential([
        keras.layers.Input(shape=(Xtr.shape[1],)),
        keras.layers.Dense(cfg["width"], activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(cfg["width"] // 2, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(K),
    ])
    opt = keras.optimizers.Adam(cfg["lr"])

    class MacroRecall(keras.callbacks.Callback):
        """Early stopping on validation macro recall -- the metric we report,
        not validation loss, which is dominated by the majority class."""
        def __init__(self):
            super().__init__(); self.best = -1.0; self.best_w = None; self.wait = 0
        def on_epoch_end(self, epoch, logs=None):
            p = np.argmax(self.model.predict(Xva, verbose=0), axis=1)
            s = macro_recall_supported(yva, p)
            if s > self.best + 1e-4:
                self.best, self.best_w, self.wait = s, self.model.get_weights(), 0
            else:
                self.wait += 1
                if self.wait >= 8:
                    self.model.stop_training = True
        def on_train_end(self, logs=None):
            if self.best_w is not None:
                self.model.set_weights(self.best_w)

    cb = MacroRecall()
    if use_reward is None:                       # supervised, class-weighted
        cw = {c: (N / (K * counts[c]) if counts[c] else 0.0) for c in range(K)}
        if method == "focal":
            net.compile(opt, loss=keras.losses.CategoricalFocalCrossentropy(
                gamma=2.0, from_logits=True))
            net.fit(Xf, keras.utils.to_categorical(yf, K), epochs=max_epochs,
                    batch_size=512, verbose=0, callbacks=[cb],
                    class_weight=None if use_balanced else cw)
        else:
            net.compile(opt, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
            net.fit(Xf, yf, epochs=max_epochs, batch_size=512, verbose=0,
                    callbacks=[cb], class_weight=None if use_balanced else cw)
    else:                                        # value head regressed to reward
        alpha, lam_use = 5.0, (lam if use_reward else np.ones(K))
        R = np.zeros((K, K), np.float32)
        for c in range(K):
            R[c, :] = -lam_use[c]; R[c, c] = alpha * lam_use[c]
        net.compile(opt, loss=keras.losses.Huber(delta=1.0))
        net.fit(Xf, R[yf], epochs=max_epochs, batch_size=512, verbose=0, callbacks=[cb])
    return net, cb.best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 13, 99, 123])
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    Xtr_all = np.load(f"{a.processed}/{a.dataset}_train.X.npy")
    ytr_all = np.load(f"{a.processed}/{a.dataset}_train.y.npy")
    Xte = np.load(f"{a.processed}/{a.dataset}_test.X.npy")
    yte = np.load(f"{a.processed}/{a.dataset}_test.y.npy")
    meta = json.load(open(f"{a.processed}/{a.dataset}_train.meta.json"))
    feat_names = meta["features"]

    results = []
    if os.path.exists(a.out):
        results = json.load(open(a.out))
    done = {(r["method"], r["seed"]) for r in results}

    for seed in a.seeds:
        # stratified train/validation split of the TRAINING data only
        rng = np.random.RandomState(seed)
        tr, va = [], []
        for c in np.unique(ytr_all):
            ix = np.where(ytr_all == c)[0]; rng.shuffle(ix)
            n_va = max(1, int(round(len(ix) * a.val_frac)))
            va.extend(ix[:n_va]); tr.extend(ix[n_va:])
        tr, va = np.array(tr), np.array(va)

        tf_fn, dim, rep = build_transform(Xtr_all[tr], feat_names)
        if seed == a.seeds[0]:
            print(f"feature transform: {rep['n_raw']} raw -> {rep['n_transformed']} dims")
            print(f"  categorical one-hot: {rep['categorical']}")
            print(f"  log1p-scaled       : {len(rep['log_scaled'])} counters")

        for method in a.methods:
            if (method, seed) in done:
                continue
            grid = GRIDS.get(method, GRIDS["_net"])
            is_tree = method in ("extra_trees", "rf", "xgboost")
            Atr = Xtr_all[tr] if is_tree else tf_fn(Xtr_all[tr])
            Ava = Xtr_all[va] if is_tree else tf_fn(Xtr_all[va])
            Ate = Xte if is_tree else tf_fn(Xte)

            best = (-1.0, None, None)
            for cfg in grid:                      # SELECT ON VALIDATION
                if is_tree:
                    m = fit_tree(method, cfg, Atr, ytr_all[tr], seed)
                    s = macro_recall_supported(ytr_all[va], m.predict(Ava))
                else:
                    m, s = fit_net(method, cfg, Atr, ytr_all[tr],
                                   Ava, ytr_all[va], seed)
                print(f"  [{method} seed={seed}] {cfg} -> val {s:.4f}", flush=True)
                if s > best[0]:
                    best = (s, cfg, m)

            val_score, cfg, model = best
            yp = (model.predict(Ate) if is_tree
                  else np.argmax(model.predict(Ate, verbose=0), axis=1))
            rec = {"dataset": a.dataset, "method": method, "seed": seed,
                   "selected_config": cfg, "val_macro_recall": val_score,
                   "feature_dims": int(dim)}
            rec.update(full_metrics(yte, yp.astype(np.int64)))
            results.append(rec)
            print(f"  => {method} seed={seed}: val {val_score:.4f} | "
                  f"TEST supMacro {rec['supported_macro_recall']:.4f} "
                  f"macroF1 {rec['macro_f1_supported']:.4f} "
                  f"detF1 {rec['detection_f1']:.4f}", flush=True)
            with open(a.out, "w") as f:
                json.dump(results, f, indent=1)

    print(f"\nwrote {len(results)} runs to {a.out}")
    print("NOTE: every config was chosen on validation; test scored once.")


if __name__ == "__main__":
    main()
