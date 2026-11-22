"""
nf_revision.py -- extended experiments of the manuscript.

Builds on nf_tune.py (same feature transform, same validation-only selection,
same stratified split) and adds every arm the reviews ask for:

  R2/R4  loss       --loss mse | huber,  --kappa <float> | auto  (auto = (alpha+1)*lambda_max,
                    i.e. every target stays in the quadratic regime), --alpha <float>
  R2/R4  sampler    none | frozen (the original single bootstrap) | epoch (redrawn every epoch)
                    + number of UNIQUE training flows seen per class
  R2/R4  control    the same 2x2 with class-weighted cross-entropy (weights -log pi_c)
  R2/R4  metrics    per-family one-vs-rest AUPRC, binary AUROC / AUPRC (threshold-free)
  R2/R4  recalib.   per-class additive offsets tuned on VALIDATION for macro-F1, then
                    applied once to test; CE + post-hoc logit adjustment (tau tuned on val)
  R2     mechanism  for value-head runs with the plain reward, the population-minimiser
                    rescaling argmax_a lambda_a (Q_a + 1) is scored as a derived condition,
                    and test predictions are saved so agreement with RB can be computed
  R4     seeds      any number of training seeds (use 10-20)

Method names:  <head>-<weight>-<sampler>
    head    val  (one-step full-information value regression, Eq. 7)
            ce   (cross-entropy)
    weight  plain | freq   (val: reward lambda_c = 1 or -log pi_c;
                            ce : sample weight 1 or -log pi_c)
    sampler none | frozen | epoch
  e.g.  val-plain-none = PU,  val-freq-none = RU,  val-plain-epoch = PB (per-epoch),
        val-freq-epoch = RB (per-epoch),  ce-freq-epoch = weighted CE + balanced sampler.
  Trees: extra_trees | rf | xgboost  (validation-tuned, as in nf_tune.py)

Usage
    python src/nf_revision.py --processed data/processed --dataset nf_unsw \
        --methods val-plain-none val-freq-none val-plain-epoch val-freq-epoch \
        --loss mse --alpha 5 --seeds 42 7 13 99 123 2024 31 57 88 101 \
        --save-preds results/preds --out results/rev_nf_unsw_mse.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nf_tune import GRIDS, K, build_transform, fit_tree, full_metrics, macro_recall_supported  # noqa: E402

TREES = ("extra_trees", "rf", "xgboost")


# ------------------------------------------------------------------ helpers
def parse_method(m):
    if m in TREES:
        return None
    head, weight, sampler = m.split("-")
    assert head in ("val", "ce", "bandit") and weight in ("plain", "freq") \
        and sampler in ("none", "frozen", "epoch", "iw"), m
    return head, weight, sampler


def class_stats(y):
    counts = np.bincount(y, minlength=K).astype(float)
    N = counts.sum()
    pi = counts / N
    lam = np.where(counts > 0, -np.log(np.where(pi > 0, pi, 1.0)), 1.0)
    return counts, pi, lam


def draw(rs, y, counts):
    """Class-balanced bootstrap of size N: p_i proportional to 1/N_{y_i}."""
    w = np.where(counts[y] > 0, 1.0 / counts[y], 0.0)
    return rs.choice(len(y), size=len(y), replace=True, p=w / w.sum())


def macro_f1(y, p):
    vals = []
    for c in range(K):
        m = y == c
        if not m.sum():
            continue
        tp = ((p == c) & m).sum(); fp = ((p == c) & ~m).sum(); fn = ((p != c) & m).sum()
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn)
        vals.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(vals))


def threshold_free(y, S):
    """S: (n, K) scores (logits or Q values; both rank-valid per column)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    auprc = []
    for c in range(K):
        yc = (y == c).astype(int)
        auprc.append(float(average_precision_score(yc, S[:, c])) if 0 < yc.sum() < len(y) else None)
    sup = [v for v in auprc if v is not None]
    # binary attack score: best attack action minus the Normal action
    a = S[:, 1:].max(axis=1) - S[:, 0]
    yb = (y != 0).astype(int)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(yb, a)
    at = {f"recall_at_fpr_{f:g}": float(np.max(tpr[fpr <= f])) if np.any(fpr <= f) else 0.0
          for f in (1e-3, 1e-2)}
    return {"per_class_auprc": auprc,
            "macro_auprc": float(np.mean(sup)),
            "binary_auroc": float(roc_auc_score(yb, a)),
            "binary_auprc": float(average_precision_score(yb, a)), **at}


def tune_offsets(yva, Sva, passes=3, npts=41):
    """Per-class additive offsets (Normal fixed at 0) maximising val macro-F1."""
    b = np.zeros(K)
    span = 3.0 * float(np.std(Sva)) + 1e-6
    grid = np.linspace(-span, span, npts)
    best = macro_f1(yva, np.argmax(Sva, 1))
    for _ in range(passes):
        for c in range(1, K):
            keep = b[c]
            for g in grid:
                b[c] = g
                s = macro_f1(yva, np.argmax(Sva + b, 1))
                if s > best + 1e-6:
                    best, keep = s, g
            b[c] = keep
    return b, best


def tune_logit_adjust(yva, Zva, pi):
    """Post-hoc logit adjustment z_a - tau log pi_a (Menon et al. 2021), tau on val."""
    lp = np.log(np.where(pi > 0, pi, 1.0))
    best = (-1.0, 0.0)
    for tau in np.arange(0.0, 2.01, 0.25):
        s = macro_recall_supported(yva, np.argmax(Zva - tau * lp, 1))
        if s > best[0] + 1e-6:
            best = (s, float(tau))
    return best[1], lp


# ------------------------------------------------------------------ model
def build_net(d, width, lr):
    from tensorflow import keras
    net = keras.Sequential([
        keras.layers.Input(shape=(d,)),
        keras.layers.Dense(width, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(width // 2, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(K),
    ])
    return net, keras.optimizers.Adam(lr)


def fit_net(spec, cfg, Xtr, ytr, Xva, yva, seed, args, max_epochs, patience=8):
    import tensorflow as tf
    from tensorflow import keras
    tf.keras.utils.set_random_seed(seed)
    head, weight, sampler = spec
    counts, pi, lam = class_stats(ytr)
    rs = np.random.RandomState(seed)

    net, opt = build_net(Xtr.shape[1], cfg["width"], cfg["lr"])
    lam_use = lam if weight == "freq" else np.ones(K)
    kappa = None
    if head in ("val", "bandit"):
        alpha = args.alpha
        R = np.zeros((K, K), np.float32)
        for c in range(K):
            R[c, :] = -lam_use[c]; R[c, c] = alpha * lam_use[c]
        if args.loss == "mse":
            loss = keras.losses.MeanSquaredError()
        else:
            kappa = (alpha + 1) * lam_use.max() if args.kappa == "auto" else float(args.kappa)
            loss = keras.losses.Huber(delta=kappa)
    else:
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    net.compile(opt, loss=loss)

    if head == "bandit":
        return fit_bandit(net, opt, R, lam_use, spec, Xtr, ytr, Xva, yva, rs, counts, args,
                          max_epochs, patience, kappa)
    iw = (len(ytr) / (K * counts[ytr])).astype(np.float32) if sampler == "iw" else None
    seen = np.zeros(len(ytr), bool)
    idx = np.arange(len(ytr)) if sampler in ("none", "iw") else draw(rs, ytr, counts)
    best, best_w, wait, ep_run = -1.0, None, 0, 0
    for ep in range(max_epochs):
        if sampler == "epoch" and ep > 0:
            idx = draw(rs, ytr, counts)          # redraw every epoch
        seen[idx] = True
        Xe, ye = Xtr[idx], ytr[idx]
        if head == "val":
            net.fit(Xe, R[ye], sample_weight=iw, epochs=1, batch_size=512, verbose=0, shuffle=True)
        else:
            sw = lam_use[ye].astype(np.float32)   # -log pi_c weights, or 1
            if iw is not None:
                sw = sw * iw
            net.fit(Xe, ye, sample_weight=sw, epochs=1, batch_size=512, verbose=0, shuffle=True)
        ep_run = ep + 1
        s = macro_recall_supported(yva, np.argmax(net.predict(Xva, verbose=0, batch_size=4096), 1))
        if s > best + 1e-4:
            best, best_w, wait = s, net.get_weights(), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_w is not None:
        net.set_weights(best_w)
    info = {"epochs_run": ep_run, "kappa": kappa,
            "unique_seen_per_class": [int(seen[ytr == c].sum()) for c in range(K)],
            "train_per_class": counts.astype(int).tolist()}
    return net, best, info


def fit_bandit(net, opt, R, lam_use, spec, Xtr, ytr, Xva, yva, rs, counts, args,
               max_epochs, patience, kappa, batch=512, buffer_cap=50000,
               eps_start=1.0, eps_end=0.05, warmup_epochs=5):
    """Epsilon-greedy contextual bandit with experience replay (reviewer M3 / R2 / R4-3).

    Unlike the full-information value regression, the agent observes ONLY the reward of
    the action it took, chosen epsilon-greedily from its own Q estimates; transitions
    (x, a, r) go into a FIFO replay buffer and gradient steps use uniform minibatches
    from that buffer, updating Q(x, a) for the taken action only.  One-step returns
    (gamma = 0): with i.i.d. next states the bootstrap term is action-independent and
    leaves the greedy policy unchanged (Section V).  The sampler controls which flows
    the agent is shown, exactly as an environment's sampling would.
    """
    import tensorflow as tf
    _, weight, sampler = spec
    loss_fn = (tf.keras.losses.Huber(delta=kappa, reduction="none") if args.loss == "huber"
               else tf.keras.losses.MeanSquaredError(reduction="none"))
    Rt = tf.constant(R)
    iw_all = (len(ytr) / (K * counts[ytr])).astype(np.float32) if sampler == "iw" \
        else np.ones(len(ytr), np.float32)

    @tf.function
    def act(x):
        return tf.argmax(net(x, training=False), axis=1, output_type=tf.int32)

    @tf.function
    def step(x, a, r, w):
        with tf.GradientTape() as tape:
            q = net(x, training=True)
            qa = tf.gather(q, a, axis=1, batch_dims=1)
            l = loss_fn(tf.expand_dims(r, 1), tf.expand_dims(qa, 1))
            l = tf.reduce_sum(l * w) / tf.reduce_sum(w)
        g = tape.gradient(l, net.trainable_variables)
        opt.apply_gradients(zip(g, net.trainable_variables))

    buf_i = np.zeros(buffer_cap, np.int64); buf_a = np.zeros(buffer_cap, np.int32)
    buf_r = np.zeros(buffer_cap, np.float32); n_buf = 0; ptr = 0
    seen = np.zeros(len(ytr), bool)
    idx = np.arange(len(ytr)) if sampler in ("none", "iw") else draw(rs, ytr, counts)
    steps_per_epoch = int(np.ceil(len(ytr) / batch))
    total_warm = warmup_epochs * steps_per_epoch
    t = 0
    best, best_w, wait, ep_run = -1.0, None, 0, 0
    for ep in range(max_epochs):
        if sampler == "epoch" and ep > 0:
            idx = draw(rs, ytr, counts)
        order = rs.permutation(idx)
        seen[order] = True
        for b in range(steps_per_epoch):
            ib = order[b * batch:(b + 1) * batch]
            if len(ib) == 0:
                continue
            eps = max(eps_end, eps_start - (eps_start - eps_end) * t / max(1, total_warm))
            greedy = act(tf.constant(Xtr[ib])).numpy()
            explore = rs.rand(len(ib)) < eps
            a = np.where(explore, rs.randint(0, K, len(ib)), greedy).astype(np.int32)
            r = R[ytr[ib], a]                                   # reward of the taken action only
            for j in range(len(ib)):                            # FIFO replay buffer
                buf_i[ptr] = ib[j]; buf_a[ptr] = a[j]; buf_r[ptr] = r[j]
                ptr = (ptr + 1) % buffer_cap
            n_buf = min(buffer_cap, n_buf + len(ib))
            sel = rs.randint(0, n_buf, batch)
            xi = buf_i[sel]
            step(tf.constant(Xtr[xi]), tf.constant(buf_a[sel]), tf.constant(buf_r[sel]),
                 tf.constant(iw_all[xi]))
            t += 1
        ep_run = ep + 1
        s = macro_recall_supported(yva, np.argmax(net.predict(Xva, verbose=0, batch_size=4096), 1))
        if s > best + 1e-4:
            best, best_w, wait = s, net.get_weights(), 0
        elif ep >= warmup_epochs:                                # no early stop during exploration warm-up
            wait += 1
            if wait >= patience:
                break
    if best_w is not None:
        net.set_weights(best_w)
    info = {"epochs_run": ep_run, "kappa": kappa, "bandit": {"eps_end": eps_end, "buffer": buffer_cap,
            "warmup_epochs": warmup_epochs},
            "unique_seen_per_class": [int(seen[ytr == c].sum()) for c in range(K)],
            "train_per_class": counts.astype(int).tolist()}
    return net, best, info


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[42, 7, 13, 99, 123, 2024, 31, 57, 88, 101])
    ap.add_argument("--loss", choices=["mse", "huber"], default="mse")
    ap.add_argument("--kappa", default="1.0", help="Huber threshold, or 'auto'")
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--grid", choices=["full", "fixed", "wide"], default="full",
                    help="fixed = width 256, lr 1e-3 (cheap sweeps); wide = width {256,512,1024} x "
                         "lr {1e-3,3e-4,1e-4} (reviewer Mo6)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--save-preds", default=None)
    ap.add_argument("--tag", default="", help="suffix for saved prediction files (keeps runs apart)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    P = a.processed
    Xtr_all = np.load(f"{P}/{a.dataset}_train.X.npy"); ytr_all = np.load(f"{P}/{a.dataset}_train.y.npy")
    Xte = np.load(f"{P}/{a.dataset}_test.X.npy"); yte = np.load(f"{P}/{a.dataset}_test.y.npy")
    feats = json.load(open(f"{P}/{a.dataset}_train.meta.json"))["features"]
    # values beyond float32 range became +-inf when prepare_nf cast to float32; apply the
    # same policy prepare_nf uses for non-finite inputs (set to 0). No-op on finite data.
    n_bad = int((~np.isfinite(Xtr_all)).sum() + (~np.isfinite(Xte)).sum())
    if n_bad:
        print(f"  [{a.dataset}] {n_bad} non-finite feature values set to 0", flush=True)
        Xtr_all = np.nan_to_num(Xtr_all, nan=0.0, posinf=0.0, neginf=0.0)
        Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)

    tag = {"loss": a.loss, "kappa": a.kappa if a.loss == "huber" else None, "alpha": a.alpha}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.save_preds:
        os.makedirs(a.save_preds, exist_ok=True)
    results = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["method"], r["seed"]) for r in results}

    for seed in a.seeds:
        rng = np.random.RandomState(seed)
        tr, va = [], []
        for c in np.unique(ytr_all):
            ix = np.where(ytr_all == c)[0]; rng.shuffle(ix)
            n_va = max(1, int(round(len(ix) * a.val_frac)))
            va.extend(ix[:n_va]); tr.extend(ix[n_va:])
        tr, va = np.array(tr), np.array(va)
        ytr, yva = ytr_all[tr], ytr_all[va]
        tfn, dim, _ = build_transform(Xtr_all[tr], feats)
        _, pi, lam = class_stats(ytr)

        for method in a.methods:
            if (method, seed) in done:
                continue
            spec = parse_method(method)
            t0 = time.time()
            rec = {"dataset": a.dataset, "method": method, "seed": seed, **tag}

            if spec is None:                                  # tree baselines
                best = (-1.0, None, None)
                for cfg in GRIDS[method]:
                    m = fit_tree(method, cfg, Xtr_all[tr], ytr, seed)
                    s = macro_recall_supported(yva, m.predict(Xtr_all[va]))
                    if s > best[0]:
                        best = (s, cfg, m)
                s, cfg, m = best
                P_ = m.predict_proba(Xte)                      # columns = m.classes_ only
                Ste = np.full((len(Xte), K), -1.0)             # absent classes never win
                Ste[:, np.asarray(m.classes_, int)] = P_
                rec.update({"selected_config": cfg, "val_macro_recall": s})
                rec.update(full_metrics(yte, np.argmax(Ste, 1)))
                rec["threshold_free"] = threshold_free(yte, Ste)
                if a.save_preds:
                    np.save(f"{a.save_preds}/{a.dataset}_{method}_{a.loss}{a.tag}_s{seed}.npy",
                            np.argmax(Ste, 1).astype(np.int8))
            else:
                Atr, Ava, Ate = tfn(Xtr_all[tr]), tfn(Xtr_all[va]), tfn(Xte)
                grid = {"full": GRIDS["_net"], "fixed": [{"width": 256, "lr": 1e-3}],
                        "wide": [{"width": w, "lr": lr} for w in (256, 512, 1024)
                                 for lr in (1e-3, 3e-4, 1e-4)]}[a.grid]
                best = (-1.0, None, None, None)
                for cfg in grid:                              # SELECT ON VALIDATION
                    net, s, info = fit_net(spec, cfg, Atr, ytr, Ava, yva, seed, a, a.max_epochs)
                    print(f"  [{method} seed={seed}] {cfg} -> val {s:.4f} ({info['epochs_run']} ep)",
                          flush=True)
                    if s > best[0]:
                        best = (s, cfg, net, info)
                s, cfg, net, info = best
                Sva = net.predict(Ava, verbose=0, batch_size=4096)
                Ste = net.predict(Ate, verbose=0, batch_size=4096)
                yp = np.argmax(Ste, 1)
                rec.update({"selected_config": cfg, "val_macro_recall": s, **info})
                rec.update(full_metrics(yte, yp))                     # argmax operating point
                rec["threshold_free"] = threshold_free(yte, Ste)

                b, vf1 = tune_offsets(yva, Sva)                        # recalibrated on val
                rec["recalibrated"] = {"offsets": b.tolist(), "val_macro_f1": vf1,
                                       **full_metrics(yte, np.argmax(Ste + b, 1))}

                head, weight, sampler = spec
                if head == "ce" and weight == "plain" and sampler == "none":
                    tau, lp = tune_logit_adjust(yva, Sva, pi)          # logit-adjust baseline
                    rec["logit_adjusted"] = {"tau": tau, **full_metrics(yte, np.argmax(Ste - tau * lp, 1))}
                if head in ("val", "bandit") and weight == "freq" and a.loss == "mse":
                    # post-hoc prior correction (reviewer Mo3): under squared loss
                    # Q_a + C(x) = (alpha+1) lambda_a P_q(a|x), with C(x) recovered from sum_a P = 1
                    al = a.alpha
                    C = (al + 1 - (Ste / lam).sum(1, keepdims=True)) / (1.0 / lam).sum()
                    Pq = (Ste + C) / ((al + 1) * lam)
                    rec["prior_corrected"] = full_metrics(yte, np.argmax(Pq, 1))
                if head in ("val", "bandit") and weight == "plain":
                    # population minimiser under MSE: Q_a = (alpha+1) P(a|x) - 1 for the plain
                    # reward, so argmax_a lambda_a (Q_a + 1) is the decision RB would make
                    yr = np.argmax(lam * (Ste + 1.0), 1)
                    rec["lambda_rescaled"] = full_metrics(yte, yr)
                    if a.save_preds:
                        np.save(f"{a.save_preds}/{a.dataset}_{method}_{a.loss}{a.tag}_s{seed}_rescaled.npy",
                                yr.astype(np.int8))
                if a.save_preds:
                    np.save(f"{a.save_preds}/{a.dataset}_{method}_{a.loss}{a.tag}_s{seed}.npy", yp.astype(np.int8))
                    if "prior_corrected" in rec:
                        np.save(f"{a.save_preds}/{a.dataset}_{method}_{a.loss}{a.tag}_s{seed}_priorcorr.npy",
                                np.argmax(Pq, 1).astype(np.int8))

            rec["seconds"] = round(time.time() - t0, 1)
            results.append(rec)
            msg = (f"  => {method} seed={seed}: supMacro {rec['supported_macro_recall']:.4f} "
                   f"macroF1 {rec['macro_f1_supported']:.4f} FPR {rec['false_positive_rate']:.4f} "
                   f"macroAUPRC {rec['threshold_free']['macro_auprc']:.4f}")
            if "recalibrated" in rec:
                msg += f" | recal macroF1 {rec['recalibrated']['macro_f1_supported']:.4f}"
            print(msg, flush=True)
            with open(a.out, "w") as f:
                json.dump(results, f, indent=1)
    print(f"\nwrote {len(results)} runs to {a.out}")


if __name__ == "__main__":
    main()
