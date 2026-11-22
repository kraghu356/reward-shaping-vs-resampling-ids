# Train/test leakage audit: exact duplicate feature vectors across splits, and the
# factorial re-scored on the test flows whose feature vector never occurs in training.
import numpy as np, json, hashlib, os
from collections import defaultdict
P = globals().get('P', '/content/drive/MyDrive/CNN-LSTM-DQN-IDS-v2-results/nf_processed')
PR = globals().get('PR', '/content/drive/MyDrive/rev_results/preds')
SEEDS = [42, 7, 13, 99, 123, 2024, 31, 57, 88, 101]
def keys(X):
    X = np.ascontiguousarray(np.nan_to_num(X.astype(np.float32), nan=0, posinf=0, neginf=0))
    return [hashlib.md5(r.tobytes()).hexdigest() for r in X]
def mr(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))
out = {}
for ds in ['nf_unsw', 'nf_ton', 'nf_bot']:
    if not os.path.exists(f'{P}/{ds}_test.X.npy'):
        continue
    Xtr, ytr = np.load(f'{P}/{ds}_train.X.npy'), np.load(f'{P}/{ds}_train.y.npy')
    Xte, yte = np.load(f'{P}/{ds}_test.X.npy'), np.load(f'{P}/{ds}_test.y.npy')
    ktr, kte = keys(Xtr), keys(Xte)
    lab = defaultdict(set)
    for k, y in zip(ktr, ytr):
        lab[k].add(int(y))
    seen = np.array([k in lab for k in kte])
    conflict = np.array([k in lab and bool(lab[k] - {int(y)}) for k, y in zip(kte, yte)])
    r = dict(n_train=len(ktr), n_test=len(kte),
             dup_within_train=len(ktr) - len(set(ktr)), dup_within_test=len(kte) - len(set(kte)),
             test_seen_in_train=int(seen.sum()), pct_test_seen=round(100 * seen.mean(), 2),
             seen_with_conflicting_label=int(conflict.sum()),
             pct_seen_by_class={int(c): round(100 * seen[yte == c].mean(), 1) for c in np.unique(yte)})
    keep = ~seen
    for loss, samp in [('mse', 'epoch'), ('mse', 'frozen'), ('huber', 'frozen')]:
        full, dedup = [], []
        for s in SEEDS:
            f = {c: f'{PR}/{ds}_val-{w}-{sm}_{loss}_s{s}.npy'
                 for c, w, sm in [('PU', 'plain', 'none'), ('RU', 'freq', 'none'),
                                  ('PB', 'plain', samp), ('RB', 'freq', samp)]}
            if not all(os.path.exists(v) for v in f.values()):
                continue
            p = {c: np.load(v) for c, v in f.items()}
            inter = lambda m: (mr(yte[m], p['RB'][m]) - mr(yte[m], p['PB'][m])) - (mr(yte[m], p['RU'][m]) - mr(yte[m], p['PU'][m]))
            full.append(inter(np.ones(len(yte), bool))); dedup.append(inter(keep))
        if full:
            r[f'interaction_{loss}_{samp}'] = dict(n_seeds=len(full), all_test=round(float(np.mean(full)), 4),
                                                  unseen_test_only=round(float(np.mean(dedup)), 4),
                                                  negative_seeds_unseen=f'{sum(d < 0 for d in dedup)}/{len(dedup)}')
    out[ds] = r
print(json.dumps(out, indent=1))
