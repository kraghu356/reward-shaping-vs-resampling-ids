# Reward Shaping or Resampling? — reproduction package

Code and results for the paper *Reward Shaping or Resampling? Disentangling the
Source of Rare-Class Gains in Deep Reinforcement Learning for Network Intrusion
Detection*.

This package reproduces **only** that paper: the 2×2 reward × sampler factorial
on NF-UNSW-NB15-v2. It is deliberately separate from the cross-dataset
CNN-LSTM-DQN work in the parent repository, which is a different study with a
different experimental design.

Release tag: `reward-vs-resampling-v1.0`

---

## What the paper claims, and where each number comes from

| Paper item | Produced by | Result file |
|---|---|---|
| Table 5 (fixed-budget, 9 methods) | `src/nf_matrix.py` | `results/nf_unsw_full.json` |
| Table 6 (validation-tuned) | `src/nf_tune.py` | `results/nf_unsw_tuned.json` |
| Table 7 (factorial contrasts, CIs) | `src/analyse.py` | derived from the above |
| Table 8 (precision / recall / FPR) | `src/nf_tune.py` | `results/nf_unsw_tuned.json` |
| Table 9 (per-family recall, F1) | `src/nf_tune.py` | `results/nf_unsw_tuned.json` |
| Table 10 (100k replication) | `src/nf_tune.py` | `results/nf_unsw_small_tuned.json` |
| Table 12 (tuned per-seed) | — | `results/nf_unsw_tuned.json` |
| Table 13 (fixed-budget per-seed) | — | `results/nf_unsw_full.json` |
| Figures 1–3 | `src/make_figures.py` | `figures/*.png` |

Every table in the paper is derived from one of the three JSON files in
`results/`. Those files contain **per-seed** values, so the paired tests can be
recomputed independently rather than taken on trust.

---

## Requirements

```
python >= 3.10
numpy, pandas, scipy, scikit-learn, xgboost
tensorflow >= 2.15   (GPU optional; see note on hardware below)
matplotlib           (figures only)
```

Install:

```bash
pip install -r requirements.txt
```

Hardware used for the reported runs: a single NVIDIA Tesla T4 (Google Colab)
for the neural conditions; tree ensembles are CPU-only. The factorial runs on
CPU but takes roughly 5–8× longer.

---

## Data

NF-UNSW-NB15-v2 is not redistributed here. Download it from the authors of the
NetFlow standardisation work:

<https://staff.itee.uq.edu.au/marius/NIDS_datasets/>

Use the **NetFlow V2** section, entry `NF-UNSW-NB15-v2`. The expected file is
`NF-UNSW-NB15-v2.csv`, 2,390,275 flows, 3.98% attack.

---

## Reproducing the main experiment

One command, after placing the CSV at `data/NF-UNSW-NB15-v2.csv`:

```bash
bash run_all.sh
```

Or step by step:

```bash
# 1. Preprocess: 400k imbalance-preserving subsample, seed 42
python src/prepare_nf.py \
    --input data/NF-UNSW-NB15-v2.csv --dataset nf_unsw \
    --out data/processed/nf_unsw \
    --cap 400000 --floor 1000 --seed 42 --stream --chunksize 1000000

# 2. Fixed-budget pass, all nine methods (Table 5, Table 13)
python src/nf_matrix.py --processed data/processed --datasets nf_unsw \
    --methods extra_trees rf xgboost ce focal \
              dqn_plain_uniform dqn_reward_uniform \
              dqn_plain_balanced dqn_reward_balanced \
    --seeds 42 7 13 99 123 --out results/nf_unsw_full.json

# 3. Validation-tuned pass (Tables 6-9, Table 12)
python src/nf_tune.py --processed data/processed --dataset nf_unsw \
    --methods extra_trees dqn_plain_uniform dqn_reward_uniform \
              dqn_plain_balanced dqn_reward_balanced \
    --seeds 42 7 13 99 123 --out results/nf_unsw_tuned.json

# 4. Independent 100k replication (Table 10)
python src/prepare_nf.py \
    --input data/NF-UNSW-NB15-v2.csv --dataset nf_unsw_small \
    --out data/processed/nf_unsw_small \
    --cap 100000 --floor 1000 --seed 42 --stream --chunksize 1000000
python src/nf_tune.py --processed data/processed --dataset nf_unsw_small \
    --methods extra_trees dqn_plain_uniform dqn_reward_uniform \
              dqn_plain_balanced dqn_reward_balanced \
    --seeds 42 7 13 --out results/nf_unsw_small_tuned.json

# 5. All statistics in the paper (Table 7 and the reported contrasts)
python src/analyse.py

# 6. Figures 1-3
python src/make_figures.py
```

---

## The four factorial conditions

| Label | Reward | Sampler | Method name in code |
|---|---|---|---|
| PU | `λ_c ≡ 1` | natural | `dqn_plain_uniform` |
| RU | `λ_c = log(N/N_c)` | natural | `dqn_reward_uniform` |
| PB | `λ_c ≡ 1` | class-balanced | `dqn_plain_balanced` |
| RB | `λ_c = log(N/N_c)` | class-balanced | `dqn_reward_balanced` |

RB is the configuration that corresponds to a published
frequency-aware-reward agent; PU is the usual control. The paper's point is
that comparing only those two confounds the reward with the sampler.

---

## Reproducibility notes, including the honest caveats

**Determinism.** Seeds fix the subsample, the train/test/validation splits, the
balanced sampler draws and the network initialisation. TensorFlow GPU kernels
are not bit-deterministic by default, so neural results may differ in the third
or fourth decimal place between machines. The reported effects (+0.142, +0.034,
−0.005) are larger than that variation, but the small RB−PB contrast is within
an order of magnitude of it, which is one reason the paper reports it with a
confidence interval and per-seed values rather than a point estimate alone.

**Feature exclusions.** `MIN_TTL` and `MAX_TTL` are dropped by default, as are
IP addresses, ports and timestamps. Passing `--keep-ttl` reproduces the
higher-but-leaking numbers that some published results correspond to; it is off
by default for a reason.

**Subsample provenance.** `prepare_nf.py` writes a `*_train.meta.json` and
`*_test.meta.json` recording the class quota, the retained counts, the cap, the
floor, the seed and the exact feature list used. Those files are the audit
trail for the sampling step; regenerate and diff them against the copies in
`results/meta/` to confirm an identical subsample.

**Scope.** The paper's validation-tuned analysis covers the four factorial
conditions plus Extra Trees. Cross-entropy, focal loss, Random Forest and
gradient-boosted trees appear only in the fixed-budget pass. Step 2 above
reproduces all nine under identical fixed-budget settings; comparisons across
the two passes are not like-for-like and the paper does not make them.

**What this package does not contain.** Results on a second corpus. The paper
states this limitation directly: the factorial is replicated across independent
subsamples of NF-UNSW-NB15-v2, not across datasets or class priors.

---

## Citation

If you use this code, please cite the paper. Please also cite the NetFlow
standardisation work that the feature schema comes from:

> Sarhan, M., Layeghy, S., Portmann, M. (2022). Towards a standard feature set
> for network intrusion detection system datasets. *Mobile Networks and
> Applications*, 27(1), 357–370.
