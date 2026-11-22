# Reward weighting versus class-balanced sampling — reproduction package

Code, per-seed results and metadata for the manuscript *Separating Reward Weighting
from Class-Balanced Sampling in Cost-Sensitive Value Regression for Network Intrusion
Detection: A Factorial Study*.

Code state for the current manuscript: commit `7dd4c95` on `main` (to be tagged `reward-vs-resampling-v2.0`).
The earlier single-corpus package is kept under tag `reward-vs-resampling-v1.0`
(files `run_all.sh`, `src/nf_matrix.py`, `src/nf_tune.py`, `src/analyse.py`,
`results/nf_unsw_*.json`); it produces the baseline specification and the
fixed-budget pass reported in the appendices.

## What is in the package

| Path | Content |
|---|---|
| `src/prepare_nf.py` | streaming quota subsampler (Eq. 7), stratified 70/30 split, source-row ids, label map for all three corpora |
| `src/nf_revision.py` | every learner of the paper: value head (squared or Huber loss), weighted cross-entropy, epsilon-greedy bandit with replay, three samplers, tuned trees; threshold-free, recalibrated, logit-adjusted and prior-corrected (Corollary 1) metrics |
| `src/rev_analyse.py` | paired contrasts, exact sign-flip tests, Poisson test-set bootstrap, overlap audit, distinct-flow counts |
| `src/leakage_audit.py` | exact-duplicate audit across train/test and the factorial re-scored on unseen test flows |
| `src/make_revision_tables.py` | regenerates every table and inline number of the manuscript from the result files |
| `run_revision.sh` | runs all reported experiments in order (groups A–I) |
| `colab/paper2_revision_all.py` | the resumable Colab driver that was used to run the same groups |
| `results/revision/*.json` | per-seed results for every condition (one record per method and seed) |
| `results/revision/leakage_audit.json` | output of `src/leakage_audit.py` (duplicate audit, factorial on unseen test flows) |
| `results/revision/meta/*.meta.json` | subsample metadata: class counts, corpus counts, planned quotas, quota rule |

## Experiment groups

| Group | Configuration | Seeds | Output |
|---|---|---|---|
| A | value head, squared loss, frozen / per-epoch / importance-weighted samplers | 10 | `rev_nf_unsw_mse.json` |
| B | weighted cross-entropy 2×2, logit adjustment, tuned Extra Trees / Random Forest / XGBoost | 10 | `rev_nf_unsw_ce.json` |
| C | baseline specification, Huber κ = 1 | 10 | `rev_nf_unsw_huber1.json` |
| G | ε-greedy contextual bandit with replay | 10 | `rev_nf_unsw_bandit.json` |
| D | α ∈ {1,2,5,10}, κ ∈ {0.5,1,2,5,10,(α+1)λmax}, fixed configuration | 5 | `rev_sweep_*.json` |
| H | wider grid for PB and RB | 5 | `rev_nf_unsw_widegrid.json` |
| E | NF-ToN-IoT-v2, NF-BoT-IoT-v2 | 10 | `rev_nf_ton_mse.json`, `rev_nf_bot_mse.json` |
| F | separately seeded 100k draw (seed 2027; 17.6% of its source rows also occur in the primary draw) | 10 | `rev_nf_unsw_small2_mse.json` |
| I | true corpus prior (no floor) | 10 | `rev_nf_unsw_trueprior_mse.json` |

## Data

The corpora are not redistributed. Download NF-UNSW-NB15-v2, NF-ToN-IoT-v2 and
NF-BoT-IoT-v2 (NetFlow V2 section) from <https://staff.itee.uq.edu.au/marius/NIDS_datasets/>
and place the CSV files in `data/`. The UQ downloads are zip archives with hashed names;
unzip them first.

## Running

```bash
pip install -r requirements.txt
bash run_revision.sh
```

Each call resumes from its output file. Saved test predictions (`results/revision/preds/`,
about 1 GB) are not committed; `run_revision.sh` writes them, and the statistics and the
leakage audit read them.

## Notes

* The learner is one-step, full-information cost-sensitive value regression. The bandit
  experiment (group G) is the only arm in which the agent observes the reward of its own
  action alone.
* TensorFlow GPU kernels are not bit-deterministic; neural results can differ in the third
  or fourth decimal place across machines.
* The reported runs used Google Colab (NVIDIA T4 and L4 GPUs). `requirements.txt` pins the
  reference versions.

## License

MIT — see LICENSE. Citation metadata in CITATION.cff.
