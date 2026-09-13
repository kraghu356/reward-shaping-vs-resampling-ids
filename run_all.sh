#!/usr/bin/env bash
# Reproduce every number in the paper. Expects data/NF-UNSW-NB15-v2.csv
set -euo pipefail
CSV=${1:-data/NF-UNSW-NB15-v2.csv}
[ -f "$CSV" ] || { echo "Missing $CSV -- see README (UQ NetFlow V2 download)"; exit 1; }
mkdir -p data/processed results figures

echo "== 1/6 preprocess 400k =="
python src/prepare_nf.py --input "$CSV" --dataset nf_unsw \
  --out data/processed/nf_unsw --cap 400000 --floor 1000 --seed 42 \
  --stream --chunksize 1000000

echo "== 2/6 fixed-budget pass, 9 methods =="
python src/nf_matrix.py --processed data/processed --datasets nf_unsw \
  --methods extra_trees rf xgboost ce focal dqn_plain_uniform \
            dqn_reward_uniform dqn_plain_balanced dqn_reward_balanced \
  --seeds 42 7 13 99 123 --out results/nf_unsw_full.json

echo "== 3/6 validation-tuned pass =="
python src/nf_tune.py --processed data/processed --dataset nf_unsw \
  --methods extra_trees dqn_plain_uniform dqn_reward_uniform \
            dqn_plain_balanced dqn_reward_balanced \
  --seeds 42 7 13 99 123 --out results/nf_unsw_tuned.json

echo "== 4/6 preprocess 100k replication =="
python src/prepare_nf.py --input "$CSV" --dataset nf_unsw_small \
  --out data/processed/nf_unsw_small --cap 100000 --floor 1000 --seed 42 \
  --stream --chunksize 1000000

echo "== 5/6 100k replication =="
python src/nf_tune.py --processed data/processed --dataset nf_unsw_small \
  --methods extra_trees dqn_plain_uniform dqn_reward_uniform \
            dqn_plain_balanced dqn_reward_balanced \
  --seeds 42 7 13 --out results/nf_unsw_small_tuned.json

echo "== 6/6 statistics and figures =="
python src/analyse.py | tee results/statistics.txt
python src/make_figures.py
echo "done -- see results/ and figures/"
