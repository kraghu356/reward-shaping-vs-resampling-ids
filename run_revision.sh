#!/usr/bin/env bash
# Extended experiments of the manuscript (Sections 6-7, Tables 9-11, 14).
# This script runs exactly the configurations reported in the paper; the Colab driver
# colab/paper2_revision_all.py runs the same calls in resumable groups (A-I).
# Every call resumes from its output JSON, so an interrupted run loses at most one cell.
set -euo pipefail
P=data/processed
R=results/revision
SEEDS="42 7 13 99 123 2024 31 57 88 101"
S5="42 7 13 99 123"
VAL="val-plain-none val-freq-none val-plain-frozen val-freq-frozen val-plain-epoch val-freq-epoch val-plain-iw val-freq-iw"
VAL6="val-plain-none val-freq-none val-plain-frozen val-freq-frozen val-plain-epoch val-freq-epoch"
CE="ce-plain-none ce-freq-none ce-plain-frozen ce-freq-frozen ce-plain-epoch ce-freq-epoch ce-plain-iw ce-freq-iw"
BANDIT="bandit-plain-none bandit-freq-none bandit-plain-frozen bandit-freq-frozen bandit-plain-epoch bandit-freq-epoch"
SW="val-plain-none val-freq-none val-plain-epoch val-freq-epoch"
PREDS="--save-preds $R/preds"
mkdir -p $R/preds

# --- preprocessing (raw CSVs from https://staff.itee.uq.edu.au/marius/NIDS_datasets/) ---
prep () { [ -f $P/$2_train.X.npy ] || python src/prepare_nf.py --input data/$1 --dataset $2 \
            --out $P/$2 --cap ${4:-400000} --floor ${5:-1000} --seed $3 --stream --chunksize 1000000; }
prep NF-UNSW-NB15-v2.csv nf_unsw            42
prep NF-UNSW-NB15-v2.csv nf_unsw_regen      42              # same seed: reproduces nf_unsw + row ids
prep NF-UNSW-NB15-v2.csv nf_unsw_small2     2027 100000      # separately seeded 100k draw
prep NF-UNSW-NB15-v2.csv nf_unsw_trueprior  42 400000 0      # no floor: corpus prior
prep NF-ToN-IoT-v2.csv   nf_ton             42
prep NF-BoT-IoT-v2.csv   nf_bot             42

run () { python src/nf_revision.py --processed $P "$@"; }

echo "== A  value regression, squared loss, three samplers =="
run --dataset nf_unsw --methods $VAL --seeds $SEEDS --loss mse --alpha 5 $PREDS --out $R/rev_nf_unsw_mse.json
echo "== B  weighted cross-entropy 2x2, logit adjustment, tuned trees =="
run --dataset nf_unsw --methods $CE extra_trees rf xgboost --seeds $SEEDS $PREDS --out $R/rev_nf_unsw_ce.json
echo "== C  baseline specification: Huber kappa=1 =="
run --dataset nf_unsw --methods $VAL6 --seeds $SEEDS --loss huber --kappa 1.0 --alpha 5 $PREDS --out $R/rev_nf_unsw_huber1.json
echo "== G  epsilon-greedy contextual bandit with replay =="
run --dataset nf_unsw --methods $BANDIT --seeds $SEEDS --grid fixed --loss mse --alpha 5 $PREDS --out $R/rev_nf_unsw_bandit.json
echo "== D  alpha and kappa sweeps (fixed configuration, 5 seeds) =="
for AL in 1 2 5 10; do
  run --dataset nf_unsw --methods $SW --seeds $S5 --grid fixed --loss mse --alpha $AL --out $R/rev_sweep_alpha$AL.json
done
for KA in 0.5 1 2 5 10 auto; do
  run --dataset nf_unsw --methods $SW --seeds $S5 --grid fixed --loss huber --kappa $KA --alpha 5 --out $R/rev_sweep_kappa$KA.json
done
echo "== H  wider grid for PB and RB (5 seeds) =="
run --dataset nf_unsw --methods val-plain-frozen val-freq-frozen val-plain-epoch val-freq-epoch --seeds $S5 \
    --grid wide --loss mse --alpha 5 --tag _wide $PREDS --out $R/rev_nf_unsw_widegrid.json
echo "== E  further corpora =="
for DS in nf_ton nf_bot; do
  run --dataset $DS --methods $VAL6 ce-plain-none extra_trees --seeds $SEEDS --loss mse --alpha 5 $PREDS --out $R/rev_${DS}_mse.json
done
echo "== F  separately seeded 100k draw =="
run --dataset nf_unsw_small2 --methods $VAL6 --seeds $SEEDS --loss mse --alpha 5 --out $R/rev_nf_unsw_small2_mse.json
echo "== I  true corpus prior =="
run --dataset nf_unsw_trueprior --methods $VAL6 --seeds $SEEDS --loss mse --alpha 5 --out $R/rev_nf_unsw_trueprior_mse.json

echo "== statistics, leakage audit and manuscript tables =="
python src/rev_analyse.py $R/rev_*.json --preds $R/preds --processed $P \
    --overlap $P/nf_unsw_regen $P/nf_unsw_small2 | tee $R/rev_statistics.txt
P=$P PR=$R/preds python -c "import os;P=os.environ['P'];PR=os.environ['PR'];exec(open('src/leakage_audit.py').read())" \
    | tee $R/leakage_audit.json
python src/make_revision_tables.py --results $R --out $R/tables --preds $R/preds --processed $P
