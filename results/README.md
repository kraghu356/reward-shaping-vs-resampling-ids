# Result files

Populate by running `bash run_all.sh` (see the top-level README).

| File | Produced by | Feeds |
|---|---|---|
| `nf_unsw_full.json` | `src/nf_matrix.py` | Tables 5, 13 |
| `nf_unsw_tuned.json` | `src/nf_tune.py` | Tables 6, 7, 8, 9, 12 |
| `nf_unsw_small_tuned.json` | `src/nf_tune.py` | Table 10 |
| `statistics.txt` | `src/analyse.py` | every reported contrast |
| `meta/*.meta.json` | `src/prepare_nf.py` | subsample audit trail |

Each JSON holds one record per (method, seed) with the per-seed metrics, so
every paired test in the paper can be recomputed independently.
