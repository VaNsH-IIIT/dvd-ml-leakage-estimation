# Part B diagnostic: gates with other model candidates

Not used for model selection (that is done on the Part A validation set). Each row uses the same 3,600 gate simulations and the best-cut sum; the vdd-node columns show the same models with the currents summed at the vdd node instead (`ml/gate_eval.py`).

| transistor models | median | p95 | mean | max | vdd-node sum: median | vdd-node sum: per-case medians | worst case (best cut) |
|---|---|---|---|---|---|---|---|
| LightGBM, 7,000 rows | 7.49 % | 30.88 % | 10.40 % | 52.4 % | 84.0 % | 6.4–610 % | INV 0 |
| MLP, 7,000 rows | 1.88 % | 8.82 % | 2.75 % | 27.4 % | 14.3 % | 1.8–91 % | NOR2 11 |
| MLP, 7,000 rows + 10,000 edge rows (rejected experiment) | 2.14 % | 9.95 % | 3.26 % | 35.8 % | 1.3 % | 0.4–6 % | NAND2 11 |
| MLP, 27,000 rows | 1.06 % | 4.11 % | 1.43 % | 10.6 % | 3.1 % | 1.1–17 % | NOR2 11 |
| MLP, 47,000 rows | 0.86 % | 3.40 % | 1.24 % | 15.0 % | 2.3 % | 0.9–13 % | INV 0 |
| Gaussian process, 7,000 rows | 12.63 % | 56.41 % | 18.70 % | 110.9 % | 136.6 % | 16.1–23005 % | NOR2 11 |
| 3-MLP ensemble, per-width targets, 47,000 rows | 0.66 % | 2.70 % | 0.97 % | 12.7 % | 2.3 % | 0.7–16 % | NOR2 11 |
| 3-MLP ensemble, per-width targets, 87,000 rows | 0.62 % | 2.95 % | 1.03 % | 21.3 % | 2.5 % | 0.6–13 % | INV 1 |
| 3-MLP ensemble, plain-current targets, 87,000 rows | 0.51 % | 1.95 % | 0.74 % | 14.2 % | 3.0 % | 0.5–19 % | NOR2 11 |
| 3-MLP ensemble on a linear fit's residual, 87,000 rows | 0.51 % | 2.24 % | 0.74 % | 4.7 % | 2.8 % | 0.4–13 % | NAND2 00 |
| **chosen: averages of 3-MLP ensemble variants, 87,000 rows (per case: `chosen.csv`)** | 0.37 % | 1.33 % | 0.51 % | 6.9 % | 2.4 % | 0.4–14 % | NAND2 00 |

