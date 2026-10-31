# Fresh-data check

The final pipeline scored on data generated with seeds used nowhere else in the project (`scripts/d3_fresh_check.py`). An earlier fresh set (seeds 777000-777003, 888) showed the errors at hot, wide, low-vth0 corners that led to the per-width and linear-baseline model variants, so it is no longer independent; this set (seeds 779000-779003, 999) was drawn after that.

## 1. Transistors: 2,000 new samples per case

| case | median | p95 | median without ib | p95 without ib |
|---|---|---|---|---|
| NMOS_ON | 0.12 % | 0.57 % | 0.12 % | 0.61 % |
| NMOS_OFF | 0.24 % | 3.26 % | 0.20 % | 1.10 % |
| PMOS_ON | 0.12 % | 0.51 % | 0.12 % | 0.52 % |
| PMOS_OFF | 0.22 % | 2.70 % | 0.17 % | 0.83 % |

## 2. Gates: 100 new corners x 18 cases

All 1,800 simulations: median 0.37 %, p95 1.39 %, mean 0.50 %, 90 % within 1 %.
Worst cases (median): NOR2 11: 0.53 %, NAND2 00: 0.52 %, NAND2 10: 0.45 %.

## 3. Parity checker: the same 100 corners x 256 vectors

All 25,600 runs (sum of NAND2 gate-model leakages): median 0.41 %, p95 1.47 %, mean 0.52 %.
