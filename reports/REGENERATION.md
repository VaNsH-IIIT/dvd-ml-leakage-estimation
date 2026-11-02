# Regeneration check

## Final version (27 Oct 2026)

After the TAs' answers the gates and the parity checker were re-simulated with random widths and the total-leakage definition, and the transistor models were retrained (three 87,000-row ensemble variants and their averages). Checks on this version:
- **Training is deterministic.** The plain-current 87,000-row ensembles were retrained from scratch for three of the four cases (the NMOS ON run was interrupted, so the earlier run of the same model is used, relabelled as `mlp_ens3_cur`; its predictions are unchanged); the three agree with the earlier runs to all printed digits of the validation metrics.
- **Selection is deterministic.** `a4_select_models.py` run twice gives the same choice and test metrics.
- **Downstream outputs reproduce.** In a staged rebuild of the repository (a fresh copy of the committed files), `b2_gates_ml.py`, `c2_parity_ml.py`, `d1_error_analysis.py` and `d3_fresh_check.py` rerun with the saved models leave every committed file unchanged.
- A full clean-clone rerun including all training takes about 6 hours on 16 cores and was done for the earlier version below.

## Earlier version (25 Oct 2026)

**Result: rerunning every script from a clean clone, including all model training, reproduces every committed result.** The only differences are the training-time column and one diagnostic value at the 10th significant digit.

**Run:** 25 Oct 2026, 03:27–04:48 (1 h 21 min on 16 cores). A fresh `git clone` of the repository as it stood before the review fixes; then every script in README order from `verify_sim.py` to `d1_error_analysis.py`, with no cached files.

| Output | Comparison with the committed file |
|---|---|
| `data/partA/*.csv`, `data/partA_extra/*` (regenerated, not committed: compared with the local copy), `data/partB/corners.csv`, `data/partB/gates_spice.csv`, `data/partC/parity_spice.csv`, all netlists, data-check report and plots | Byte-identical |
| `data/partA_results/compare_val.csv` (80 rows), `data_size.csv` (the 80 rows of that time) | Every metric identical. Only `fit_s` (wall-clock training time) differs. |
| `data/partA_results/chosen.csv`: model choice and test metrics | Identical |
| `data/partB/gates_ml.csv`: 3,600 ML gate leakages | Identical; one diagnostic column (`err_vddsum`) differs in one row by 1.7e-10 relative (floating-point rounding) |
| `data/partC/parity_ml.csv`: 51,200 ML circuit leakages | Identical |

The scripts and rows added in the review (`a3c_mixed_scale_study.py`, `d2_gate_diagnostics.py`, `d3_fresh_check.py`, and the 2,000/4,000-row data-size runs) postdate this run. They were run on the committed state, and an independent re-execution of `a1` (NMOS OFF), `a2`, `a4`, `b2`, `c2`, `d1` and `d3` in a scratch copy reproduced their outputs byte-identically.
