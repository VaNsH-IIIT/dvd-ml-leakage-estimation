# DVD Project: ML-driven PVT-aware leakage estimation

Course project (Digital VLSI Design, IIIT Hyderabad), team 22 ("Madhurbala_riders"). Spec: the official project document `DVD_ML_Project.pdf`.

| Item | Team 22 value |
|---|---|
| Model file | `models/45nm_MGK.pm` (PTM BSIM4, from the course `model.tar`) |
| Nominal VDD | 0.9 V (sampled 0.81–0.99 V) |
| Channel length | L = 45 nm; widths 1–8 × Lmin |
| Test circuit | 8-bit parity checker (7 XOR2 = 28 NAND2, 112 transistors) |
| Simulator | ngspice 42 (`/usr/bin/ngspice`), batch mode |

## What it does

1. **Part A: transistor models.** For NMOS ON, NMOS OFF, PMOS ON and PMOS OFF:
   - simulate 10,000 random samples in SPICE (VDD, T, W, Vd, Vs and five process parameters, drawn as the doc specifies), plus 80,000 extra training samples of the same distribution (regenerated, not committed);
   - train four model types, then larger-data MLPs, 3-MLP ensembles and averages of ensemble variants; pick the best on the validation set and report it on the test set.
2. **Part B: gates.** Compute leakage for INV, NAND2, NOR2, AND2 and OR2 (every input combination, 200 corners):
   - SPICE gives the reference;
   - the ML value comes from the transistor models, fed with the SPICE-recorded node voltages.
3. **Part C: test circuit.** The parity checker:
   - SPICE of the full netlist at the same corners, for all 256 input vectors;
   - the ML value is composed from the NAND2 gate models.

## Run everything (in order, fixed seeds)

```bash
source .venv/bin/activate
python scripts/verify_sim.py          # driver checks (professor's sample netlist, KCL, physics)
python scripts/a1_generate_data.py --extra 8   # Part A datasets + 8 training-only blocks per case (data/partA_extra/)
python scripts/a2_check_data.py       # data check, trend plots, example netlists
python scripts/a3_compare_models.py   # 4 model types x 4 cases (validation)
python scripts/a3b_data_size.py       # MLP vs training-set size (27k, 47k rows; 7k comes from a3)
python scripts/a3b_data_size.py --blocks --subsample 2000 4000   # ... and 2k, 4k rows
python scripts/a3b_data_size.py --kind mlp_ens3 --blocks 4 --out data_size_ens.csv   # 3-MLP ensembles (per-width targets), 47k rows (~50 min)
python scripts/a3b_data_size.py --kind mlp_ens3 --blocks 8 --out data_size_ens.csv   # 3-MLP ensembles (per-width targets), 87k rows (~1.5-2.5 h)
python scripts/a3b_data_size.py --kind mlp_ens3_cur --blocks 8 --out data_size_ens.csv   # same, plain-current targets (~1.5-2.5 h)
python scripts/a3b_data_size.py --kind mlp_ens3_lin --blocks 8 --out data_size_ens.csv   # same, linear fit + MLPs on its residual (~1.5-2.5 h)
python scripts/a1b_edge_data.py       # edge-sample experiment: data (rejected idea, kept for the report)
python scripts/a3b_data_size.py --blocks 0 --edge 1 --out edge_study.csv   # edge-sample experiment: training
python scripts/a3c_mixed_scale_study.py   # target-scale study for the mixed-sign OFF terminal
python scripts/a4_select_models.py    # average the 87k ensemble variants, choose on validation, score on test, save -> trained_models/
python scripts/b1_gates_spice.py      # gate SPICE (200 corners)
python scripts/b2_gates_ml.py         # gate ML vs SPICE
python scripts/b3_leakage_components.py   # what the gate leakage consists of (gate tunnelling vs sub-threshold)
python scripts/d2_gate_diagnostics.py # gates with the non-chosen candidates (for the report's discussion)
python scripts/c1_parity_spice.py     # parity checker SPICE (51,200 runs, ~5 min)
python scripts/c2_parity_ml.py        # parity checker ML vs SPICE
python scripts/d1_error_analysis.py   # error analysis -> reports/ERROR_ANALYSIS.md
python scripts/d3_fresh_check.py      # final pipeline on fresh seeds -> reports/FRESH_CHECK.md (~5 min)
python scripts/e1_build_report.py     # reports/REPORT.md -> reports/REPORT.pdf (headless Chrome)
python scripts/demo_predict.py NMOS_OFF --temp 85   # demo: one operating point, ML vs live SPICE
```

Memory note: the Gaussian-process fits need about 2 GB each and run at most two at a time. Running four at once (each fitting its outputs in parallel) exhausted the 23 GB development machine.

## Repository layout

| Path | Contents |
|---|---|
| `DVD_ML_Project.pdf` | Official project document |
| `problem-statement/` | Professor's original files: annotated slides, sample netlist, `model.tar` (unchanged) |
| `models/` | Extracted PTM model cards (`45nm_MGK.pm` is ours) |
| `sim/` | `spec.py` (team assignment and sampling spec), single-FET driver (`runner.py`), gates (`gates.py`), parity checker (`parity.py`) |
| `ml/` | Features and targets (`data.py`), model types (`models.py`), gate-leakage composition (`gate_eval.py`), metrics |
| `scripts/` | The pipeline above: `verify_sim.py`; `a*` Part A; `b*` Part B; `c*` Part C; `d*` error analysis, gate diagnostics and fresh-data check; `e1` report build; `demo_predict.py` |
| `data/partA/` | The four 10,000-row datasets with the doc's columns |
| `data/partB/`, `data/partC/` | Corners, SPICE references, ML predictions |
| `netlists/` | Example netlists: one per transistor case, per gate, and the test circuit |
| `trained_models/` | The four chosen models, averages of 3-MLP ensemble variants (`joblib.load(...).predict(df)` → id, ig, is, ib) |
| `reports/` | `REPORT.pdf` and its source `REPORT.md` (the typed report), `ERROR_ANALYSIS.md`, `FRESH_CHECK.md`, `REGENERATION.md`, `partA/DATA_CHECK.md`, `partA/MODELS.md`, `partB/GATES.md`, `partB/GATES_DIAGNOSTICS.md`, `partB/LEAKAGE_COMPONENTS.md`, `partC/PARITY.md`, with their figures |

## Setup

```bash
sudo apt install -y ngspice                      # SPICE simulator (v42 works with the BSIM4 models)
# google-chrome is needed only to rebuild reports/REPORT.pdf (scripts/e1_build_report.py)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt
```

## Workflow

- `main` holds reviewed work only.
- Each piece of work lands as a **pull request**, so teammates can review it before merging.
