# ML-Driven PVT-Variation-Aware Leakage Estimation

Digital VLSI Design course project, IIIT Hyderabad. Team 22 (**Madhurbala_riders**).

We train machine-learning models of single-transistor leakage currents from SPICE data, then use them to predict the leakage of logic gates and of an 8-bit parity checker under random voltage, temperature, process and width variation, and check every prediction against SPICE.

**Report:** [`reports/REPORT.pdf`](reports/REPORT.pdf) (source [`reports/REPORT.tex`](reports/REPORT.tex)). **Project document:** [`DVD_ML_Project.pdf`](DVD_ML_Project.pdf).

| Name | Roll number | Part |
|---|---|---|
| Aryan Agrawal | 2023102050 | Simulation setup and Part A data |
| Vansh Jaiswal | 2024102013 | Part A transistor models |
| Saarthak Sabharwal | 2023102055 | Part B gate leakage |
| Manikya Pant | 2023112024 | Part C parity checker, error analysis and report |

## Results at a glance

| | Median error | p95 error | Mean error | Mean accuracy |
|---|---|---|---|---|
| Transistor models (test set, per case) | 0.12–0.23 % | 0.53–3.16 % | – | – |
| Gates: INV, NAND2, NOR2, AND2, OR2 (3,600 SPICE runs) | 0.37 % | 1.33 % | **0.51 %** | **99.49 %** |
| 8-bit parity checker (51,200 SPICE runs) | 0.38 % | 1.35 % | **0.51 %** | **99.49 %** |
| Fresh corners, gates / parity checker | 0.37 / 0.41 % | 1.39 / 1.47 % | 0.50 / 0.52 % | 99.50 / 99.48 % |

Error = \|ML − SPICE\| / \|SPICE\| per sample. The transistor-model p95 in the OFF cases comes from the ~1e-14 A body current; the currents that matter for leakage have p95 ≤ 1.4 %.

## Our assignment

| Item | Team 22 value |
|---|---|
| Model file | `models/45nm_MGK.pm` (PTM, BSIM4 level 54, from the course `model.tar`) |
| Nominal VDD | 0.9 V (sampled 0.81–0.99 V) |
| Channel length, widths | L = 45 nm; W = 1–8 × Lmin = 45–360 nm |
| Temperature, process | −40 to 125 °C; `vth0`, `toxe`, `ndep`, `u0`, `xj` at nominal ± 10 % |
| Test circuit | 8-bit parity checker (7 XOR2 = 28 NAND2, 112 transistors) |
| Simulator | ngspice 42, DC operating point |

## How it works

```
 SPICE, one transistor  ──train──>  4 case models  ──>  gate leakage  ──>  parity checker
 (random PVT, Vd, Vs)               (Id, Ig, Is, Ib)    (KCL over a cut)    (sum of 28 NAND2)
     Part A data                    Part A models        Part B              Part C
```

1. **Part A: transistor models.** For NMOS ON, NMOS OFF, PMOS ON and PMOS OFF we simulate 10,000 random samples (plus 80,000 extra training samples of the same distribution), compare four techniques (polynomial, LightGBM, MLP, Gaussian process) and keep the best: an average of 3-MLP ensemble variants, chosen on the validation set.
2. **Part B: gates.** For every input combination at 200 random corners, the transistor models predict each transistor's currents from the SPICE node voltages, and Kirchhoff's law over a cut through the OFF transistors turns them into the gate leakage.
3. **Part C: parity checker.** The circuit leakage is the sum of its 28 NAND2 gate-model leakages, compared with SPICE of the full circuit for all 256 input vectors.

Leakage is the total current drawn from VDD-level sources (the VDD supply plus inputs held at VDD, i.e. static power / VDD). With this definition a circuit's leakage is the sum of its gates' leakages (0.18 % median difference in SPICE).

## Setup

```bash
sudo apt install -y ngspice                      # SPICE simulator (v42 works with the BSIM4 models)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt
```

To rebuild the report PDF you also need LaTeX, e.g. [Tectonic](https://tectonic-typesetting.github.io) (`tectonic reports/REPORT.tex`) or `pdflatex`.

## Quick check (a few minutes)

```bash
python scripts/verify_sim.py                       # simulation driver vs the professor's sample netlist -> ALL CHECKS PASSED
python scripts/demo_predict.py NMOS_OFF --temp 85  # one operating point: ML vs live SPICE
python scripts/b2_gates_ml.py                      # gates, ML vs SPICE (~1 min) -> median 0.37 %, mean 0.51 %
python scripts/c2_parity_ml.py                     # parity checker, ML vs SPICE -> median 0.38 %, mean 0.51 %
```

These use the committed data and trained models and leave the repository unchanged.

## Full run (in order, fixed seeds)

```bash
python scripts/verify_sim.py          # driver checks (professor's sample netlist, KCL, physics)
python scripts/a1_generate_data.py --extra 8   # Part A datasets + 8 training-only blocks per case (data/partA_extra/)
python scripts/a2_check_data.py       # data check, trend plots, example netlists
python scripts/a3_compare_models.py   # 4 model types x 4 cases (validation)
python scripts/a3b_data_size.py       # MLP vs training-set size (27k, 47k rows; 7k comes from a3)
python scripts/a3b_data_size.py --blocks --subsample 2000 4000   # ... and 2k, 4k rows
python scripts/a3b_data_size.py --kind mlp_ens3 --blocks 4 --out data_size_ens.csv       # 3-MLP ensembles, 47k rows (~50 min)
python scripts/a3b_data_size.py --kind mlp_ens3 --blocks 8 --out data_size_ens.csv       # per-width targets, 87k rows (~1.5-2.5 h)
python scripts/a3b_data_size.py --kind mlp_ens3_cur --blocks 8 --out data_size_ens.csv   # plain-current targets (~1.5-2.5 h)
python scripts/a3b_data_size.py --kind mlp_ens3_lin --blocks 8 --out data_size_ens.csv   # linear baseline + MLPs (~1.5-2.5 h)
python scripts/a1b_edge_data.py       # edge-sample experiment: data (rejected idea, kept for the report)
python scripts/a3b_data_size.py --blocks 0 --edge 1 --out edge_study.csv   # edge-sample experiment: training
python scripts/a3c_mixed_scale_study.py   # target-scale study for the mixed-sign OFF terminal
python scripts/a4_select_models.py    # average the ensemble variants, choose on validation, score on test -> trained_models/
python scripts/b1_gates_spice.py      # gate SPICE (200 corners)
python scripts/b2_gates_ml.py         # gate ML vs SPICE
python scripts/b3_leakage_components.py   # what the gate leakage consists of (gate tunnelling vs sub-threshold)
python scripts/d2_gate_diagnostics.py # gates with the non-chosen candidates (for the report's discussion)
python scripts/c1_parity_spice.py     # parity checker SPICE (51,200 runs, ~5 min)
python scripts/c2_parity_ml.py        # parity checker ML vs SPICE
python scripts/d1_error_analysis.py   # error analysis -> reports/ERROR_ANALYSIS.md
python scripts/d3_fresh_check.py      # final pipeline on fresh seeds -> reports/FRESH_CHECK.md (~5 min)
tectonic reports/REPORT.tex           # the report -> reports/REPORT.pdf
```

The Gaussian-process fits need about 2 GB of memory each; run at most two at a time on a machine with less than 32 GB.

## Repository layout

| Path | Contents |
|---|---|
| `DVD_ML_Project.pdf` | Official project document |
| `problem-statement/` | Professor's original files: annotated slides, sample netlist, `model.tar` (unchanged) |
| `models/` | Extracted PTM model cards (`45nm_MGK.pm` is ours) |
| `sim/` | Team spec and sampling (`spec.py`), single-transistor driver (`runner.py`), gates (`gates.py`), parity checker (`parity.py`) |
| `ml/` | Features and targets (`data.py`), model types (`models.py`), gate-leakage composition (`gate_eval.py`), metrics |
| `analysis/` | Shared plot style |
| `scripts/` | The pipeline above: `verify_sim.py`; `a*` Part A; `b*` Part B; `c*` Part C; `d*` error analysis, gate diagnostics and fresh-data check; `demo_predict.py` |
| `data/partA/` | The four 10,000-row datasets with the project document's columns |
| `data/partA_results/` | Model comparison, data-size and ensemble results |
| `data/partB/`, `data/partC/` | Corners, SPICE references, ML predictions |
| `netlists/` | Example netlists: one per transistor case, per gate, and the test circuit |
| `trained_models/` | The four chosen models (`joblib.load(...).predict(df)` → id, ig, is, ib) |
| `reports/` | `REPORT.pdf` / `REPORT.tex` (the report) and the detailed result pages: `partA/DATA_CHECK.md`, `partA/MODELS.md`, `partB/GATES.md`, `partB/LEAKAGE_COMPONENTS.md`, `partB/GATES_DIAGNOSTICS.md`, `partC/PARITY.md`, `ERROR_ANALYSIS.md`, `FRESH_CHECK.md`, `REGENERATION.md`, with their figures |
| `DELIVERABLES.md` | Checklist of the project document's deliverables and where each one is |

## Workflow

- `main` holds reviewed work only.
- Each part went in as its own commit; the pull requests were reviewed and merged by the owner of each part.
