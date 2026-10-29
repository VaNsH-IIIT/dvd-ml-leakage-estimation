"""Gate-level diagnostics with model types that were NOT chosen.

The chosen models are selected on the Part A validation set only. This
script records, for the report's discussion, how other candidates behave
when used in the gates: it runs scripts/b2_gates_ml.py <kind> for each
candidate cached in runs/partA/ (written by a3_compare_models.py,
a3b_data_size.py and the edge study) and tabulates the results.

Writes reports/partB/GATES_DIAGNOSTICS.md (and per-kind details in
runs/partB_diag/<kind>/).
Run:  python scripts/d2_gate_diagnostics.py   (after a3, a3b and the edge study)
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "scripts"))

from a4_select_models import label  # noqa: E402
from ml import metrics  # noqa: E402

KINDS = [
    ("lgbm", "LightGBM, 7,000 rows"),
    ("mlp", "MLP, 7,000 rows"),
    ("mlp_edge1_n17000", "MLP, 7,000 rows + 10,000 edge rows (rejected experiment)"),
    ("mlp_n27000", "MLP, 27,000 rows"),
    ("mlp_n47000", "MLP, 47,000 rows"),
    ("gp", "Gaussian process, 7,000 rows"),
    ("mlp_ens3_n47000", "3-MLP ensemble, per-width targets, 47,000 rows"),
    ("mlp_ens3_n87000", "3-MLP ensemble, per-width targets, 87,000 rows"),
    ("mlp_ens3_cur_n87000", "3-MLP ensemble, plain-current targets, 87,000 rows"),
    ("mlp_ens3_lin_n87000", "3-MLP ensemble on a linear fit's residual, 87,000 rows"),
]


def main():
    rows = []
    ch = pd.read_csv(ROOT / "data" / "partA_results" / "chosen.csv")
    chosen_kinds = {f"{m}_n{n}" for m, n in zip(ch.model, ch.n_train)}
    for kind, name in KINDS:
        if len(chosen_kinds) == 1 and kind in chosen_kinds:
            continue      # same models as the chosen row below
        out = ROOT / "runs" / "partB_diag" / kind / "gates_ml.csv"
        print(f"running gates with {kind} ...", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "b2_gates_ml.py"), kind], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        g = pd.read_csv(out, dtype={"inputs": str})
        s = metrics.summary(g.err)
        per_case_vdd = g.groupby(["gate", "inputs"]).err_vddsum.median()
        rows.append(dict(model=name, median=s["median"], p95=s["p95"], mean=s["mean"], mx=g.err.max(),
                         vdd_median=np.median(g.err_vddsum), vdd_case_lo=per_case_vdd.min(),
                         vdd_case_hi=per_case_vdd.max(),
                         worst=g.groupby(["gate", "inputs"]).err.median().idxmax()))
    chosen = pd.read_csv(ROOT / "data" / "partB" / "gates_ml.csv", dtype={"inputs": str})
    s = metrics.summary(chosen.err)
    pcv = chosen.groupby(["gate", "inputs"]).err_vddsum.median()
    names = sorted({f"{label(m)}, {n:,} rows" for m, n in zip(ch.model, ch.n_train)})
    chosen_name = (f"averages of 3-MLP ensemble variants, {ch.n_train.iloc[0]:,} rows (per case: `chosen.csv`)"
                   if ch.model.str.startswith("blend").all() and ch.n_train.nunique() == 1 else "; ".join(names))
    rows.append(dict(model=f"**chosen: {chosen_name}**", median=s["median"], p95=s["p95"], mx=chosen.err.max(),
                     mean=s["mean"], vdd_median=np.median(chosen.err_vddsum), vdd_case_lo=pcv.min(),
                     vdd_case_hi=pcv.max(), worst=chosen.groupby(["gate", "inputs"]).err.median().idxmax()))

    lines = ["# Part B diagnostic: gates with other model candidates", "",
             "Not used for model selection (that is done on the Part A validation set). Each row uses the same "
             "3,600 gate simulations and the best-cut sum; the vdd-node columns show the same models with the "
             "currents summed at the vdd node instead (`ml/gate_eval.py`).", "",
             "| transistor models | median | p95 | mean | max | vdd-node sum: median | vdd-node sum: per-case medians | worst case (best cut) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['median']:.2f} % | {r['p95']:.2f} % | {r['mean']:.2f} % | {r['mx']:.1f} % | "
                     f"{r['vdd_median']:.1f} % | {r['vdd_case_lo']:.1f}–{r['vdd_case_hi']:.0f} % | "
                     f"{r['worst'][0]} {r['worst'][1]} |")
    lines.append("")
    (ROOT / "reports" / "partB" / "GATES_DIAGNOSTICS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
