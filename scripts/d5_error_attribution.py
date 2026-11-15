"""Which transistors carry the gate-leakage error (report section "Error analysis").

The gate leakage is a linear combination of the transistor terminal currents
(the cut through the OFF devices plus the gate currents of the inputs held at
VDD; ml/gate_eval.py). So the ML error of every gate run splits exactly into
one part per transistor: that transistor's ML currents minus its currents
when simulated alone in SPICE at the same terminal voltages ("device-SPICE",
as in b2), pushed through the same combination. The parts add up to
ML - device-SPICE, which equals ML - full-gate SPICE to within 0.0005 %.

Each part is expressed in % of the gate's SPICE leakage and grouped by the
transistor's state (ON / OFF), type (NMOS / PMOS), voltage across it, and
current (channel id + is, gate ig, body ib).

Writes data/partB/error_attribution.csv and reports/partB/ERROR_ATTRIBUTION.md.
Run:  python scripts/d5_error_attribution.py   (after b1 and a4; ~2 min)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

import b2_gates_ml as b2  # noqa: E402
from ml.gate_eval import leakage, predict_currents  # noqa: E402
from sim import gates  # noqa: E402

DATA = ROOT.parent / "data" / "partB"
REPORT = ROOT.parent / "reports" / "partB"
CURRENTS = {"channel (id + is)": (0, 2), "gate (ig)": (1,), "body (ib)": (3,)}


def contributions(fets, inputs, bits, delta, on, nodes):
    """Leakage change caused by one transistor's current error `delta`
    (n x 4, columns d, g, s, b), per current group."""
    n = len(nodes)
    zero = {f.name: np.zeros((n, 4)) for f in fets}
    out = {}
    for group, cols in CURRENTS.items():
        d = np.zeros_like(delta[1])
        d[:, cols] = delta[1][:, cols]
        cur = zero | {delta[0].name: d}
        pins = sum(bits[p] * -d[:, 1] for p in inputs if delta[0].g == p)
        out[group] = leakage(fets, cur, on, "best-cut", nodes) + pins
    return out


def main():
    spice = pd.read_csv(DATA / "gates_spice.csv", dtype={"inputs": str})
    corners = pd.read_csv(DATA / "corners.csv").set_index("corner")
    models = b2.load_models()
    parts = []
    for gate, rows in spice.groupby("gate", sort=False):
        rows = rows.reset_index(drop=True)
        c = corners.loc[rows.corner].reset_index()
        nodes = b2.node_table(rows, c, gate)
        inputs, _, fets = gates.GATES[gate]
        cur, on = predict_currents(fets, nodes, c, models)
        dev = b2.device_spice_currents(fets, nodes, c)
        bits = {p: rows.inputs.str[k].astype(int).to_numpy() for k, p in enumerate(inputs)}
        vdd = c["vdd"].to_numpy()
        for j, f in enumerate(fets):
            parts_f = contributions(fets, inputs, bits, (f, cur[f.name] - dev[f.name]), on, nodes)
            vds = np.abs(nodes[f.d].to_numpy() - nodes[f.s].to_numpy())
            base = pd.DataFrame({"gate": gate, "inputs": rows.inputs, "corner": rows.corner, "fet": f.name,
                                 "devtype": f.devtype.upper(), "state": np.where(on[:, j], "ON", "OFF"),
                                 "full_vds": vds > 0.5 * vdd, "leak": rows.leak})
            for group, d in parts_f.items():
                parts.append(base.assign(current=group, delta=d, share_pct=d / rows.leak * 100))
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(DATA / "error_attribution.csv", index=False, float_format="%.6g")
    write_report(df)


def role(r):
    if r.state == "ON":
        return f"{r.devtype} ON"
    return f"{r.devtype} OFF, {'full VDD across' if r.full_vds else 'small Vds (stacked)'}"


def write_report(df):
    df = df.assign(role=df.apply(role, axis=1), a=df.share_pct.abs())
    run_err = df.groupby(["gate", "inputs", "corner"]).share_pct.sum().abs()
    tot = df.a.sum()

    def share(by):
        return (df.groupby(by).a.sum() / tot * 100).sort_values(ascending=False)

    by_role, by_cur = share("role"), share("current")
    gate_err = df.groupby(["gate", "inputs", "corner", "role"]).share_pct.sum().abs().reset_index()
    top = gate_err.loc[gate_err.groupby(["gate", "inputs", "corner"]).share_pct.idxmax()]
    lines = ["# Part B: which transistors carry the gate-leakage error", "",
             "Each gate run's ML error is split exactly into one part per transistor: its ML currents "
             "minus its single-transistor SPICE currents at the same terminal voltages, pushed through the "
             "same cut-based combination as the leakage (`scripts/d5_error_attribution.py`). Parts are in % of "
             "the gate's SPICE leakage; shares are of the summed |part| over all 3,600 runs.",
             f"Check: the parts add up to the gate error (median |error| {run_err.median():.2f} %, "
             f"mean {run_err.mean():.2f} %).", "",
             "## By transistor", "", "| transistor | share of the error |", "|---|---|"]
    lines += [f"| {k} | {v:.0f} % |" for k, v in by_role.items()]
    lines += ["", "## By current", "", "| current | share of the error |", "|---|---|"]
    lines += [f"| {k} | {v:.0f} % |" for k, v in by_cur.items()]
    lines += ["", "## Transistor with the largest error, per gate case (share of the 200 corners)", "",
              "| gate | inputs | most often largest | how often |", "|---|---|---|---|"]
    for (g, i), t in top.groupby(["gate", "inputs"], sort=False):
        vc = t.role.value_counts(normalize=True)
        lines.append(f"| {g} | {i} | {vc.index[0]} | {vc.iloc[0] * 100:.0f} % |")
    (REPORT / "ERROR_ATTRIBUTION.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:22]))


if __name__ == "__main__":
    main()
