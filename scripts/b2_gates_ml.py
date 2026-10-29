"""Part B, steps 4-5: ML gate leakage vs SPICE (deliverables 6 and 7).

For every gate x input x corner of data/partB/gates_spice.csv, the ML
leakage is built from the four chosen transistor models (trained_models/),
fed with the SPICE-recorded output and internal node voltages
(ml/gate_eval.py).

As a diagnostic, the same sum is also done with every transistor simulated
alone in its single-FET SPICE netlist at the same terminal voltages
("device-SPICE"). Its agreement with the full-gate SPICE shows how much of
the error comes from the method itself rather than from the ML models.

Writes data/partB/gates_ml.csv, reports/partB/GATES.md and the parity plot.
Run:  python scripts/b2_gates_ml.py            (chosen models, trained_models/)
      python scripts/b2_gates_ml.py <kind>     (diagnostic: runs/partA/<CASE>_<kind>.joblib,
                                                output to runs/partB_diag/<kind>/)
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis import plotstyle  # noqa: E402
from ml import metrics  # noqa: E402
from ml.gate_eval import TERMINALS, leakage, predict_currents, solve_internal_nodes  # noqa: E402
from sim import gates, spec  # noqa: E402
from sim.runner import Sample, simulate  # noqa: E402

DATA = ROOT / "data" / "partB"
REPORT = ROOT / "reports" / "partB"
MODELS = ROOT / "trained_models"
OUT_DATA, OUT_REPORT = DATA, REPORT   # redirected for diagnostic runs (see main)


def load_models(kind=None):
    """The chosen models, or (diagnostics) one model type from runs/partA/."""
    if kind:
        return {case: joblib.load(ROOT / "runs" / "partA" / f"{case}_{kind}.joblib") for case in spec.CASES}
    return {case: joblib.load(MODELS / f"{case}.joblib") for case in spec.CASES}


def node_table(rows, corners, gate):
    """Node voltages per row: supply, ground, exact inputs, SPICE-recorded nodes."""
    inputs, internal, _ = gates.GATES[gate]
    vdd = corners["vdd"].to_numpy()
    nodes = pd.DataFrame({"vdd": vdd, "0": np.zeros(len(rows))})
    for k, name in enumerate(inputs):
        nodes[name] = vdd * rows["inputs"].str[k].astype(int).to_numpy()
    for name in ("out", *internal):
        nodes[name] = rows[f"v_{name}"].to_numpy()
    return nodes


def device_spice_currents(fets, nodes, corners):
    """Every FET simulated alone in its single-FET SPICE netlist at the same
    terminal voltages: {fet name: n x 4 currents}, like the ML prediction."""
    samples = []
    for f in fets:
        pre = f.devtype[0]
        for i in range(len(nodes)):
            c = corners.iloc[i]
            samples.append(Sample(f.devtype, nodes[f.d].iat[i], nodes[f.g].iat[i], nodes[f.s].iat[i],
                                  nodes[f.b].iat[i], c["temp"], gates.width(f, c), spec.L,
                                  {p: c[f"{pre}_{p}"] for p in spec.PROCESS_PARAMS}))
    res = simulate(samples)
    if any(c is None for c, _ in res):
        raise RuntimeError(next(e for c, e in res if c is None))
    flat = np.array([[c[f"i{t}"] for t in TERMINALS] for c, _ in res])
    n = len(nodes)
    return {f.name: flat[k * n:(k + 1) * n] for k, f in enumerate(fets)}


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    spice = pd.read_csv(DATA / "gates_spice.csv", dtype={"inputs": str})
    corners = pd.read_csv(DATA / "corners.csv").set_index("corner")
    kind = sys.argv[1] if len(sys.argv) > 1 else None
    models = load_models(kind)
    if kind:   # diagnostic run with another model type: never overwrite the final results
        global OUT_DATA, OUT_REPORT
        OUT_DATA = OUT_REPORT = ROOT / "runs" / "partB_diag" / kind
        OUT_DATA.mkdir(parents=True, exist_ok=True)

    parts = []
    for gate, rows in spice.groupby("gate", sort=False):
        rows = rows.reset_index(drop=True)
        c = corners.loc[rows.corner].reset_index()
        nodes = node_table(rows, c, gate)
        fets = gates.GATES[gate][2]
        cur, on = predict_currents(fets, nodes, c, models)
        dev = device_spice_currents(fets, nodes, c)
        inputs = gates.GATES[gate][0]
        for pin in inputs:   # current drawn from each input = minus the gate currents on that pin
            rows[f"ml_pin_{pin}"] = -sum(cur[f.name][:, 1] for f in fets if f.g == pin)
        bits = {pin: rows.inputs.str[k].astype(int).to_numpy() for k, pin in enumerate(inputs)}

        def high_pins(currents):
            """Current drawn by the inputs held at VDD (part of the total leakage)."""
            return sum(bits[pin] * -sum(currents[f.name][:, 1] for f in fets if f.g == pin) for pin in inputs)

        ml_pins, dev_pins = high_pins(cur), high_pins(dev)
        rows["ml_leak_vdd"] = leakage(fets, cur, on, "best-cut", nodes)
        rows["ml_leak"] = rows.ml_leak_vdd + ml_pins
        rows["ml_leak_supernode"] = leakage(fets, cur, on, "supernode") + ml_pins
        rows["ml_leak_vddsum"] = leakage(fets, cur, on, "vdd") + ml_pins
        rows["dev_spice_leak"] = leakage(fets, dev, on, "best-cut", nodes) + dev_pins
        rows["dev_spice_leak_vddsum"] = leakage(fets, dev, on, "vdd") + dev_pins
        internal = list(gates.GATES[gate][1])
        if internal and not kind:   # ML's own internal-node voltages (KCL with the ML models), for the record
            sol = solve_internal_nodes(fets, nodes, c, models, internal)
            for n in internal:
                rows[f"ml_v_{n}"] = sol[n].to_numpy()
        parts.append(rows)
    df = pd.concat(parts, ignore_index=True)
    df["err"] = metrics.rel_err(df.ml_leak, df.leak)
    df["err_vdd_only"] = metrics.rel_err(df.ml_leak_vdd, df.leak_vdd)
    df["err_vddsum"] = metrics.rel_err(df.ml_leak_vddsum, df.leak)
    df["err_supernode"] = metrics.rel_err(df.ml_leak_supernode, df.leak)
    df["dev_err"] = metrics.rel_err(df.dev_spice_leak, df.leak)
    df["dev_err_vddsum"] = metrics.rel_err(df.dev_spice_leak_vddsum, df.leak)
    df.to_csv(OUT_DATA / "gates_ml.csv", index=False, float_format="%.10g")
    write_report(df)


def table(df, by):
    out = []
    for key, g in df.groupby(by, sort=False):
        s = metrics.summary(g.err)
        out.append(dict(zip(by, key if isinstance(key, tuple) else (key,)),
                        spice_median=g.leak.median(), ml_median=g.ml_leak.median(),
                        bias=100 * np.median(g.ml_leak / g.leak - 1),
                        median_err=s["median"], p95_err=s["p95"], mean_err=s["mean"],
                        vddsum_median_err=np.median(g.err_vddsum),
                        supernode_median_err=np.median(g.err_supernode),
                        dev_median_err=np.median(g.dev_err),
                        node_dv=" ".join(f"{n}: {1e3 * np.median(np.abs(g[f'ml_v_{n}'] - g[f'v_{n}'])):.3g} mV"
                                         for n in ("x", "y", "n") if f"ml_v_{n}" in g and g[f"ml_v_{n}"].notna().any())))
    return pd.DataFrame(out)


def node_summary(df):
    """One sentence on how well ML's own KCL solution of the internal nodes matches SPICE."""
    parts = [pd.DataFrame({"case": df.gate + " " + df.inputs, "node": n,
                           "dv": 1e3 * (df[f"ml_v_{n}"] - df[f"v_{n}"]).abs()})
             for n in ("x", "y", "n") if f"ml_v_{n}" in df]
    if not parts:     # diagnostic runs skip the node solver
        return "Not solved in this run."
    dv = pd.concat(parts).dropna()
    per = dv.groupby(["case", "node"]).dv
    worst = per.max().idxmax()
    return (f"Per gate case and node it agrees with SPICE to a median of {per.median().max():.2g} mV or better "
            f"(p95 at most {per.quantile(0.95).max():.2g} mV). It fails in a few corners where the balance hinges on "
            f"an ON device at a drain-source voltage of microvolts, the region the uniformly sampled training data "
            f"barely covers: worst {worst[0]}, node {worst[1]}, {per.max().max():.0f} mV.")


def write_report(df):
    t = table(df, ["gate", "inputs"])
    lines = ["# Part B: gate leakage, ML vs SPICE", "",
             "18 gate × input cases × 200 corners. Leakage = total current drawn from VDD-level sources: the VDD "
             "supply current plus the current drawn by the inputs held at VDD (`sim/gates.py`). "
             "Bias = median of (ML / SPICE − 1). Errors are per-simulation |ML − SPICE| / SPICE.", "",
             "| gate | inputs | SPICE median (A) | ML median (A) | bias | median error | p95 error | mean error | median error, vdd-node sum | median error, minimal supernode | device-SPICE median error | ML-solved internal node vs SPICE (median \\|ΔV\\|) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        lines.append(f"| {r.gate} | {r.inputs} | {r.spice_median:.3e} | {r.ml_median:.3e} | {r.bias:+.2f} % | "
                     f"{r.median_err:.2f} % | {r.p95_err:.2f} % | {r.mean_err:.2f} % | {r.vddsum_median_err:.2f} % | {r.supernode_median_err:.2f} % | {r.dev_median_err:.4f} % | {r.node_dv or '—'} |")
    s = metrics.summary(df.err)
    d = metrics.summary(df.dev_err)
    v = metrics.summary(df.err_vddsum)
    sn = metrics.summary(df.err_supernode)
    dv = metrics.summary(df.dev_err_vddsum)
    vo = metrics.summary(df.err_vdd_only)
    lines += ["", f"**All 3,600 gate simulations:** median error {s['median']:.2f} %, p95 {s['p95']:.2f} %, "
              f"mean {s['mean']:.2f} %, {s['within_1pct']:.0f} % of simulations within 1 %.", "",
              f"The VDD supply current alone (without the input currents) is predicted to median {vo['median']:.2f} %, "
              f"p95 {vo['p95']:.2f} %, mean {vo['mean']:.2f} %.", "",
              f"**Device-SPICE check** (same recipe, each transistor simulated alone instead of the ML model): "
              f"median {d['median']:.4f} %, p95 {d['p95']:.4f} %, max {df.dev_err.max():.4f} % "
              f"(vdd-node sum: median {dv['median']:.4f} %, max {df.dev_err_vddsum.max():.4f} %). "
              "So the recipe itself is exact; the remaining error is the transistor models'.", "",
              "**How the device currents are combined** (`ml/gate_eval.py`), all 3,600 simulations:", "",
              "| method | median | p95 | mean |", "|---|---|---|---|",
              f"| sum at the vdd node | {v['median']:.2f} % | {v['p95']:.2f} % | {v['mean']:.2f} % |",
              f"| minimal supernode (vdd + ON-connected nodes) | {sn['median']:.2f} % | {sn['p95']:.2f} % | {sn['mean']:.2f} % |",
              f"| **best cut (used)** | **{s['median']:.2f} %** | **{s['p95']:.2f} %** | **{s['mean']:.2f} %** |", "",
              "**Internal node voltages.** The ML leakage uses the SPICE-recorded node voltages (doc step 4). For the "
              "record, the last column also gives ML's own estimate: the voltage where the ML device currents into "
              f"the node balance (KCL, `ml/gate_eval.solve_internal_nodes`). {node_summary(df)}", "",
              "![Gate parity plot](gates_parity.png)", ""]
    (OUT_REPORT / "GATES.md").write_text("\n".join(lines) + "\n")
    t.to_csv(OUT_DATA / "gates_table.csv", index=False, float_format="%.6g")
    parity(df, OUT_REPORT / "gates_parity.png", "Gate leakage: ML vs SPICE (3,600 simulations)", "gate")
    print("\n".join(lines[:26]))


def parity(df, path, title, by):
    """One log-log panel per gate (all inputs and corners), plus the y = x line."""
    plotstyle.apply()
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd8"]
    groups = list(df.groupby(by, sort=False))
    fig, axes = plt.subplots(1, len(groups), figsize=(3.0 * len(groups), 3.1))
    lo, hi = df[["leak", "ml_leak"]].min().min() / 1.5, df[["leak", "ml_leak"]].max().max() * 1.5
    for k, (ax, (name, g)) in enumerate(zip(axes, groups)):
        ax.loglog(g.leak, g.ml_leak, ".", ms=2.5, alpha=0.6, color=colors[k % 5])
        ax.plot([lo, hi], [lo, hi], color=plotstyle.MUTED, lw=1)
        s = metrics.summary(g.err)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="SPICE leakage (A)",
               title=f"{name}\nmedian {s['median']:.2f} %, p95 {s['p95']:.1f} %")
    axes[0].set_ylabel("ML leakage (A)")
    fig.suptitle(title, fontsize=10, y=1.08)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
