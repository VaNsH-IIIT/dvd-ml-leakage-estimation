"""Part C, steps 7-8: hierarchical ML leakage of the parity checker vs SPICE
(deliverable 8).

For each input vector, the logic value at the inputs of every NAND2 is found
(sim/parity.py) and the NAND2 gate model of Part B is called for that input
pattern at the same corner; the 28 gate leakages are summed, as the doc
prescribes. The gate model is the Part B ML prediction (data/partB/gates_ml.csv:
transistor models fed with the SPICE-recorded node voltages of a stand-alone
NAND2).

Leakage is the total current drawn from VDD-level sources: VDD supply current
plus the current drawn by the inputs held at VDD (sim/gates.py). With that
definition the sum of gate leakages is the circuit leakage: the current an
internal input draws is drawn from its driver's VDD, and it is counted once,
as part of the receiving gate's leakage.

Diagnostics: the same sum with the stand-alone SPICE NAND2 leakages
("gate-SPICE sum") isolates the composition error (every gate input assumed
at exactly 0 or VDD, gates simulated alone) from the ML error; and the same
sum with the VDD supply current alone shows what that narrower definition
would miss.

Writes data/partC/parity_ml.csv, reports/partC/PARITY.md and the parity plot.
Run:  python scripts/c2_parity_ml.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis import plotstyle  # noqa: E402
from ml import metrics  # noqa: E402
from sim import parity  # noqa: E402

DATA = ROOT / "data" / "partC"
REPORT = ROOT / "reports" / "partC"
PATTERNS = ("00", "01", "10", "11")


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    spice = pd.read_csv(DATA / "parity_spice.csv", dtype={"inputs": str})
    gates = pd.read_csv(ROOT / "data" / "partB" / "gates_ml.csv", dtype={"inputs": str})
    nand = gates[gates.gate == "NAND2"].set_index(["inputs", "corner"])

    # per input vector: how many of the 28 NAND2s see each input pattern
    counts = {v: [parity.nand_inputs(tuple(int(ch) for ch in v)).count(p) for p in PATTERNS]
              for v in spice.inputs.unique()}
    cnt = np.array([counts[v] for v in spice.inputs])

    def gate_sum(col):
        per = np.column_stack([nand.loc[list(zip([p] * len(spice), spice.corner)), col].to_numpy()
                               for p in PATTERNS])
        return (cnt * per).sum(axis=1)

    spice["ml_leak"] = gate_sum("ml_leak")
    spice["gate_spice_sum"] = gate_sum("leak")
    spice["gate_spice_sum_vdd"] = gate_sum("leak_vdd")
    spice["err"] = metrics.rel_err(spice.ml_leak, spice.leak)
    spice["sum_err"] = metrics.rel_err(spice.gate_spice_sum, spice.leak)
    spice["ml_vs_sum_err"] = metrics.rel_err(spice.ml_leak, spice.gate_spice_sum)
    spice["sum_err_vdd"] = metrics.rel_err(spice.gate_spice_sum_vdd, spice.leak_vdd)
    spice.to_csv(DATA / "parity_ml.csv", index=False, float_format="%.10g")
    write_report(spice)


def row(name, g):
    s = metrics.summary(g.err)
    return (f"| {name} | {len(g):,} | {g.leak.median():.3e} | {g.ml_leak.median():.3e} | "
            f"{100 * np.median(g.ml_leak / g.leak - 1):+.2f} % | {s['median']:.2f} % | {s['p95']:.2f} % | "
            f"{s['mean']:.2f} % | {np.median(g.sum_err):.2f} % |")


def write_report(df):
    lines = ["# Part C: 8-bit parity checker, hierarchical ML vs SPICE", "",
             "7 XOR2 in a tree, each XOR2 = 4 NAND2 (28 NAND2, 112 transistors). 256 input vectors × the 200 "
             "Part B corners = 51,200 full-netlist SPICE runs. ML = sum of the 28 NAND2 gate-model leakages "
             "(each gate model = the Part B transistor models). Leakage = VDD supply current + current drawn by "
             "the inputs held at VDD (`sim/gates.py`).", "",
             "| set | runs | SPICE median (A) | ML median (A) | bias | median error | p95 error | mean error "
             "| same sum with SPICE gates: median error |",
             "|---|---|---|---|---|---|---|---|---|", row("all", df)]
    for k, g in df.groupby(df.inputs.str.count("1")):
        lines.append(row(f"{k} input{'' if k == 1 else 's'} high", g))
    s = metrics.summary(df.sum_err)
    m = metrics.summary(df.ml_vs_sum_err)
    v = metrics.summary(df.sum_err_vdd)
    lines += ["", "## Where the error comes from", "",
              f"- **Composition** (sum of stand-alone SPICE NAND2 leakages vs the full netlist): median "
              f"{s['median']:.2f} %, p95 {s['p95']:.2f} %, max {df.sum_err.max():.2f} %. It comes from internal "
              "inputs not sitting exactly at 0 or VDD and from the next gate's input current slightly shifting "
              "each node.",
              f"- **ML vs that SPICE gate sum:** median {m['median']:.2f} %, p95 {m['p95']:.2f} %.",
              f"- **With the VDD supply current alone** as the leakage, the same sum of SPICE gate values misses the "
              f"circuit by {v['median']:.1f} % median (p95 {v['p95']:.1f} %): inside the circuit, the input current of "
              "an internal net at logic 1 is supplied by its driver's VDD, which a stand-alone gate's VDD current "
              "does not include.", "",
              "![Parity-checker parity plot](parity_parity.png)", ""]
    (REPORT / "PARITY.md").write_text("\n".join(lines) + "\n")
    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.loglog(df.leak, df.ml_leak, ".", ms=2, alpha=0.25, color="#2a78d6")
    lo = df[["leak", "ml_leak"]].min().min() / 1.2
    hi = df[["leak", "ml_leak"]].max().max() * 1.2
    ax.plot([lo, hi], [lo, hi], color=plotstyle.MUTED, lw=1)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="SPICE leakage (A)", ylabel="ML leakage (A)",
           title="8-bit parity checker: ML vs SPICE (51,200 runs)")
    fig.savefig(REPORT / "parity_parity.png")
    plt.close(fig)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
