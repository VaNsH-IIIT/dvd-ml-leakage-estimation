"""Are the SPICE references unique operating points? (report section "Error analysis")

A DC solve can converge to a wrong operating point. Every gate run and part of
the parity-checker runs (the 300 with the largest ML error + 300 random) are
solved again with every non-rail node first pushed to 0 V and then to VDD
(.nodeset), and the leakage is compared with the committed reference. Any
other solution would show up as a difference.

Writes data/partB/op_check.csv and reports/SPICE_CHECK.md.
Run:  python scripts/d4_op_check.py   (after b1 and c2; ~5 min on 16 cores)
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

import b1_gates_spice as b1  # noqa: E402
import c1_parity_spice as c1  # noqa: E402
from sim import gates, parity  # noqa: E402

DATA = ROOT.parent / "data"
REPORT = ROOT.parent / "reports"


def free_nodes(fets, inputs):
    return sorted({n for f in fets for n in (f.d, f.s)} - {"vdd", "0", *inputs})


def with_nodeset(text, nodes, v):
    return text.replace(".control", ".nodeset " + " ".join(f"v({n})={v:.6g}" for n in nodes) + "\n.control", 1)


def resolve(base, fets, inputs, bits, vdd):
    """Leakage after starting from all free nodes at 0 V and at VDD."""
    out = []
    for v in (0.0, vdd):
        res, _ = gates.run(with_nodeset(base, free_nodes(fets, inputs), v))
        out.append(np.nan if res is None else
                   gates.total_leakage(res["leak"], [res[f"pin_{n}"] for n in inputs], bits))
    return out


def gate_job(a):
    gate, bits, corner, ref = a
    inputs, _, fets = gates.GATES[gate]
    return ("gates", gate, "".join(map(str, bits)), int(corner["corner"]), ref,
            *resolve(b1.netlist(gate, bits, corner), fets, inputs, bits, corner["vdd"]))


def parity_job(a):
    bits, corner, ref = a
    return ("parity checker", "PARITY8", "".join(map(str, bits)), int(corner["corner"]), ref,
            *resolve(c1.netlist(bits, corner), c1.FETS, parity.INPUTS, bits, corner["vdd"]))


def main():
    corners = pd.read_csv(DATA / "partB" / "corners.csv").set_index("corner")

    def corner(k):
        return corners.loc[k].to_dict() | {"corner": k}

    g = pd.read_csv(DATA / "partB" / "gates_ml.csv", dtype={"inputs": str})
    p = pd.read_csv(DATA / "partC" / "parity_ml.csv", dtype={"inputs": str})
    pick = pd.concat([p.nlargest(300, "err"), p.sample(300, random_state=0)]).drop_duplicates(["inputs", "corner"])
    with Pool(min(12, os.cpu_count())) as pool:
        rows = pool.map(gate_job, [(r.gate, tuple(map(int, r.inputs)), corner(r.corner), r.leak)
                                   for r in g.itertuples()], chunksize=20)
        rows += pool.map(parity_job, [(tuple(map(int, r.inputs)), corner(r.corner), r.leak)
                                      for r in pick.itertuples()], chunksize=5)
    df = pd.DataFrame(rows, columns=["set", "gate", "inputs", "corner", "leak", "leak_from_0", "leak_from_vdd"])
    df["max_diff_pct"] = np.maximum(abs(df.leak_from_0 / df.leak - 1), abs(df.leak_from_vdd / df.leak - 1)) * 100
    df.to_csv(DATA / "partB" / "op_check.csv", index=False, float_format="%.10g")
    lines = ["# SPICE reference check: unique operating points", "",
             "Each run solved again from every non-rail node at 0 V and at VDD (`scripts/d4_op_check.py`).", "",
             "| set | runs | restarts that failed | largest difference from the reference | runs differing by > 0.01 % |",
             "|---|---|---|---|---|"]
    for s, d in df.groupby("set", sort=False):
        failed = int(d[["leak_from_0", "leak_from_vdd"]].isna().sum().sum())
        lines.append(f"| {s} | {len(d):,} | {failed} | {d.max_diff_pct.max():.1e} % | {(d.max_diff_pct > 0.01).sum()} |")
    lines += ["", "Parity checker: the 300 runs with the largest ML error plus 300 random runs (overlaps removed)."]
    (REPORT / "SPICE_CHECK.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
