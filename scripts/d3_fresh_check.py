"""Fresh-data check: score the final pipeline on data that influenced nothing.

New seeds, never used elsewhere in the project:
  1. transistors: 2,000 new samples per case (seeds 779000-779003);
  2. gates: 100 new corners (seed 999), all 18 gate cases, SPICE vs ML;
  3. parity checker: those 100 corners x 256 vectors, SPICE vs ML
     (sum of NAND2 gate-model leakages).
An earlier set (seeds 777000-777003 and 888) pointed to the errors at hot,
wide, low-vth0 corners that the per-width and linear-baseline model variants
address, so it is no longer independent; this set was drawn afterwards.
Writes reports/FRESH_CHECK.md.
Run:  python scripts/d3_fresh_check.py   (about 5 minutes)
"""
import itertools
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import a1_generate_data as a1  # noqa: E402
import b1_gates_spice as b1  # noqa: E402
import b2_gates_ml as b2  # noqa: E402
from ml import metrics  # noqa: E402
from ml.gate_eval import leakage, predict_currents  # noqa: E402
from sim import gates, parity, spec  # noqa: E402
from sim.modelcard import load_cards  # noqa: E402
from sim.runner import parallel_map  # noqa: E402

FETS = parity.fets()


def pjob(a):
    bits, c = a
    r, err = gates.run(gates.build("parity", FETS, parity.INPUTS, bits, ("p",), c, gates.cards()))
    return gates.total_leakage(r["leak"], [r[f"pin_{n}"] for n in parity.INPUTS], bits)


LINES = ["# Fresh-data check", "",
         "The final pipeline scored on data generated with seeds used nowhere else in the project "
         "(`scripts/d3_fresh_check.py`). An earlier fresh set (seeds 777000-777003, 888) showed the errors at hot, "
         "wide, low-vth0 corners that led to the per-width and linear-baseline model variants, so it is no longer "
         "independent; this set (seeds 779000-779003, 999) was drawn after that.", ""]


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def transistors(cards, models):
    say("## 1. Transistors: 2,000 new samples per case")
    say()
    say("| case | median | p95 | median without ib | p95 without ib |")
    say("|---|---|---|---|---|")
    for k, case in enumerate(spec.CASES):
        df = a1.draw_inputs(case, 2000, cards[case[:4].lower()].params, 779000 + k)
        res = a1.simulate(a1.to_samples(case, df))
        for n in ("id", "ig", "is", "ib"):
            df[n] = [c[n] for c, _ in res]
        e = metrics.rel_err(models[case].predict(df), df[["id", "ig", "is", "ib"]].to_numpy())
        say(f"| {case} | {np.median(e):.2f} % | {np.percentile(e, 95):.2f} % | "
            f"{np.median(e[:, :3]):.2f} % | {np.percentile(e[:, :3], 95):.2f} % |")


def gate_results(models):
    """SPICE and ML (best-cut sum, plus input-pin currents) for every gate case at 100 new corners."""
    say()
    say("## 2. Gates: 100 new corners x 18 cases")
    corners = b1.draw_corners(100, seed=999)
    recs = corners.to_dict("records")
    jobs = [(g, b, c) for g, b in b1.cases() for c in recs]
    out = parallel_map(b1._job, jobs, chunksize=8)
    rows = pd.DataFrame([{"gate": g, "inputs": "".join(map(str, b)), "corner": int(c["corner"]),
                          "leak": gates.total_leakage(r["leak"], [r[f"pin_{n}"] for n in gates.GATES[g][0]], b),
                          **{f"v_{k}": v for k, v in r.items() if k != "leak" and not k.startswith("pin_")}}
                         for (g, b, c), r in zip(jobs, out)])
    cidx = corners.set_index("corner")
    parts = []
    for gate, g in rows.groupby("gate", sort=False):
        g = g.reset_index(drop=True)
        c = cidx.loc[g.corner].reset_index()
        nodes = b2.node_table(g, c, gate)
        fets = gates.GATES[gate][2]
        cur, on = predict_currents(fets, nodes, c, models)
        g["ml"] = leakage(fets, cur, on, "best-cut", nodes)
        for k, pin in enumerate(gates.GATES[gate][0]):   # + current drawn by the inputs held at VDD
            g["ml"] += g.inputs.str[k].astype(int).to_numpy() * -sum(cur[f.name][:, 1] for f in fets if f.g == pin)
        parts.append(g)
    G = pd.concat(parts, ignore_index=True)
    G["err"] = metrics.rel_err(G.ml, G.leak)
    s = metrics.summary(G.err)
    worst = G.groupby(["gate", "inputs"]).err.median().sort_values(ascending=False).head(3)
    say()
    say(f"All 1,800 simulations: median {s['median']:.2f} %, p95 {s['p95']:.2f} %, mean {s['mean']:.2f} %, "
        f"{s['within_1pct']:.0f} % within 1 %.")
    say("Worst cases (median): " + ", ".join(f"{a} {b}: {v:.2f} %" for (a, b), v in worst.items()) + ".")
    return G, recs


def parity_results(G, recs):
    """Full parity-checker SPICE vs the sum of NAND2 gate-model leakages, 100 corners x 256 vectors."""
    say()
    say("## 3. Parity checker: the same 100 corners x 256 vectors")
    vectors = list(itertools.product((0, 1), repeat=8))
    pairs = [(b, c) for c in recs for b in vectors]
    spice = np.array(parallel_map(pjob, pairs, chunksize=16))
    nand = G[G.gate == "NAND2"].set_index(["inputs", "corner"])
    est = []
    for bits, c in pairs:
        est.append(sum(nand.loc[(pat, int(c["corner"]))].ml for pat in parity.nand_inputs(bits)))
    s = metrics.summary(metrics.rel_err(np.array(est), spice))
    say()
    say(f"All 25,600 runs (sum of NAND2 gate-model leakages): median {s['median']:.2f} %, p95 {s['p95']:.2f} %, "
        f"mean {s['mean']:.2f} %.")


def main():
    cards = load_cards(spec.MODEL_FILE)
    models = {c: joblib.load(ROOT / "trained_models" / f"{c}.joblib") for c in spec.CASES}
    transistors(cards, models)
    G, recs = gate_results(models)
    parity_results(G, recs)
    (ROOT / "reports" / "FRESH_CHECK.md").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
