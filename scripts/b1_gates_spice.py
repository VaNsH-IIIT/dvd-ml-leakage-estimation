"""Part B, step 3: SPICE reference for the five gates (doc sections 3 and 5).

Draws 200 corners (VDD, temperature, NMOS process, PMOS process, NMOS width Wn,
PMOS width Wp; doc section 4 distributions, fixed seed) -> data/partB/corners.csv; Part C reuses them.
Simulates every gate x input combination x corner (18 x 200 runs) and
records the total leakage (VDD current + current drawn by the inputs held at
VDD, see sim/gates.py), the VDD current alone, the input-pin currents, the
output voltage and the internal nodes -> data/partB/gates_spice.csv. These results are for evaluation only; they
are never used for training.

Also writes one example netlist per gate (corner 0, all inputs 0) to
netlists/partB/.
Run:  python scripts/b1_gates_spice.py
"""
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim import gates, spec  # noqa: E402
from sim.runner import parallel_map  # noqa: E402

OUT = ROOT / "data" / "partB"
NETLISTS = ROOT / "netlists" / "partB"
N_CORNERS = 200
CORNER_SEED = 2200


def draw_corners(n=N_CORNERS, seed=CORNER_SEED):
    rng = np.random.default_rng(seed)
    c = pd.DataFrame({"corner": np.arange(n),
                      "vdd": rng.uniform(*spec.VDD_RANGE, n),
                      "temp": rng.uniform(*spec.TEMP_RANGE, n)})
    cards = gates.cards()
    for dev in ("nmos", "pmos"):
        nom = cards[dev].params
        for p in spec.PROCESS_PARAMS:
            c[f"{dev[0]}_{p}"] = nom[p] * rng.uniform(1 - spec.PROCESS_SPREAD, 1 + spec.PROCESS_SPREAD, n)
    # transistor widths, drawn after the process so the earlier draws are unchanged
    c["wn"] = rng.uniform(*spec.W_RANGE, n)
    c["wp"] = rng.uniform(*spec.W_RANGE, n)
    return c


def cases():
    for gate, (inputs, _, _) in gates.GATES.items():
        for bits in itertools.product((0, 1), repeat=len(inputs)):
            yield gate, bits


def netlist(gate, bits, corner):
    inputs, internal, fets = gates.GATES[gate]
    title = f"{gate} inputs {''.join(map(str, bits))} corner {int(corner['corner'])}"
    return gates.build(title, fets, inputs, bits, ("out", *internal), corner, gates.cards())


def _job(args):
    gate, bits, corner = args
    res, err = gates.run(netlist(gate, bits, corner))
    if res is None:
        raise RuntimeError(f"{gate} {bits} corner {corner['corner']}: {err}")
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    NETLISTS.mkdir(parents=True, exist_ok=True)
    corners = draw_corners()
    corners.to_csv(OUT / "corners.csv", index=False, float_format="%.10g")
    recs = corners.to_dict("records")

    t0 = time.time()
    jobs = [(g, b, c) for g, b in cases() for c in recs]
    results = parallel_map(_job, jobs, chunksize=8)
    rows = []
    for (g, b, c), r in zip(jobs, results):
        pins = [r[f"pin_{n}"] for n in gates.GATES[g][0]]
        rows.append({"gate": g, "inputs": "".join(map(str, b)), "corner": int(c["corner"]),
                     "out_logic": gates.logic(g, b), "leak": gates.total_leakage(r["leak"], pins, b),
                     "leak_vdd": r["leak"],
                     **{k: v for k, v in r.items() if k.startswith("pin_")},
                     **{f"v_{k}": v for k, v in r.items() if k != "leak" and not k.startswith("pin_")}})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gates_spice.csv", index=False, float_format="%.10g")
    print(f"{len(df)} gate simulations in {time.time() - t0:.0f} s")

    for g, (inputs, _, _) in gates.GATES.items():
        (NETLISTS / f"{g}.cir").write_text(netlist(g, (0,) * len(inputs), recs[0]))
    print(df.groupby(["gate", "inputs"]).leak.median().map("{:.3e}".format).to_string())


if __name__ == "__main__":
    main()
