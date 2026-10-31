"""Part C, step 6: SPICE reference for the 8-bit parity checker.

The full 112-transistor netlist (sim/parity.py) at the 200 Part B corners
(data/partB/corners.csv) and all 256 input vectors: 51,200 runs. Records the
VDD current -i(vdd), the input currents and the output voltage, and the
total leakage (VDD current + current drawn by the inputs held at VDD)
-> data/partC/parity_spice.csv.
Writes the example netlist (corner 0, inputs 00000000) to netlists/partC/.
Run:  python scripts/c1_parity_spice.py
"""
import itertools
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim import gates, parity  # noqa: E402
from sim.runner import parallel_map  # noqa: E402

OUT = ROOT / "data" / "partC"
NETLISTS = ROOT / "netlists" / "partC"
FETS = parity.fets()
VECTORS = list(itertools.product((0, 1), repeat=8))   # bit k drives input i_k


def netlist(bits, corner):
    title = f"8-bit parity checker, inputs {''.join(map(str, bits))}, corner {int(corner['corner'])}"
    return gates.build(title, FETS, parity.INPUTS, bits, ("p",), corner, gates.cards())


def _job(args):
    bits, corner = args
    res, err = gates.run(netlist(bits, corner))
    if res is None:
        raise RuntimeError(f"{bits} corner {corner['corner']}: {err}")
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    NETLISTS.mkdir(parents=True, exist_ok=True)
    corners = pd.read_csv(ROOT / "data" / "partB" / "corners.csv").to_dict("records")
    (NETLISTS / "parity8.cir").write_text(netlist(VECTORS[0], corners[0]))

    t0 = time.time()
    jobs = [(b, c) for c in corners for b in VECTORS]
    res = parallel_map(_job, jobs, chunksize=16)
    df = pd.DataFrame([{"inputs": "".join(map(str, b)), "corner": int(c["corner"]),
                        "parity": sum(b) % 2, "leak_vdd": r["leak"],
                        **{k: v for k, v in r.items() if k.startswith("pin_")}, "v_p": r["p"]}
                       for (b, c), r in zip(jobs, res)])
    # total leakage = current from VDD + current drawn by the inputs held at VDD (see sim/gates.py)
    df["leak"] = [gates.total_leakage(r.leak_vdd, [getattr(r, f"pin_{n}") for n in parity.INPUTS],
                                      [int(ch) for ch in r.inputs]) for r in df.itertuples()]
    df.to_csv(OUT / "parity_spice.csv", index=False, float_format="%.10g")
    print(f"{len(df)} parity-checker simulations in {time.time() - t0:.0f} s; "
          f"median leakage {df.leak.median():.3e} A")


if __name__ == "__main__":
    main()
