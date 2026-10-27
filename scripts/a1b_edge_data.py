"""Optional training-only "edge" samples (experiment; not part of the doc's
datasets).

Gates bias their transistors at the edges of the (Vd, Vs) square: terminals
at a rail (0 or VDD) and ON devices at a drain-source voltage of microvolts
to millivolts. Uniform sampling puts few rows there. This script draws the
same doc ranges for VDD, T, W and process, but places Vd and Vs near the
edges: each terminal is, with probability 1/3 each, uniform on 0..VDD,
within a log-uniform distance (1e-6..0.1 V) of a rail, or within a
log-uniform distance of the other terminal.

Output: data/partA_edge/<CASE>_<k>.csv (doc columns). Validation and test
always stay on the doc's dataset.
Run:  python scripts/a1b_edge_data.py [--n 10000] [--blocks 1]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from a1_generate_data import COLUMNS, SEED, draw_inputs, to_samples  # noqa: E402
from sim import spec  # noqa: E402
from sim.bias import devtype, terminal_voltages  # noqa: E402
from sim.modelcard import load_cards  # noqa: E402
from sim.runner import CURRENT_NAMES, simulate  # noqa: E402

OUT = ROOT / "data" / "partA_edge"


def edge_voltages(rng, vdd):
    n = len(vdd)
    dist = 10 ** rng.uniform(-6, -1, (2, n))
    vd, vs = vdd * rng.random(n), vdd * rng.random(n)
    mode = rng.integers(0, 3, (2, n))      # 0 uniform, 1 near a rail, 2 near the other terminal
    rail = rng.integers(0, 2, (2, n)) * vdd
    near_rail = lambda k: np.where(rail[k] > 0, vdd - dist[k], dist[k])  # noqa: E731
    vd = np.where(mode[0] == 1, near_rail(0), vd)
    vs = np.where(mode[1] == 1, near_rail(1), vs)
    sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    vd = np.where((mode[0] == 2) & (mode[1] != 2), vs + sign * dist[0], vd)
    vs = np.where(mode[1] == 2, vd + sign * dist[1], vs)
    return np.clip(vd, 0, vdd), np.clip(vs, 0, vdd)


def build(case, n, cards, block):
    rng = np.random.default_rng((SEED[case], 900 + block))
    df = draw_inputs(case, n, cards[devtype(case)].params, (SEED[case], 800 + block))
    df["vd"], df["vs"] = edge_voltages(rng, df.vdd.to_numpy())
    _, df["vg"], _, df["vb"] = terminal_voltages(case, df.vdd, df.vd, df.vs)
    res = simulate(to_samples(case, df))
    if any(c is None for c, _ in res):
        raise RuntimeError(next(e for c, e in res if c is None))
    for name in CURRENT_NAMES:
        df[name] = [c[name] for c, _ in res]
    return df[COLUMNS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--blocks", type=int, default=1)
    args = ap.parse_args()
    cards = load_cards(spec.MODEL_FILE)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for case in spec.CASES:
        for k in range(1, args.blocks + 1):
            build(case, args.n, cards, k).to_csv(OUT / f"{case}_{k}.csv", index=False, float_format="%.10g")
            print(f"{case:9s} edge block {k}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
