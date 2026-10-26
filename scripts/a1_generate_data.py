"""Part A, step 1: generate the single-transistor datasets (doc section 3/4).

For each case (NMOS_ON, NMOS_OFF, PMOS_ON, PMOS_OFF) draw N samples, every
parameter independently and uniformly (doc section 4):
  VDD   0.81..0.99 V (drawn first)        Vd, Vs   0..VDD
  W     45..360 nm (1..8 x Lmin)          T        -40..125 degC
  vth0, toxe, ndep, u0, xj   nominal (45nm_MGK.pm) +/-10 %
Vg and Vb are fixed by the case. L = 45 nm. Each sample is one ngspice
operating-point run with a voltage source on every terminal.

Output: data/partA/<CASE>.csv with exactly the doc's columns
  vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj, id, ig, is, ib
The train/validation/test split is made later from a fixed seed (ml/data.py).

--extra K also writes K further blocks of N rows (same distribution, other
seeds) to data/partA_extra/<CASE>_<k>.csv. They are used only to enlarge the
training set in the data-size study; validation and test always come from
the doc's 10,000-row dataset.

Run:  python scripts/a1_generate_data.py [--n 10000] [--extra 0]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim import spec  # noqa: E402
from sim.bias import devtype, terminal_voltages  # noqa: E402
from sim.modelcard import load_cards  # noqa: E402
from sim.runner import CURRENT_NAMES, Sample, simulate  # noqa: E402

OUT_DIR = ROOT / "data" / "partA"
EXTRA_DIR = ROOT / "data" / "partA_extra"
COLUMNS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", *spec.PROCESS_PARAMS, *CURRENT_NAMES]
SEED = {case: 1000 + k for k, case in enumerate(spec.CASES)}


def draw_inputs(case, n, nominal, seed):
    """Independent uniform draws, VDD first (doc section 4)."""
    rng = np.random.default_rng(seed)
    u = lambda lo, hi: rng.uniform(lo, hi, n)  # noqa: E731
    vdd = u(*spec.VDD_RANGE)
    vd = vdd * rng.random(n)
    vs = vdd * rng.random(n)
    df = pd.DataFrame({"vdd": vdd, "temp": u(*spec.TEMP_RANGE), "w": u(*spec.W_RANGE),
                       "vd": vd, "vs": vs})
    _, df["vg"], _, df["vb"] = terminal_voltages(case, vdd, vd, vs)
    for p in spec.PROCESS_PARAMS:
        df[p] = nominal[p] * u(1 - spec.PROCESS_SPREAD, 1 + spec.PROCESS_SPREAD)
    return df


def to_samples(case, df):
    dt = devtype(case)
    return [Sample(dt, r.vd, r.vg, r.vs, r.vb, r.temp, r.w, spec.L,
                   {p: getattr(r, p) for p in spec.PROCESS_PARAMS})
            for r in df.itertuples()]


def build(case, n, cards, block=0):
    seed = SEED[case] if block == 0 else (SEED[case], block)
    df = draw_inputs(case, n, cards[devtype(case)].params, seed)
    results = simulate(to_samples(case, df))
    failed = [e for c, e in results if c is None]
    if failed:
        raise RuntimeError(f"{case}: {len(failed)} simulations failed, e.g. {failed[0]}")
    for name in CURRENT_NAMES:
        df[name] = [c[name] for c, _ in results]
    return df[COLUMNS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--extra", type=int, default=0)
    ap.add_argument("--extra-only", action="store_true")
    args = ap.parse_args()

    cards = load_cards(spec.MODEL_FILE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    for case in spec.CASES:
        if not args.extra_only:
            df = build(case, args.n, cards)
            df.to_csv(OUT_DIR / f"{case}.csv", index=False, float_format="%.10g")
            print(f"{case:9s} {len(df)} rows  ({time.time() - t0:.0f} s)", flush=True)
        for k in range(1, args.extra + 1):
            df = build(case, args.n, cards, block=k)
            df.to_csv(EXTRA_DIR / f"{case}_{k}.csv", index=False, float_format="%.10g")
            print(f"{case:9s} extra block {k}: {len(df)} rows  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
