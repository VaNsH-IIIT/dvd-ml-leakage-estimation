"""Demo: predict one transistor operating point with a saved model and check it
against a live SPICE run (useful in the viva).

Example:
  python scripts/demo_predict.py NMOS_OFF --vdd 0.9 --temp 85 --w 90e-9 --vd 0.9 --vs 0
  python scripts/demo_predict.py PMOS_ON --vd 0.85 --vs 0.9 --vth0-scale 0.95

Process parameters default to the 45nm_MGK nominal values; --<param>-scale
multiplies them (0.9..1.1 is the trained range).
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim import spec  # noqa: E402
from sim.bias import devtype, terminal_voltages  # noqa: E402
from sim.modelcard import load_cards  # noqa: E402
from sim.runner import Sample, simulate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", choices=list(spec.CASES))
    ap.add_argument("--vdd", type=float, default=spec.VDD_NOM)
    ap.add_argument("--temp", type=float, default=27.0)
    ap.add_argument("--w", type=float, default=spec.L)
    ap.add_argument("--vd", type=float, default=None, help="default: VDD")
    ap.add_argument("--vs", type=float, default=0.0)
    for p in spec.PROCESS_PARAMS:
        ap.add_argument(f"--{p}-scale", type=float, default=1.0)
    a = ap.parse_args()

    vd = a.vdd if a.vd is None else a.vd
    dt = devtype(a.case)
    nominal = load_cards(spec.MODEL_FILE)[dt].params
    _, vg, _, vb = terminal_voltages(a.case, a.vdd, vd, a.vs)
    proc = {p: nominal[p] * getattr(a, f"{p}_scale") for p in spec.PROCESS_PARAMS}
    row = pd.DataFrame([{"vdd": a.vdd, "temp": a.temp, "w": a.w, "vd": vd, "vs": a.vs, "vg": vg, "vb": vb, **proc}])

    model = joblib.load(ROOT / "trained_models" / f"{a.case}.joblib")
    ml = model.predict(row)[0]
    (sp, err), = simulate([Sample(dt, vd, vg, a.vs, vb, a.temp, a.w, spec.L, proc)], workers=1)
    if sp is None:
        sys.exit(f"SPICE failed: {err}")

    print(f"{a.case}: VDD={a.vdd} V, T={a.temp} degC, W={a.w * 1e9:.0f} nm, "
          f"Vd={vd} V, Vg={vg} V, Vs={a.vs} V, Vb={vb} V")
    print(f"{'current':>8} {'ML (A)':>13} {'SPICE (A)':>13} {'error':>8}")
    for k, name in enumerate(("id", "ig", "is", "ib")):
        e = abs(ml[k] - sp[name]) / abs(sp[name]) * 100
        print(f"{name:>8} {ml[k]:13.4e} {sp[name]:13.4e} {e:7.2f} %")


if __name__ == "__main__":
    main()
