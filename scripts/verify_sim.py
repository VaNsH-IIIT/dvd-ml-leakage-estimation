"""Verify the simulation driver (sim/) before trusting any dataset.

Checks:
 1. The driver reproduces the professor's sample netlist (22n_NMOS_ON.net)
    exactly, at every Vref step and both temperatures it simulates.
 2. Kirchhoff: the four terminal currents sum to ~0 in all four cases
    (team model 45nm_MGK, the doc's .options).
 3. Physics sanity: off-state leakage rises with temperature and falls as
    vth0 increases; swapping drain and source mirrors the channel current.

Run:  python scripts/verify_sim.py      (exit code 0 = all checks passed)
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim import spec  # noqa: E402
from sim.bias import devtype, terminal_voltages  # noqa: E402
from sim.runner import NGSPICE, Sample, simulate  # noqa: E402

PROF_NETLIST = ROOT / "problem-statement" / "22n_NMOS_ON.net"
PROF_MODEL = ROOT / "models" / "22nm_MGK.pm"       # the model his sample uses
PROF_OPTIONS = ".OPTIONS GMIN=1e-020 ABSTOL=1e-18"  # his sample's options
failures = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def run_professor_netlist():
    """Run 22n_NMOS_ON.net (only its .INCLUDE path changed) and parse rows."""
    text = PROF_NETLIST.read_text().replace("\r", "")
    text = re.sub(r"^\.INCLUDE .*$", f".INCLUDE {PROF_MODEL}", text, flags=re.M)
    with tempfile.TemporaryDirectory() as d:
        net = Path(d) / "prof.net"
        net.write_text(text)
        out = subprocess.run([NGSPICE, "-b", str(net)], capture_output=True, text=True).stdout
    num = r"([-+0-9.e]+)"
    rows = re.findall(rf"^([01])\t{num}\t{num}\t{num}\t{num}\s*$", out, re.M)
    results = []
    for k in range(0, len(rows), 6):  # 3 tables x 2 temperatures per Vdd step
        blk = rows[k:k + 6]
        for t in (0, 1):
            a, b, c = blk[t], blk[2 + t], blk[4 + t]
            results.append(dict(temp=float(a[1]), v=float(a[2]), idrain=float(b[4]),
                                igate=float(c[2]), isource=float(c[3]), ibulk=float(c[4])))
    return results


from sim.modelcard import load_cards  # noqa: E402

CARDS = load_cards(spec.MODEL_FILE)


def main():
    print("1. Reproduce professor's sample netlist (NMOS_ON)")
    ref = run_professor_netlist()
    got = simulate([Sample("nmos", 0, r["v"], 0, 0, r["temp"], 22e-9, 22e-9) for r in ref],
                   model_file=PROF_MODEL, options=PROF_OPTIONS)
    worst = 0.0
    for r, (c, err) in zip(ref, got):
        if c is None:
            check(False, f"driver failed at v={r['v']} T={r['temp']}: {err}")
            continue
        for k, ours in (("idrain", "id"), ("igate", "ig"), ("isource", "is"), ("ibulk", "ib")):
            worst = max(worst, abs(c[ours] - r[k]) / max(abs(r[k]), 1e-30))
    # The loop runs Vdd = 0.05..1.2 V in 0.05 V steps (24 steps) at 30 and 90 degC.
    check(len(ref) == 48, f"professor netlist produced {len(ref)} points (expect 24 Vdd x 2 T)")
    # Agreement is limited by print precision (ngspice prints negative numbers
    # with 6 significant digits) and by solver tolerances (his run is a DC
    # temperature sweep, ours an operating point). 1e-4 is still 100x below
    # the project's 1% error target.
    check(worst < 1e-4, f"max relative difference vs professor netlist = {worst:.2e}")

    print(f"2. Kirchhoff current law, all four cases ({spec.MODEL_FILE.name}, doc options)")
    vdd = spec.VDD_NOM

    def dev(case, vd, vs, temp=27.0, vth0_scale=1.0):
        _, vg, _, vb = terminal_voltages(case, vdd, vd, vs)
        return Sample(devtype(case), vd, vg, vs, vb, temp, spec.L, spec.L,
                      {"vth0": vth0_scale * CARDS[devtype(case)].params["vth0"]})

    cases = tuple(spec.CASES)
    res = dict(zip(cases, simulate([dev(c, vdd, 0) for c in cases])))
    for case, (c, _) in res.items():
        total = sum(c.values())
        scale = max(abs(x) for x in c.values())
        check(abs(total) < 1e-4 * scale, f"{case}: sum of terminal currents {total:+.1e} A (scale {scale:.1e} A)")

    print("3. Physics sanity")
    (cold, _), (hot, _) = simulate([dev("NMOS_OFF", vdd, 0, temp=-40), dev("NMOS_OFF", vdd, 0, temp=125)])
    check(abs(hot["is"]) > 10 * abs(cold["is"]),
          f"NMOS_OFF sub-threshold current grows with T: {abs(cold['is']):.2e} -> {abs(hot['is']):.2e} A")
    (lo, _), (hi, _) = simulate([dev("NMOS_OFF", vdd, 0, vth0_scale=0.9), dev("NMOS_OFF", vdd, 0, vth0_scale=1.1)])
    check(abs(lo["is"]) > abs(hi["is"]),
          f"higher vth0 -> lower off leakage: {abs(lo['is']):.2e} > {abs(hi['is']):.2e} A")
    (fwd, _), (rev, _) = simulate([dev("NMOS_ON", 0.5, 0.1), dev("NMOS_ON", 0.1, 0.5)])
    check(abs(fwd["id"] - rev["is"]) < 1e-3 * abs(fwd["id"]),
          f"swapping drain and source mirrors the channel current (id = {fwd["id"]:.6e} A, swapped is = {rev["is"]:.6e} A)")

    print("4. Netlist formatting")
    import numpy as np
    from sim.runner import build_netlist
    f = np.float64
    text = build_netlist(Sample("nmos", f(0.9), f(0.9), f(0), f(0), f(27), f(90e-9), f(45e-9),
                                {"vth0": f(0.35)}), CARDS)
    check("np." not in text and "float64" not in text,
          "numpy inputs are written as plain numbers (regression: L/W sweeps failed in the pilot)")

    print()
    print("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
