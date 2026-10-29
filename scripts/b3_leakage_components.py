"""Which currents make up a gate's leakage (report section 4, "What is leakage").

Leakage = VDD supply current + current drawn by the inputs held at VDD
(sim/gates.py). Re-simulates every gate case at the Part B corners with the
model's gate-tunnelling currents switched off (BSIM4 igcmod = igbmod = 0).
What is left is the sub-threshold channel current plus the junction currents,
so
  gate-tunnelling share = 1 - leakage(no tunnelling) / leakage(full model).
Also reports which part of the leakage the inputs held at VDD draw.

Writes data/partB/leakage_components.csv and reports/partB/LEAKAGE_COMPONENTS.md.
Run:  python scripts/b3_leakage_components.py   (after b1_gates_spice.py)
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import b1_gates_spice as b1  # noqa: E402
from sim import gates  # noqa: E402
from sim.runner import parallel_map  # noqa: E402

DATA = ROOT / "data" / "partB"
NO_TUNNEL = {"igcmod": 0.0, "igbmod": 0.0}


def _job(args):
    gate, bits, corner = args
    inputs, internal, fets = gates.GATES[gate]
    cards = {k: v.with_overrides(NO_TUNNEL) for k, v in gates.cards().items()}
    res, err = gates.run(gates.build(f"{gate} no tunnelling", fets, inputs, bits, ("out",), corner, cards))
    if res is None:
        raise RuntimeError(err)
    return gates.total_leakage(res["leak"], [res[f"pin_{n}"] for n in inputs], bits)


def main():
    spice = pd.read_csv(DATA / "gates_spice.csv", dtype={"inputs": str})
    corners = pd.read_csv(DATA / "corners.csv").to_dict("records")
    jobs = [(g, b, c) for g, b in b1.cases() for c in corners]
    no_tun = parallel_map(_job, jobs, chunksize=8)
    nt = pd.DataFrame([{"gate": g, "inputs": "".join(map(str, b)), "corner": int(c["corner"]), "leak_no_tunnel": v}
                       for (g, b, c), v in zip(jobs, no_tun)])
    df = spice.merge(nt, on=["gate", "inputs", "corner"])
    df["tunnel_share"] = 100 * (1 - df.leak_no_tunnel / df.leak)
    df["pin_share"] = 100 * (df.leak - df.leak_vdd) / df.leak
    df[["gate", "inputs", "corner", "leak", "leak_vdd", "leak_no_tunnel", "tunnel_share", "pin_share"]].to_csv(
        DATA / "leakage_components.csv", index=False, float_format="%.6g")

    lines = ["# Part B: what the gate leakage consists of", "",
             "Leakage = VDD supply current + current drawn by the inputs held at VDD. Gate-tunnelling share = 1 − "
             "leakage with the model's gate tunnelling switched off (igcmod = igbmod = 0) / leakage of the full "
             "model; the rest is sub-threshold channel current plus junction current. Drawn by inputs at VDD = the "
             "part of the leakage that flows in through the inputs held at VDD rather than the VDD supply pin. "
             "Medians over 200 corners.", "",
             "| gate | inputs | leakage (A) | gate-tunnelling share | sub-threshold + junction share | drawn by inputs at VDD |",
             "|---|---|---|---|---|---|"]
    for (g, i), r in df.groupby(["gate", "inputs"], sort=False):
        ts = r.tunnel_share.median()
        lines.append(f"| {g} | {i} | {r.leak.median():.3e} | {ts:.0f} % | {100 - ts:.0f} % | {r.pin_share.median():.0f} % |")
    lines += ["", f"All 3,600 simulations: gate tunnelling is {df.tunnel_share.median():.0f} % of the leakage "
              f"(median; {df.tunnel_share.quantile(0.05):.0f}–{df.tunnel_share.quantile(0.95):.0f} % for 90 % of runs).", ""]
    (ROOT / "reports" / "partB" / "LEAKAGE_COMPONENTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
