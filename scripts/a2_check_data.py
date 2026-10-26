"""Part A data check (deliverables 2 and 3).

  1. Completeness and ranges: 10,000 rows per case, no missing values, every
     input inside its doc range, Vg/Vb fixed by the case.
  2. Kirchhoff: id + ig + is + ib = 0 in every row.
  3. Trend plots (SPICE sweeps around the nominal point, one input at a time):
       OFF current vs temperature, OFF current vs vth0, gate current vs toxe.
     Plus the same trends measured on the dataset itself (rank correlation).
  4. Example netlists: the first sample of each case -> netlists/partA/.

Writes reports/partA/DATA_CHECK.md and the figures. Exit code 0 = all passed.
Run:  python scripts/a2_check_data.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis import plotstyle  # noqa: E402
from sim import spec  # noqa: E402
from sim.bias import devtype, terminal_voltages  # noqa: E402
from sim.modelcard import load_cards  # noqa: E402
from sim.runner import CURRENT_NAMES, Sample, build_netlist, simulate  # noqa: E402

DATA = ROOT / "data" / "partA"
REPORT = ROOT / "reports" / "partA"
NETLISTS = ROOT / "netlists" / "partA"
N_EXPECTED = 10_000
failures, lines = [], []


def check(cond, msg):
    lines.append(f"- {'✅' if cond else '❌'} {msg}")
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def load(case):
    return pd.read_csv(DATA / f"{case}.csv")


# ---------------------------------------------------------------- 1. ranges
def check_ranges(cards):
    lines.append("\n## 1. Completeness and ranges\n")
    for case in spec.CASES:
        df = load(case)
        nom = cards[devtype(case)].params
        ok = len(df) == N_EXPECTED and not df.isna().any().any()
        ok &= df.vdd.between(*spec.VDD_RANGE).all()
        ok &= (df.vd.between(0, df.vdd) & df.vs.between(0, df.vdd)).all()
        ok &= df.w.between(spec.W_RANGE[0] * (1 - 1e-9), spec.W_RANGE[1] * (1 + 1e-9)).all()
        ok &= df.temp.between(*spec.TEMP_RANGE).all()
        _, vg, _, vb = terminal_voltages(case, df.vdd, df.vd, df.vs)
        ok &= np.allclose(df.vg, vg, atol=1e-9) and np.allclose(df.vb, vb, atol=1e-9)
        for p in spec.PROCESS_PARAMS:
            r = df[p] / nom[p]
            ok &= r.between(0.9 - 1e-9, 1.1 + 1e-9).all()
        check(ok, f"{case}: {len(df)} rows, no missing values, every input inside its doc range")


# ------------------------------------------------------------- 2. Kirchhoff
def check_kirchhoff():
    lines.append("\n## 2. Kirchhoff check (id + ig + is + ib = 0)\n")
    lines.append("| case | max \\|sum\\| (A) | median \\|sum\\| / max\\|I\\| | max \\|sum\\| / max\\|I\\| | rows with \\|sum\\| > 1e-15 A and > 1e-3·max\\|I\\| |")
    lines.append("|---|---|---|---|---|")
    rows = []
    for case in spec.CASES:
        df = load(case)
        cur = df[list(CURRENT_NAMES)]
        s = cur.sum(axis=1).abs()
        rel = s / cur.abs().max(axis=1)
        bad = int(((s > 1e-15) & (rel > 1e-3)).sum())
        rows.append((case, s, rel))
        lines.append(f"| {case} | {s.max():.1e} | {rel.median():.1e} | {rel.max():.1e} | {bad} |")
    lines.append("")
    for case, s, rel in rows:
        check(((s <= 1e-15) | (rel <= 1e-3)).all(),
              f"{case}: every row balances to within 1e-15 A or 0.1 % of its largest current")
    lines.append("\nThe residual comes from ngspice printing 6–7 significant digits and from the "
                 "solver tolerances (abstol = 1e-18 A, reltol = 1e-6). It is far below the 1 % accuracy target.")


# ----------------------------------------------------------- 3. trend plots
def nominal_sample(case, cards, **kw):
    """The doc's nominal point: VDD = 0.9 V, 27 degC, W = Lmin, nominal process.
    OFF current is taken at full drain-source bias; gate current at Vd = Vs = 0
    (NMOS ON) / VDD (PMOS ON)."""
    vdd = spec.VDD_NOM
    dt = devtype(case)
    vd, vs = {"NMOS_OFF": (vdd, 0), "PMOS_OFF": (0, vdd), "NMOS_ON": (0, 0), "PMOS_ON": (vdd, vdd)}[case]
    _, vg, _, vb = terminal_voltages(case, vdd, vd, vs)
    nom = cards[dt].params
    proc = {p: nom[p] * kw.pop(f"{p}_scale", 1.0) for p in spec.PROCESS_PARAMS}
    return Sample(dt, vd, vg, vs, vb, kw.pop("temp", 27.0), spec.L, spec.L, proc)


def off_current(case, c):
    """Drain-to-source leakage of an OFF device: the current at the low-voltage terminal."""
    return abs(c["is"] if case == "NMOS_OFF" else c["id"])


def trend_plots(cards):
    lines.append("\n## 3. Trend plots\n")
    plotstyle.apply()
    color = {"NMOS": plotstyle.CURRENT_COLORS["idrain"], "PMOS": plotstyle.CURRENT_COLORS["igate"]}
    sweeps = [
        ("off_vs_temp", "OFF current vs temperature", "temperature (°C)", "OFF current |I| (A)",
         np.linspace(-40, 125, 34), "temp", ("NMOS_OFF", "PMOS_OFF"), off_current),
        ("off_vs_vth0", "OFF current vs vth0", "vth0 / nominal", "OFF current |I| (A)",
         np.linspace(0.9, 1.1, 21), "vth0_scale", ("NMOS_OFF", "PMOS_OFF"), off_current),
        ("gate_vs_toxe", "Gate current vs toxe", "toxe / nominal", "gate current |Ig| (A)",
         np.linspace(0.9, 1.1, 21), "toxe_scale", ("NMOS_ON", "PMOS_ON"), lambda case, c: abs(c["ig"])),
    ]
    results = {}
    for key, title, xlabel, ylabel, xs, arg, cases, metric in sweeps:
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        for case in cases:
            res = simulate([nominal_sample(case, cards, **{arg: x}) for x in xs])
            ys = np.array([metric(case, c) for c, _ in res])
            results[(key, case)] = (xs, ys)
            ax.semilogy(xs, ys, color=color[case[:4]], label=case.replace("_", " "))
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend()
        fig.savefig(REPORT / f"trend_{key}.png")
        plt.close(fig)
        lines.append(f"![{title}](trend_{key}.png)\n")

    lines.append("| sweep | case | low end | high end | ratio |")
    lines.append("|---|---|---|---|---|")
    for (key, case), (xs, ys) in results.items():
        lines.append(f"| {key} | {case} | {ys[0]:.3e} A | {ys[-1]:.3e} A | ×{ys[-1] / ys[0]:.3g} |")
    lines.append("")
    for case in ("NMOS_OFF", "PMOS_OFF"):
        xs, ys = results[("off_vs_temp", case)]
        check(np.all(np.diff(ys) > 0) and ys[-1] > 10 * ys[0], f"{case}: OFF current rises steadily with temperature (×{ys[-1] / ys[0]:.0f} over −40..125 °C)")
        xs, ys = results[("off_vs_vth0", case)]
        check(np.all(np.diff(ys) < 0), f"{case}: OFF current falls as |vth0| rises (×{ys[-1] / ys[0]:.2f} over ±10 %)")
    for case in ("NMOS_ON", "PMOS_ON"):
        xs, ys = results[("gate_vs_toxe", case)]
        check(np.all(np.diff(ys) < 0), f"{case}: gate current falls as toxe rises (×{ys[-1] / ys[0]:.2f} over ±10 %)")


def dataset_trends():
    """The same trends inside the random dataset: Spearman rank correlation of
    log|I| with each input. Rows with |Vd - Vs| > 0.5 VDD keep the OFF current
    well defined."""
    lines.append("\n### The same trends in the random dataset\n")
    lines.append("Spearman rank correlation of the current with each input, over the whole dataset "
                 "(OFF cases: rows with |Vd − Vs| > 0.5·VDD, OFF current at the low-voltage terminal).\n")
    pending = []
    lines.append("| case | current | vs temperature | vs \\|vth0\\| | vs toxe |")
    lines.append("|---|---|---|---|---|")
    for case in spec.CASES:
        df = load(case)
        if case.endswith("OFF"):
            sel = (df.vd - df.vs).abs() > 0.5 * df.vdd
            df = df[sel]
            lo_is_s = df.vs < df.vd
            i = np.where(lo_is_s, df["is"], df["id"])
            name = "OFF current"
        else:
            i = df["ig"]
            name = "gate current"
        a = np.abs(i)
        r = [spearmanr(df[x].abs(), a)[0] for x in ("temp", "vth0", "toxe")]
        lines.append(f"| {case} | {name} | {r[0]:+.2f} | {r[1]:+.2f} | {r[2]:+.2f} |")
        if case.endswith("OFF"):
            pending.append((r[0] > 0.1 and r[1] < -0.05, f"{case} dataset: OFF current rises with T (ρ = {r[0]:+.2f}) and falls with |vth0| (ρ = {r[1]:+.2f})"))
        else:
            pending.append((r[2] < 0, f"{case} dataset: gate current falls with toxe (ρ = {r[2]:+.2f})"))
    lines.append("")
    for ok, msg in pending:   # after the table, so the table stays in one piece
        check(ok, msg)


# ------------------------------------------------------- 4. example netlists
def example_netlists(cards):
    NETLISTS.mkdir(parents=True, exist_ok=True)
    for case in spec.CASES:
        r = load(case).iloc[0]
        s = Sample(devtype(case), r.vd, r.vg, r.vs, r.vb, r.temp, r.w, spec.L,
                   {p: r[p] for p in spec.PROCESS_PARAMS})
        text = build_netlist(s, cards, title=f"{case}: first sample of data/partA/{case}.csv")
        (NETLISTS / f"{case}.cir").write_text(text)
    lines.append("\n## 4. Example netlists\n")
    lines.append("`netlists/partA/<CASE>.cir`: the first sample of each dataset, exactly as simulated "
                 "(run with `ngspice -b netlists/partA/NMOS_ON.cir`).")


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    cards = load_cards(spec.MODEL_FILE)
    print("1. ranges");      check_ranges(cards)
    print("2. Kirchhoff");   check_kirchhoff()
    print("3. trends");      trend_plots(cards); dataset_trends()
    print("4. netlists");    example_netlists(cards)
    head = ["# Part A data check", "",
            f"Model `{spec.MODEL_FILE.name}`, nominal VDD {spec.VDD_NOM} V, L = {spec.L * 1e9:.0f} nm. "
            f"Datasets: `data/partA/<CASE>.csv`, {N_EXPECTED:,} rows each (generated by `scripts/a1_generate_data.py`).",
            "", f"**Result: {'all checks passed' if not failures else f'{len(failures)} check(s) failed'}.**"]
    (REPORT / "DATA_CHECK.md").write_text("\n".join(head + lines) + "\n")
    print("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
