"""Error analysis (report section "Error analysis").

Where are the largest errors? For the chosen transistor models (test set),
the gate results and the parity checker: error by temperature, VDD and
process (vth0, toxe) bins, plus the worst cases.

Writes reports/ERROR_ANALYSIS.md and reports/error_vs_corner.png.
Run:  python scripts/d1_error_analysis.py   (after a4, b2 and c2)
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis import plotstyle  # noqa: E402
from ml import data, metrics  # noqa: E402
from sim import spec  # noqa: E402
from sim.modelcard import load_cards  # noqa: E402

REPORT = ROOT / "reports"
BINS = {"temp": np.linspace(-40, 125, 6), "vdd": np.linspace(*spec.VDD_RANGE, 5)}


def binned(df, col, err, bins):
    cut = pd.cut(df[col], bins, include_lowest=True)
    g = df.assign(_e=err).groupby(cut, observed=True)._e
    return pd.DataFrame({"median": g.median(), "p95": g.quantile(0.95), "n": g.size()})


def fmt_bins(t, unit=""):
    return " | ".join(f"{max(iv.left, 0) if unit == ' V' else iv.left:.3g}–{iv.right:.3g}{unit}: {r['median']:.2f} / {r['p95']:.1f} %"
                      for iv, r in t.iterrows())


def part_a(lines):
    lines += ["## Part A: chosen transistor models on the test set", "",
              "Median / p95 of the worst of id, ig and is per sample, by input range. The body current ib is "
              "listed separately: in the OFF cases it is ~1e-14 A, 6-8 decades below the logic-gate leakages, and it "
              "would otherwise dominate the statistics. This per-sample worst-of-three metric is stricter than the "
              "pooled metric of reports/partA/MODELS.md, so its p95 values are higher.", ""]
    cards = load_cards(spec.MODEL_FILE)
    for case in data.CASES:
        df = data.load(case)
        test = df.iloc[data.split_index(len(df))["test"]].reset_index(drop=True)
        model = joblib.load(ROOT / "trained_models" / f"{case}.joblib")
        err = metrics.rel_err(model.predict(test), test[list(data.CURRENTS)].to_numpy())
        pooled = err[:, :3].max(axis=1)   # worst of id, ig, is per sample
        nom = cards[case[:4].lower()].params
        test["vth0_dev"] = 100 * (test.vth0 / nom["vth0"] - 1)
        test["toxe_dev"] = 100 * (test.toxe / nom["toxe"] - 1)
        test["vds"] = (test.vd - test.vs).abs()
        ib_e = err[:, 3]
        lines.append(f"**{case}** (worst of id, ig, is per sample; ib: median {np.median(ib_e):.2f} %, "
                     f"p95 {np.percentile(ib_e, 95):.1f} %, median |ib| {np.median(np.abs(test.ib)):.1e} A):")
        lines.append(f"- temperature: {fmt_bins(binned(test, 'temp', pooled, BINS['temp']), ' °C')}")
        lines.append(f"- vth0 deviation: {fmt_bins(binned(test, 'vth0_dev', pooled, np.linspace(-10, 10, 5)), ' %')}")
        lines.append(f"- toxe deviation: {fmt_bins(binned(test, 'toxe_dev', pooled, np.linspace(-10, 10, 5)), ' %')}")
        lines.append(f"- |Vd − Vs|: {fmt_bins(binned(test, 'vds', pooled, [0, 0.01, 0.05, 0.2, 0.5, 1.0]), ' V')}")
        worst = int(np.argmax(err.max(axis=0)))
        lines.append(f"- current with the largest p95: {data.CURRENTS[np.argmax(np.percentile(err, 95, axis=0))]} "
                     f"(p95 {np.percentile(err, 95, axis=0).max():.1f} %); worst single error {err.max():.0f} % on "
                     f"{data.CURRENTS[worst]} at |I| = {abs(test[data.CURRENTS[worst]].iloc[int(np.argmax(err[:, worst]))]):.2e} A")
        lines.append("")


def gates_and_circuit(lines):
    g = pd.read_csv(ROOT / "data" / "partB" / "gates_ml.csv", dtype={"inputs": str})
    c = pd.read_csv(ROOT / "data" / "partB" / "corners.csv")
    g = g.merge(c, on="corner")
    nom = load_cards(spec.MODEL_FILE)
    g["n_vth0_dev"] = 100 * (g.n_vth0 / nom["nmos"].params["vth0"] - 1)
    g["p_vth0_dev"] = 100 * (g.p_vth0 / nom["pmos"].params["vth0"] - 1)
    g["signed"] = 100 * (g.ml_leak / g.leak - 1)
    lines += ["## Part B: gates by corner condition", "", "Median / p95 error over all 18 gate cases.", "",
              f"- temperature: {fmt_bins(binned(g, 'temp', g.err, BINS['temp']), ' °C')}",
              f"- VDD: {fmt_bins(binned(g, 'vdd', g.err, BINS['vdd']), ' V')}",
              f"- NMOS vth0 deviation: {fmt_bins(binned(g, 'n_vth0_dev', g.err, np.linspace(-10, 10, 5)), ' %')}",
              f"- PMOS vth0 deviation: {fmt_bins(binned(g, 'p_vth0_dev', g.err, np.linspace(-10, 10, 5)), ' %')}",
              f"- NMOS width Wn: {fmt_bins(binned(g.assign(wn_nm=g.wn * 1e9), 'wn_nm', g.err, np.linspace(45, 360, 5)), ' nm')}",
              f"- PMOS width Wp: {fmt_bins(binned(g.assign(wp_nm=g.wp * 1e9), 'wp_nm', g.err, np.linspace(45, 360, 5)), ' nm')}", ""]
    worst = g.groupby(["gate", "inputs"]).err.median().sort_values(ascending=False)
    lines += ["Gate cases by median error (worst first): " +
              ", ".join(f"{a} {b}: {v:.2f} %" for (a, b), v in worst.items()), ""]
    lines += [f"Signed bias by output state: output high {np.median(g.signed[g.out_logic == 1]):+.2f} %, "
              f"output low {np.median(g.signed[g.out_logic == 0]):+.2f} %.", ""]

    p = pd.read_csv(ROOT / "data" / "partC" / "parity_ml.csv", dtype={"inputs": str}).merge(c, on="corner")
    lines += ["## Part C: parity checker by corner condition", "",
              f"- temperature: {fmt_bins(binned(p, 'temp', p.err, BINS['temp']), ' °C')}",
              f"- VDD: {fmt_bins(binned(p, 'vdd', p.err, BINS['vdd']), ' V')}", ""]

    plotstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, (col, xl) in zip(axes, (("temp", "temperature (°C)"), ("vdd", "VDD (V)"))):
        for df, e, name, color in ((g, g.err, "gates", "#2a78d6"), (p, p.err, "parity checker", "#eb6834")):
            t = binned(df, col, e, BINS[col])
            mid = [iv.mid for iv in t.index]
            ax.plot(mid, t["median"], "o-", color=color, label=f"{name}: median")
            ax.plot(mid, t["p95"], "o--", color=color, alpha=0.6, label=f"{name}: p95")
        ax.set(xlabel=xl)
    axes[0].set_ylabel("|ML − SPICE| / SPICE (%)")
    axes[1].legend(fontsize=7)
    fig.suptitle("Leakage error vs corner condition", fontsize=10)
    fig.savefig(REPORT / "error_vs_corner.png")
    plt.close(fig)
    lines += ["![Error vs corner](error_vs_corner.png)", ""]


def main():
    lines = ["# Error analysis", ""]
    part_a(lines)
    gates_and_circuit(lines)
    (REPORT / "ERROR_ANALYSIS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
