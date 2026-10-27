"""Part A, step 2b: pick the best model per case on the validation set, report
it once on the test set, and save it (deliverables 4 and 5).

Candidates: the four model types of a3_compare_models.py (7,000 training
rows), the MLPs of the data-size study, the MLP ensembles (a3b_data_size.py,
per-width targets, plain-current targets, linear fit + MLPs on its residual)
and averages of those variants at the same size (built and scored here). Selection rule:
lowest validation p95 error over the four currents, then lowest median. The
test split is used only here, for the chosen models.

Writes trained_models/<CASE>.joblib, data/partA_results/chosen.csv,
reports/partA/MODELS.md and reports/partA/pred_vs_spice_<CASE>.png.
Run:  python scripts/a4_select_models.py
"""
import itertools
import os
import shutil
import sys
import warnings
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from a3_compare_models import evaluate  # noqa: E402
from analysis import plotstyle  # noqa: E402
from ml import data, metrics  # noqa: E402
from ml.models import MODEL_TYPES, Blend  # noqa: E402

RES = ROOT / "data" / "partA_results"
RUNS = ROOT / "runs" / "partA"
OUT = ROOT / "trained_models"
REPORT = ROOT / "reports" / "partA"
LABEL = {"poly": "polynomial (deg 3) + ridge", "lgbm": "LightGBM (gradient-boosted trees)",
         "mlp": "MLP (3×128, tanh)", "gp": "Gaussian process (ARD RBF)"}
N_DOC = 7000
I_FLOOR = 1e-11   # A; 1,000x below the smallest gate leakage (Part B), for the "mean without negligible currents"


VARIANTS = {"mlp_ens3": "per-width targets", "mlp_ens3_cur": "plain-current targets",
            "mlp_ens3_lin": "linear fit + MLPs on its residual"}


def label(model):
    if model.startswith("blend"):
        return "average of " + " / ".join(VARIANTS[m] for m in model.split(":")[1].split("+")) + " ensembles"
    if model in VARIANTS:
        return f"ensemble of 3 MLPs (3×128, tanh), {VARIANTS[model]}"
    if model.startswith("mlp_ens"):
        return f"ensemble of {model[7:]} MLPs (3×128, tanh)"
    return LABEL[model]


def size_results():
    """Data-size study rows (data_size.csv) plus the ensembles (data_size_ens.csv, written separately so the
    two trainings can run side by side)."""
    parts = [pd.read_csv(RES / f) for f in ("data_size.csv", "data_size_ens.csv") if (RES / f).exists()]
    out = pd.concat(parts, ignore_index=True)
    if "model" not in out:
        out["model"] = "mlp"
    return out.drop_duplicates(["case", "model", "n_train", "current"], keep="last")


def blend_candidates():
    """Averages of the currents predicted by two or three of the 3-MLP ensemble variants of the same
    training size (runs/partA/), scored on the validation set like every other candidate."""
    rows = []
    for case in data.CASES:
        df = data.load(case)
        val = df.iloc[data.split_index(len(df))["val"]].reset_index(drop=True)
        sizes = sorted({int(f.stem.rsplit("_n", 1)[1]) for f in RUNS.glob(f"{case}_mlp_ens3*_n*.joblib")})
        for n in sizes:
            have = [v for v in VARIANTS if (RUNS / f"{case}_{v}_n{n}.joblib").exists()]
            for k in range(2, len(have) + 1):
                for combo in itertools.combinations(have, k):
                    name = "blend:" + "+".join(combo)
                    model = Blend([joblib.load(RUNS / f"{case}_{v}_n{n}.joblib") for v in combo])
                    joblib.dump(model, RUNS / f"{case}_{name.replace(':', '_').replace('+', '-')}_n{n}.joblib")
                    rows += [dict(case=case, model=name, n_train=n, **r) for r in evaluate(model, val)]
    out = pd.DataFrame(rows)
    out.to_csv(RES / "blend_val.csv", index=False, float_format="%.6g")
    return out


def candidates():
    cmp_ = pd.read_csv(RES / "compare_val.csv")
    cmp_["n_train"] = N_DOC
    cmp_["file"] = cmp_.case + "_" + cmp_.model + ".joblib"
    parts = [cmp_]
    if (RES / "data_size.csv").exists():
        ds = size_results()
        if "model" not in ds:
            ds["model"] = "mlp"
        ds = ds[(ds.n_train > N_DOC) | (ds.model != "mlp")]
        ds["file"] = ds.case + "_" + ds.model + "_n" + ds.n_train.astype(str) + ".joblib"
        parts.append(ds)
    bl = blend_candidates()
    if len(bl):
        bl["file"] = (bl.case + "_" + bl.model.str.replace(":", "_").str.replace("+", "-") + "_n"
                      + bl.n_train.astype(str) + ".joblib")
        parts.append(bl)
    return pd.concat(parts, ignore_index=True)


def comparison_table(c):
    allc = c[(c.current == "all") & (c.n_train == N_DOC)].set_index(["case", "model"])
    lines = ["| case | metric | " + " | ".join(LABEL[m].split(" (")[0] for m in MODEL_TYPES) + " |",
             "|---|---|" + "---|" * len(MODEL_TYPES)]
    for case in data.CASES:
        for key, name, fmt in (("r2_min", "R² (worst current)", "{:.4f}"), ("median", "median error", "{:.2f} %"),
                               ("p95", "p95 error", "{:.2f} %")):
            vals = []
            for m in MODEL_TYPES:
                v = allc[key].get((case, m), np.nan)
                vals.append("—" if pd.isna(v) else fmt.format(v))
            lines.append(f"| {case} | {name} | " + " | ".join(vals) + " |")
    return lines


def per_current_table(c):
    lines = ["| case | model | " + " | ".join(f"{k} median / p95" for k in data.CURRENTS) + " |",
             "|---|---|" + "---|" * 4]
    d = c[c.n_train == N_DOC].set_index(["case", "model", "current"])
    for case in data.CASES:
        for m in MODEL_TYPES:
            if (case, m, "id") not in d.index:
                continue
            cells = [f"{d.loc[(case, m, k), 'median']:.2f} / {d.loc[(case, m, k), 'p95']:.1f} %" for k in data.CURRENTS]
            lines.append(f"| {case} | {m} | " + " | ".join(cells) + " |")
    return lines


def plot_case(case, model, test):
    plotstyle.apply()
    pred = model.predict(test)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    colors = list(plotstyle.CURRENT_COLORS.values())
    for k, (ax, name) in enumerate(zip(axes, data.CURRENTS)):
        t, p = np.abs(test[name].to_numpy()), np.abs(pred[:, k])
        ax.loglog(t, p, ".", ms=2.5, alpha=0.5, color=colors[k])
        lo, hi = min(t.min(), p.min()) / 2, max(t.max(), p.max()) * 2
        ax.plot([lo, hi], [lo, hi], color=plotstyle.MUTED, lw=1)
        e = metrics.rel_err(pred[:, k], test[name].to_numpy())
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel=f"SPICE |{name}| (A)",
               title=f"{name}: median {np.median(e):.2f} %, p95 {np.percentile(e, 95):.1f} %")
    axes[0].set_ylabel("ML |I| (A)")
    fig.suptitle(f"{case}: predicted vs SPICE on the test set ({len(test)} samples)", fontsize=10)
    fig.savefig(REPORT / f"pred_vs_spice_{case}.png")
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(exist_ok=True)
    c = candidates()
    allc = c[c.current == "all"].sort_values(["case", "p95", "median"])
    chosen, test_rows, lines_choice = [], [], []
    for case in data.CASES:
        best = allc[allc.case == case].iloc[0]
        model = joblib.load(RUNS / best.file)
        shutil.copy(RUNS / best.file, OUT / f"{case}.joblib")
        df = data.load(case)
        test = df.iloc[data.split_index(len(df))["test"]].reset_index(drop=True)
        rows = evaluate(model, test)
        true = test[list(data.CURRENTS)].to_numpy()
        big = np.abs(true) >= I_FLOOR
        mean_big = metrics.rel_err(model.predict(test), true)[big].mean()
        plot_case(case, model, test)
        t = rows[-1]
        chosen.append(dict(case=case, model=best.model, n_train=int(best.n_train),
                           val_median=best["median"], val_p95=best.p95,
                           test_median=t["median"], test_p95=t["p95"], test_mean=t["mean"], test_mean_above_floor=mean_big,
                           **{f"test_{r['current']}_median": r["median"] for r in rows[:-1]},
                           **{f"test_{r['current']}_p95": r["p95"] for r in rows[:-1]}))
        test_rows.append((case, rows))
    ch = pd.DataFrame(chosen)
    ch.to_csv(RES / "chosen.csv", index=False, float_format="%.6g")

    lines = ["# Part A: transistor models", "",
             "Split: 70 / 15 / 15 % (7,000 / 1,500 / 1,500 rows) of each 10,000-row dataset, fixed seed "
             "(`ml/data.split_index`). Errors are per-sample |ML − SPICE| / |SPICE| pooled over the four terminal "
             "currents; R² is on the transformed target (log-current, see `ml/data.py`), worst of the four currents.", "",
             "## Model comparison (validation set, 7,000 training rows)", "", *comparison_table(c), "",
             "### Per current (validation, median / p95)", "", *per_current_table(c), ""]
    if (RES / "data_size.csv").exists():
        raw = size_results()
        if "model" not in raw:
            raw["model"] = "mlp"
        raw = raw[raw.current == "all"]
        ens = raw[raw.model.str.startswith("mlp_ens")]
        ds = raw[raw.model == "mlp"].pivot(index="case", columns="n_train", values=["median", "p95"])
        sizes = sorted(ds.columns.get_level_values(1).unique())
        lines += ["## Data-size study (MLP, validation set)", "",
                  "Fewer than 7,000 rows = the first rows of the doc's (random) training split; more = that split plus "
                  "extra rows from `data/partA_extra/` (same distribution, other seeds). "
                  "Validation and test rows are always the doc's dataset.", "",
                  "| case | " + " | ".join(f"{n:,} rows: median / p95" for n in sizes) + " |",
                  "|---|" + "---|" * len(sizes)]
        for case in data.CASES:
            lines.append(f"| {case} | " + " | ".join(
                f"{ds.loc[case, ('median', n)]:.2f} / {ds.loc[case, ('p95', n)]:.2f} %" for n in sizes) + " |")
        lines.append("")
        ens = pd.concat([ens, c[c.model.str.startswith("blend") & (c.current == "all")]], ignore_index=True)
        if len(ens):
            combos = sorted(ens[["model", "n_train"]].drop_duplicates().itertuples(index=False, name=None),
                            key=lambda mn: (mn[0].startswith("blend"), mn[1], mn[0]))
            e = ens.set_index(["case", "model", "n_train"])
            lines += ["**Ensembles** (3 MLPs trained with different seeds, averaged in the log-like target domain) "
                      "and averages of the ensemble variants of the same size, validation median / p95:", "",
                      "| candidate | rows | " + " | ".join(data.CASES) + " |", "|---|---|" + "---|" * len(data.CASES)]
            for m, n in combos:
                lines.append(f"| {label(m)} | {n:,} | " + " | ".join(
                    f"{e.loc[(case, m, n), 'median']:.2f} / {e.loc[(case, m, n), 'p95']:.2f} %"
                    if (case, m, n) in e.index else "—" for case in data.CASES) + " |")
            lines.append("")
    lines += ["## Chosen models (test set, used only here)", "",
              "| case | model | training rows | val median / p95 | **test median** | **test p95** | test mean "
              "| test mean, currents ≥ 1e-11 A |", "|---|---|---|---|---|---|---|---|"]
    for r in ch.itertuples():
        lines.append(f"| {r.case} | {label(r.model)} | {r.n_train:,} | {r.val_median:.2f} / {r.val_p95:.2f} % | "
                     f"**{r.test_median:.2f} %** | **{r.test_p95:.2f} %** | {r.test_mean:.2f} % | "
                     f"{r.test_mean_above_floor:.2f} % |")
    lines += ["", "The last column leaves out currents below 1e-11 A (1,000× below the smallest gate leakage of Part B): "
              "the ~1e-14 A OFF body current and the few ON-device terminal currents that cross zero near Vd = Vs, "
              "where a relative error is large but the absolute error negligible.",
              "", "Per current on the test set (median / p95):", "",
              "| case | " + " | ".join(data.CURRENTS) + " |", "|---|" + "---|" * 4]
    for case, rows in test_rows:
        lines.append(f"| {case} | " + " | ".join(f"{r['median']:.2f} / {r['p95']:.1f} %" for r in rows[:-1]) + " |")
    lines += ["", "Saved models: `trained_models/<CASE>.joblib` (load with joblib; `.predict(df)` takes the doc's "
              "input columns and returns id, ig, is, ib in amperes).", ""]
    lines += [f"![{case}](pred_vs_spice_{case}.png)\n" for case in data.CASES]
    (REPORT / "MODELS.md").write_text("\n".join(lines) + "\n")
    print(ch[["case", "model", "n_train", "val_median", "val_p95", "test_median", "test_p95", "test_mean"]].to_string())


if __name__ == "__main__":
    main()
