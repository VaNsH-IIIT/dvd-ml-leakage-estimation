"""Part A, step 2a: train the four model types for each case and compare them
on the validation set (deliverable 4).

Split: 70/15/15 with a fixed seed (ml/data.split_index). Every fitted model
is cached in runs/partA/ so a4_select_models.py can reuse it.

Writes data/partA_results/compare_val.csv (one row per case x model x current).
Run:  python scripts/a3_compare_models.py [--kinds poly lgbm mlp gp] [--cases ...]
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")   # one thread per fit; fits run side by side

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml import data, metrics  # noqa: E402
from ml.models import MODEL_TYPES, CaseModel  # noqa: E402
from sim import spec  # noqa: E402
from sim.bias import devtype  # noqa: E402
from sim.modelcard import load_cards  # noqa: E402

RUNS = ROOT / "runs" / "partA"
OUT = ROOT / "data" / "partA_results"


def evaluate(model, df):
    """Rows of metrics on df: one per current plus 'all' (pooled over the four)."""
    pred = model.predict(df)
    true = df[list(data.CURRENTS)].to_numpy()
    err = metrics.rel_err(pred, true)
    rows = []
    for k, name in enumerate(data.CURRENTS):
        rows.append(dict(current=name, **metrics.summary(err[:, k])))
    rows.append(dict(current="all", **metrics.summary(err)))
    # R² of the regressors on their transformed targets (a Blend has no single target scale)
    r2_min = float(min(metrics.r2(model.targets(df), model.predict_transformed(df)))) if hasattr(model, "targets") else np.nan
    for r in rows:
        r["r2_min"] = r2_min
    return rows


def fit_one(case, kind):
    import warnings as w
    w.filterwarnings("ignore")
    cards = load_cards(spec.MODEL_FILE)
    df = data.load(case)
    idx = data.split_index(len(df))
    t0 = time.time()
    model = CaseModel(case, kind, cards[devtype(case)].params).fit(df.iloc[idx["train"]])
    fit_s = time.time() - t0
    RUNS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, RUNS / f"{case}_{kind}.joblib")
    rows = evaluate(model, df.iloc[idx["val"]])
    allrow = rows[-1]
    print(f"{case:9s} {kind:5s} val: median {allrow['median']:.3f} %  p95 {allrow['p95']:.2f} %  "
          f"mean {allrow['mean']:.2f} %  R2min {allrow['r2_min']:.5f}  ({fit_s:.0f} s)", flush=True)
    return [dict(case=case, model=kind, fit_s=fit_s, **r) for r in rows]


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=list(MODEL_TYPES))
    ap.add_argument("--cases", nargs="+", default=list(data.CASES))
    args = ap.parse_args()
    # Light models side by side; GP fits at most two at a time (memory: ~2 GB
    # each; four in parallel, with outputs in parallel too, exhausted the 23 GB
    # machine on 2026-10-25).
    light = [(c, k) for k in args.kinds if k != "gp" for c in args.cases]
    heavy = [(c, k) for k in args.kinds if k == "gp" for c in args.cases]
    res = joblib.Parallel(n_jobs=min(4, max(1, len(light))))(joblib.delayed(fit_one)(*j) for j in light)
    res += joblib.Parallel(n_jobs=2)(joblib.delayed(fit_one)(*j) for j in heavy) if heavy else []
    new = pd.DataFrame([r for rows in res for r in rows])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "compare_val.csv"
    if path.exists():   # keep results of model types not rerun
        old = pd.read_csv(path)
        old = old[~old.set_index(["case", "model"]).index.isin(new.set_index(["case", "model"]).index)]
        new = pd.concat([old, new])
    new.sort_values(["case", "model", "current"]).to_csv(path, index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
