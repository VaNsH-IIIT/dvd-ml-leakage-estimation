"""Part A, data-size study: does more training data improve the MLP?

The doc asks for high accuracy "preferably with the smallest dataset
possible". The MLP is trained on subsets of the doc's 7,000-row training
split (--subsample), on the whole split, then with extra 10,000-row blocks (--blocks; data/partA_extra/, same
distribution, other seeds); --kind trains other model kinds, e.g. the 3-MLP ensemble variants mlp_ens3,
mlp_ens3_cur and mlp_ens3_lin (ml/models.py), and always scored on the same 1,500-row validation split of
the doc's dataset. Fitted models are cached in runs/partA/ as
<CASE>_mlp_n<rows>.joblib.

Writes data/partA_results/data_size.csv.
Run:  python scripts/a3b_data_size.py
"""
import os
import sys
import time
import warnings
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml import data  # noqa: E402
from ml.models import CaseModel  # noqa: E402
from sim import spec  # noqa: E402
from sim.bias import devtype  # noqa: E402
from sim.modelcard import load_cards  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from a3_compare_models import evaluate  # noqa: E402

EXTRA_BLOCKS = (0, 2, 4)


def training_set(case, blocks, edge=0, subsample=None):
    df = data.load(case)
    train_idx = data.split_index(len(df))["train"]
    if subsample:   # the first rows of the (already random) training permutation
        train_idx = train_idx[:subsample]
    parts = [df.iloc[train_idx]]
    parts += [pd.read_csv(ROOT / "data" / "partA_extra" / f"{case}_{k}.csv") for k in range(1, blocks + 1)]
    parts += [pd.read_csv(ROOT / "data" / "partA_edge" / f"{case}_{k}.csv") for k in range(1, edge + 1)]
    return pd.concat(parts, ignore_index=True), df.iloc[data.split_index(len(df))["val"]]


def fit_one(case, blocks, kind="mlp", edge=0, subsample=None):
    warnings.filterwarnings("ignore")
    train, val = training_set(case, blocks, edge, subsample)
    if edge:
        kind_tag = f"{kind}_edge{edge}"
    else:
        kind_tag = kind
    t0 = time.time()
    nominal = load_cards(spec.MODEL_FILE)[devtype(case)].params
    model = CaseModel(case, kind, nominal).fit(train)
    fit_s = time.time() - t0
    joblib.dump(model, ROOT / "runs" / "partA" / f"{case}_{kind_tag}_n{len(train)}.joblib")
    rows = evaluate(model, val)
    a = rows[-1]
    print(f"{case:9s} {kind_tag} n_train={len(train):6d}  val median {a['median']:.3f} %  p95 {a['p95']:.2f} %  "
          f"mean {a['mean']:.2f} %  ({fit_s:.0f} s)", flush=True)
    return [dict(case=case, model=kind_tag, n_train=len(train), fit_s=fit_s, **r) for r in rows]


def main():
    warnings.filterwarnings("ignore")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="mlp")
    ap.add_argument("--blocks", nargs="*", type=int, default=list(EXTRA_BLOCKS))
    ap.add_argument("--edge", type=int, default=0, help="add N edge blocks (data/partA_edge/)")
    ap.add_argument("--out", default="data_size.csv")
    ap.add_argument("--subsample", nargs="+", type=int, default=[],
                    help="also train on the first N rows of the doc's training split")
    args = ap.parse_args()
    jobs = [(c, b, args.kind, args.edge) for b in args.blocks for c in data.CASES]
    jobs += [(c, 0, args.kind, args.edge, n) for n in args.subsample for c in data.CASES]
    res = joblib.Parallel(n_jobs=4)(joblib.delayed(fit_one)(*j) for j in jobs)
    out = ROOT / "data" / "partA_results"
    out.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([r for rows in res for r in rows])
    path = out / args.out
    if path.exists():   # keep other (model, size) results
        old = pd.read_csv(path)
        if "model" not in old:
            old["model"] = "mlp"
        key = ["case", "model", "n_train"]
        old = old[~old.set_index(key).index.isin(new.set_index(key).index)]
        new = pd.concat([old, new])
    new.to_csv(path, index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
