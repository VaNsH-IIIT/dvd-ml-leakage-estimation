"""Validation study for the target scale of the mixed-sign OFF terminal.

The OFF-state terminal whose channel current and gate-tunnelling current can
cancel (lo for NMOS OFF, hi for PMOS OFF) is learned as asinh(I / I0). This
compares two choices of I0 with the same MLP and the doc's 7,000 training
rows, scored on the validation split (Part A data only):
  constant : I0 = 1e-3 x median |I| of the training set
  gate     : I0 = 0.1 x |ig| of the same sample (used in ml/data.py)

Writes data/partA_results/mixed_scale_study.csv.
Run:  python scripts/a3c_mixed_scale_study.py
"""
import os
import sys
import warnings
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml import data, metrics  # noqa: E402
from ml.data import MIXED_SCALE  # noqa: E402
from ml.models import make  # noqa: E402
from sim import spec  # noqa: E402
from sim.modelcard import load_cards  # noqa: E402

MIXED = {"NMOS_OFF": data.CANON.index("ilo"), "PMOS_OFF": data.CANON.index("ihi")}
IG = data.CANON.index("ig")


def run(case, variant):
    warnings.filterwarnings("ignore")
    df = data.load(case)
    idx = data.split_index(len(df))
    tr, va = df.iloc[idx["train"]], df.iloc[idx["val"]]
    nom = load_cards(spec.MODEL_FILE)[case[:4].lower()].params
    xs = StandardScaler().fit(data.features(tr, nom))
    ctr, cva = data.canonical(tr), data.canonical(va)
    k = MIXED[case]
    if variant == "constant":
        i0 = 1e-3 * np.median(np.abs(ctr[:, k]))
        s_tr, s_va = i0, i0
    else:   # gate: the sample's own gate current (true on train; at validation the model sees no truth,
        # so use the gate-current MLP prediction, as the deployed model does)
        ig_model = make("mlp").fit(xs.transform(data.features(tr, nom)), np.log(np.abs(ctr[:, IG])))
        s_tr = MIXED_SCALE * np.abs(ctr[:, IG])
        s_va = MIXED_SCALE * np.exp(ig_model.predict(xs.transform(data.features(va, nom))))
    y = np.arcsinh(ctr[:, k] / s_tr)
    m, sd = y.mean(), y.std()
    reg = make("mlp").fit(xs.transform(data.features(tr, nom)), (y - m) / sd)
    pred = np.sinh(reg.predict(xs.transform(data.features(va, nom))) * sd + m) * s_va
    e = metrics.rel_err(pred, cva[:, k])
    print(f"{case} {variant:8s} median {np.median(e):.2f} %  p95 {np.percentile(e, 95):.2f} %", flush=True)
    return dict(case=case, terminal=data.CANON[k], scale=variant, **metrics.summary(e))


def main():
    jobs = [(c, v) for c in MIXED for v in ("constant", "gate")]
    res = joblib.Parallel(n_jobs=4)(joblib.delayed(run)(*j) for j in jobs)
    pd.DataFrame(res).to_csv(ROOT / "data" / "partA_results" / "mixed_scale_study.csv", index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
