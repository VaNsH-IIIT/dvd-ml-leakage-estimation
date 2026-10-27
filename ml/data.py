"""Features, targets and the train/validation/test split for the Part A models.

Drain/source symmetry. A planar MOSFET is symmetric: swapping the drain and
source voltages swaps id and is exactly (checked in scripts/verify_sim.py).
So each sample is rewritten in terms of the higher-voltage terminal ("hi")
and the lower-voltage one ("lo"). The channel current then always flows the
same way, and three of the four terminal currents of every case keep one sign
over the whole dataset (reports/partA/DATA_CHECK.md). After prediction the
hi/lo currents are mapped back to id/is.

Per-unit-width targets. The models learn current per unit W/L
(ml/models.CaseModel): leakage scales almost linearly with width, so a wide
device at a hot, low-vth0 corner is not an extreme input for the model.

Target transform. Currents span 7-10 decades and one terminal per OFF case
(lo for NMOS, hi for PMOS) changes sign where the outgoing channel current
and the incoming gate-tunnelling current cancel. Each current is therefore
learned as asinh(I / I0): equal to sign(I) * ln(2|I|/I0) (i.e. log-current)
wherever |I| >> I0, and smooth through zero (see Transform for I0).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from sim import spec

DATA = Path(__file__).resolve().parent.parent / "data" / "partA"
CASES = tuple(spec.CASES)
CURRENTS = ("id", "ig", "is", "ib")            # doc's output columns
CANON = ("ihi", "ig", "ilo", "ib")             # what the models learn
FEATURES = ("temp", "inv_temp", "vdd", "w", "log_w", "vhi", "vlo", "vds", "log_vds",
            "vg_hi", "vg_lo", "vb_lo", *(f"{p}_s" for p in spec.PROCESS_PARAMS))


def load(case):
    return pd.read_csv(DATA / f"{case}.csv")


def split_index(n, seed=spec.SPLIT_SEED, fractions=spec.SPLIT):
    """Random 70/15/15 split with a fixed seed: {'train','val','test'} -> row indices."""
    perm = np.random.default_rng(seed).permutation(n)
    a = int(round(fractions[0] * n))
    b = a + int(round(fractions[1] * n))
    return {"train": perm[:a], "val": perm[a:b], "test": perm[b:]}


def features(df, nominal):
    """Model inputs from the doc's columns. `nominal`: model-card params of the device."""
    vhi = np.maximum(df.vd, df.vs)
    vlo = np.minimum(df.vd, df.vs)
    vds = vhi - vlo
    x = pd.DataFrame({
        "temp": df.temp,
        "inv_temp": 1.0 / (df.temp + 273.15),
        "vdd": df.vdd,
        "w": df.w / spec.L,
        "log_w": np.log(df.w / spec.L),
        "vhi": vhi, "vlo": vlo, "vds": vds,
        "log_vds": np.log10(vds + 1e-7),
        "vg_hi": df.vg - vhi, "vg_lo": df.vg - vlo, "vb_lo": df.vb - vlo,
    })
    for p in spec.PROCESS_PARAMS:
        x[f"{p}_s"] = df[p] / nominal[p] - 1.0
    return x[list(FEATURES)].to_numpy(float)


def canonical(df):
    """Terminal currents in hi/lo form (n x 4, order CANON)."""
    d_hi = (df.vd >= df.vs).to_numpy()
    ihi = np.where(d_hi, df["id"], df["is"])
    ilo = np.where(d_hi, df["is"], df["id"])
    return np.column_stack([ihi, df.ig, ilo, df.ib])


def uncanonical(df, c):
    """Map hi/lo predictions back to the doc's (id, ig, is, ib)."""
    d_hi = (df.vd >= df.vs).to_numpy()
    i_d = np.where(d_hi, c[:, 0], c[:, 2])
    i_s = np.where(d_hi, c[:, 2], c[:, 0])
    return np.column_stack([i_d, c[:, 1], i_s, c[:, 3]])


MIXED_SCALE = 0.1   # c in asinh(I / (c |ig|)) for mixed-sign currents


class Transform:
    """asinh(I / I0) per output.

    Fixed-sign currents: I0 = 1e-3 * median |I| of the training set, so the
    transform is log-current. Mixed-sign currents (the OFF-state terminal
    where channel current and gate tunnelling cancel, detected as a current
    with both signs in more than 1 % of training rows): I0 = c * |ig| of the
    same sample. The current it cancels against sets the scale, so the zero
    crossing becomes a gentle slope instead of a near-step. At prediction
    time ig is the model's own (fixed-sign) gate-current prediction.
    """

    IG = CANON.index("ig")

    def fit(self, c):
        self.i0 = 1e-3 * np.median(np.abs(c), axis=0)
        pos = (c > 0).mean(axis=0)
        self.mixed = (pos > 0.01) & (pos < 0.99)
        self.mixed[self.IG] = False
        return self

    def scale(self, ig):
        s = np.tile(self.i0, (len(ig), 1))
        # floor at the solver's abstol (1e-18 A): ig is exactly 0 when all four
        # terminals sit at the same voltage
        s[:, self.mixed] = np.maximum(MIXED_SCALE * np.abs(ig), 1e-18)[:, None]
        return s

    def forward(self, c):
        return np.arcsinh(c / self.scale(c[:, self.IG]))

    def inverse(self, y):
        ig = np.sinh(y[:, self.IG]) * self.i0[self.IG]
        return np.sinh(y) * self.scale(ig)
