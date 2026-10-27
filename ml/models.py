"""The four model types compared in Part A, and the per-case model wrapper.

Every model type learns one regressor per canonical current (ihi, ig, ilo,
ib) on standardised features and standardised asinh-transformed targets
(ml/data.py). A CaseModel bundles the four regressors with the feature and
target transforms, so one saved file per case predicts (id, ig, is, ib)
straight from the doc's input columns.
"""
import numpy as np
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sim import spec

from . import data

MODEL_TYPES = ("poly", "lgbm", "mlp", "gp")
GP_SUBSET = 1000      # rows used to fit the GP kernel hyperparameters


class Ensemble:
    """Average of several regressors (in the transformed, log-like domain)."""

    def __init__(self, members):
        self.members = members

    def fit(self, x, y):
        self.members = [m.fit(x, y) for m in self.members]
        return self

    def predict(self, x):
        return np.mean([m.predict(x) for m in self.members], axis=0)


class LinearResidual:
    """A linear (ridge) fit on the standardised features plus a regressor trained on what the
    linear fit leaves. The linear part carries the main trends (log-current is close to linear in
    1/T, vth0, toxe, VDD, ...) to the corners of the input box, where training rows are sparse."""

    def __init__(self, base):
        self.base = base

    def fit(self, x, y):
        self.lin = Ridge(alpha=1e-3).fit(x, y)
        self.base = self.base.fit(x, y - self.lin.predict(x))
        return self

    def predict(self, x):
        return self.lin.predict(x) + self.base.predict(x)


def make(kind, seed=0):
    if kind.endswith("_lin"):   # e.g. mlp_ens3_lin: linear fit + 3-MLP ensemble on its residual
        return LinearResidual(make(kind.removesuffix("_lin"), seed))
    if kind.startswith("mlp_ens"):   # e.g. mlp_ens3: average of 3 MLPs with different seeds
        return Ensemble([make("mlp", seed=s) for s in range(int(kind[7:]))])
    if kind == "poly":    # degree-3 polynomial + ridge: transparent baseline
        return make_pipeline(PolynomialFeatures(3), Ridge(alpha=1e-4))
    if kind == "lgbm":    # gradient-boosted trees
        return LGBMRegressor(n_estimators=4000, learning_rate=0.03, num_leaves=63,
                             min_child_samples=10, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, verbose=-1, n_jobs=1, random_state=seed)
    if kind == "mlp":     # fully connected network
        return MLPRegressor(hidden_layer_sizes=(128, 128, 128), activation="tanh", alpha=1e-6,
                            learning_rate_init=2e-3, max_iter=4000, tol=1e-10,
                            n_iter_no_change=200, early_stopping=True, validation_fraction=0.1,
                            random_state=seed)
    if kind == "gp":      # Gaussian process, one length scale per input (ARD)
        return GPR()
    raise KeyError(kind)


class GPR:
    """Gaussian process regression: kernel fitted on a subset, then the
    posterior conditioned on all training rows with that kernel."""

    def fit(self, x, y):
        n_feat = x.shape[1]
        kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.ones(n_feat), (1e-2, 1e3))
                  + WhiteKernel(1e-6, (1e-10, 1e-1)))
        rng = np.random.default_rng(0)
        sub = rng.choice(len(x), min(GP_SUBSET, len(x)), replace=False)
        hyper = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=0,
                                         random_state=0).fit(x[sub], y[sub])
        self.gp = GaussianProcessRegressor(hyper.kernel_, normalize_y=True, optimizer=None).fit(x, y)
        return self

    def predict(self, x):
        return self.gp.predict(x)


class CaseModel:
    """Predicts (id, ig, is, ib) for one case from the doc's input columns.

    By default the regressors learn the current per unit W/L; a kind ending in
    "_cur" (e.g. mlp_ens3_cur) learns the current itself."""

    def __init__(self, case, kind, nominal):
        self.case, self.kind, self.nominal = case, kind, dict(nominal)

    def fit(self, df):
        x = data.features(df, self.nominal)
        self.per_width = not self.kind.endswith("_cur")
        c = self.densities(df)
        self.tf = data.Transform().fit(c)
        y = self.tf.forward(c)
        self.xs = StandardScaler().fit(x)
        self.ym, self.ysd = y.mean(axis=0), y.std(axis=0)
        xz, yz = self.xs.transform(x), (y - self.ym) / self.ysd
        # the four outputs train side by side (BLAS/LAPACK release the GIL),
        # except the GP: its kernel matrices take ~2 GB per output
        n_jobs = 1 if self.kind == "gp" else y.shape[1]
        self.regs = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(make(self.kind.removesuffix("_cur")).fit)(xz, yz[:, k]) for k in range(y.shape[1]))
        return self

    def _wl(self, df):
        return (df.w / spec.L).to_numpy()[:, None] if getattr(self, "per_width", False) else 1.0

    def densities(self, df):
        """Canonical currents divided by W/L: leakage scales almost linearly with width, so a wide
        device at a hot, low-vth0 corner is not an extreme input for the model."""
        return data.canonical(df) / self._wl(df)

    def targets(self, df):
        """The transformed targets the regressors learn (for R² on the transformed scale)."""
        return self.tf.forward(self.densities(df))

    def predict_transformed(self, df):
        xz = self.xs.transform(data.features(df, self.nominal))
        yz = np.column_stack([r.predict(xz) for r in self.regs])
        return yz * self.ysd + self.ym

    def predict(self, df):
        """(n x 4) array of id, ig, is, ib in amperes."""
        return data.uncanonical(df, self.tf.inverse(self.predict_transformed(df)) * self._wl(df))


class Blend:
    """Average of the currents predicted by several case models (e.g. the per-width and
    the plain-current 3-MLP ensembles of the same case)."""

    def __init__(self, members):
        self.members = members
        self.case = members[0].case

    def predict(self, df):
        return np.mean([m.predict(df) for m in self.members], axis=0)
