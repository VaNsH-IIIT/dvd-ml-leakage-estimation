"""ML leakage of a transistor-level circuit from the four transistor models
(doc Part B step 4 and Part C step 7).

For each transistor: take its terminal voltages from the node voltages
(SPICE-recorded internal nodes and output; supply, ground and inputs are
exact), decide its case from the gate voltage (ON / OFF), and predict its
four terminal currents with that case's model. Terminal currents are
positive out of the device, as in the Part A data.

How the currents are summed. This gives the current the VDD source
delivers (the inputs' currents are added by the caller, scripts/b2_gates_ml.py). Summing the device currents at the vdd node directly ("vdd-node
sum") needs the channel current of ON pull-up devices, which sit at a
drain-source voltage of microvolts: there the channel current is a tiny
difference set by the OFF network, and the uniformly sampled training data
(Vd, Vs in 0..VDD) has almost no rows. Instead we apply Kirchhoff's law to
the supernode S = vdd plus every node joined to it through ON channels
("supernode sum"):

    I_leak = - sum of device-terminal currents at nodes in S

Inside S, each device with both drain and source in S contributes
id + is = -(ig + ib) by its own KCL, so its channel current cancels exactly
and is replaced by its gate and body currents. What remains are OFF devices
at large drain-source voltage plus gate and body currents, all well inside
the training data. With exact device currents both sums equal the SPICE
leakage (scripts/b2_gates_ml.py checks this with device-level SPICE).

Choice of cut. KCL holds for any node set that contains vdd but not ground
or an input, so S may also include further internal nodes. Among those sets
("best-cut"), we use the one whose cut devices (exactly one of drain/source
in S) are all OFF and have the largest smallest |Vds|: there the channel
current dominates the gate-tunnelling current it would otherwise cancel
against. Example: NOR2 inputs 11 cuts through the lower PMOS (|Vds| ~ 0.8 V)
instead of the upper one (|Vds| ~ 0.07 V).
"""
from itertools import combinations

import numpy as np
import pandas as pd

from sim.spec import PROCESS_PARAMS

TERMINALS = ("d", "g", "s", "b")


def device_on(devtype, vg, vdd):
    return (vg > 0.5 * vdd) if devtype == "nmos" else (vg < 0.5 * vdd)


def supernode(fets, on):
    """Nodes joined to vdd through the channels of ON devices."""
    s, grew = {"vdd"}, True
    while grew:
        grew = False
        for f, is_on in zip(fets, on):
            if is_on and ((f.d in s) != (f.s in s)):
                s |= {f.d, f.s}
                grew = True
    return s


RAILS = {"vdd", "0"}


def best_cut(fets, on, vmed):
    """Node set S (see module docstring) maximising the smallest |Vds| of the
    devices it cuts; `vmed` maps node -> typical voltage for this ON pattern."""
    base = supernode(fets, on)
    channel = {f.d for f in fets} | {f.s for f in fets}
    gates_in = {f.g for f in fets} - channel          # primary inputs
    free = sorted(channel - base - RAILS - gates_in)
    best, best_score = base, -1.0
    for k in range(len(free) + 1):
        for extra in combinations(free, k):
            s = base | set(extra)
            cut = [(f, o) for f, o in zip(fets, on) if (f.d in s) != (f.s in s)]
            if not cut or any(o for _, o in cut):
                continue
            score = min(abs(vmed[f.d] - vmed[f.s]) for f, _ in cut)
            if score > best_score + 1e-9:
                best, best_score = s, score
    return best


def combine(fets, on, currents, method="supernode", vmed=None):
    """Leakage from per-device terminal currents (n x 4 arrays, order d, g, s, b)
    for rows that share one ON/OFF pattern."""
    n = len(next(iter(currents.values())))
    leak = np.zeros(n)
    s = {"vdd": lambda: {"vdd"}, "supernode": lambda: supernode(fets, on),
         "best-cut": lambda: best_cut(fets, on, vmed)}[method]()
    for f in fets:
        cur = currents[f.name]
        if method != "vdd" and f.d in s and f.s in s:
            # channel cancels: id + is = -(ig + ib)
            leak += cur[:, 1] * (f.g not in s) + cur[:, 3] * (f.b not in s)
            continue
        for k, t in enumerate(TERMINALS):
            if getattr(f, t) in s:
                leak -= cur[:, k]
    return leak


def device_inputs(f, nodes, corners):
    """The doc's input columns for one transistor across all rows."""
    n = len(nodes)
    df = pd.DataFrame({
        "vdd": corners["vdd"].to_numpy(), "temp": corners["temp"].to_numpy(),
        "w": corners["wn" if f.devtype == "nmos" else "wp"].to_numpy() if "wn" in corners else np.full(n, f.w),
        "vd": nodes[f.d].to_numpy(), "vs": nodes[f.s].to_numpy(),
        "vg": nodes[f.g].to_numpy(), "vb": nodes[f.b].to_numpy(),
    })
    for p in PROCESS_PARAMS:
        df[p] = corners[f"{f.devtype[0]}_{p}"].to_numpy()
    return df


def predict_currents(fets, nodes, corners, models):
    """{fet name: n x 4 currents (id, ig, is, ib)} and the n x n_fets ON matrix."""
    cur, on = {}, []
    for f in fets:
        df = device_inputs(f, nodes, corners)
        is_on = device_on(f.devtype, df.vg.to_numpy(), df.vdd.to_numpy())
        out = np.zeros((len(df), 4))
        for flag in (True, False):
            sel = is_on == flag
            if sel.any():
                case = f"{f.devtype.upper()}_{'ON' if flag else 'OFF'}"
                out[sel] = models[case].predict(df[sel].reset_index(drop=True))
        cur[f.name] = out
        on.append(is_on)
    return cur, np.column_stack(on)


def leakage(fets, currents, on, method="best-cut", nodes=None):
    """Combine per-device currents into the leakage, row by row ON pattern.
    `nodes` (node voltages per row) is needed for method="best-cut"."""
    n = on.shape[0]
    leak = np.zeros(n)
    patterns = {tuple(r) for r in on}
    for pat in patterns:
        sel = (on == np.array(pat)).all(axis=1)
        vmed = nodes[sel].median().to_dict() if nodes is not None else None
        leak[sel] = combine(fets, pat, {k: v[sel] for k, v in currents.items()}, method, vmed)
    return leak


def predict_leakage(fets, nodes, corners, models, method="best-cut"):
    cur, on = predict_currents(fets, nodes, corners, models)
    return leakage(fets, cur, on, method, nodes), cur, on


def solve_internal_nodes(fets, nodes, corners, models, unknown, iters=30, sweeps=2):
    """ML estimate of internal node voltages: for each node in `unknown`,
    find V in [0, VDD] where the ML device currents into it sum to zero
    (vectorised bisection; several Gauss-Seidel sweeps when there are two
    coupled nodes). Everything else (inputs, output) stays as given.
    Returns a copy of `nodes` with the solved voltages."""
    nodes = nodes.copy()
    vdd = corners["vdd"].to_numpy()

    def residual(node):
        """Sum of device-terminal currents flowing out of devices into `node`."""
        cur, _ = predict_currents([f for f in fets if node in (f.d, f.g, f.s, f.b)], nodes, corners, models)
        total = np.zeros(len(nodes))
        for f in fets:
            if f.name in cur:
                for k, t in enumerate(TERMINALS):
                    if getattr(f, t) == node:
                        total += cur[f.name][:, k]
        return total

    for _ in range(sweeps if len(unknown) > 1 else 1):
        for node in unknown:
            lo, hi = np.zeros(len(nodes)), vdd.copy()
            nodes[node] = lo
            r_lo = residual(node)
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                nodes[node] = mid
                r = residual(node)
                same = np.sign(r) == np.sign(r_lo)
                lo, r_lo = np.where(same, mid, lo), np.where(same, r, r_lo)
                hi = np.where(same, hi, mid)
            nodes[node] = 0.5 * (lo + hi)
    return nodes
