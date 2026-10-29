"""Static CMOS gates for Part B and the netlist/ML plumbing they share with
Part C (doc section 5).

Each run also records the input-pin currents (current drawn from each input
source, -i(v<input>)), which the leakage below needs.

Sizing: each corner draws one NMOS width Wn and one PMOS width Wp, uniform in
1..8 x Lmin (the trained range), shared by every NMOS / PMOS of the gate or
circuit at that corner, like the process draws. A corner without widths
falls back to Wn = 1 x Lmin, Wp = 2 x Lmin (the Fet defaults below).

A corner is one draw of VDD, temperature and process (doc section 5): all
NMOS share one process draw and all PMOS another. Inputs are exactly 0 or
VDD.

Leakage = the total current drawn from VDD-level sources: the VDD supply
current plus the current drawn by the inputs held at VDD (gate tunnelling into
the input transistors). This is the leakage power divided by VDD; an input at
0 V delivers no power, and its current already comes from VDD. With this
definition the leakage of a circuit is the sum of the leakages of its gates
(an internal input's current is drawn from its driver's VDD).
"""
import re
from dataclasses import dataclass

from .modelcard import load_cards
from .runner import _num, run_text
from .spec import L, MODEL_FILE, OPTIONS, PROCESS_PARAMS

WN, WP = 1 * L, 2 * L


@dataclass(frozen=True)
class Fet:
    name: str
    devtype: str     # nmos / pmos
    d: str
    g: str
    s: str
    b: str
    w: float


def inv(a, out, p=""):
    return [Fet(f"M{p}P", "pmos", out, a, "vdd", "vdd", WP),
            Fet(f"M{p}N", "nmos", out, a, "0", "0", WN)]


def nand2(a, b, out, x, p=""):
    return [Fet(f"M{p}PA", "pmos", out, a, "vdd", "vdd", WP),
            Fet(f"M{p}PB", "pmos", out, b, "vdd", "vdd", WP),
            Fet(f"M{p}NA", "nmos", out, a, x, "0", WN),      # top of the series stack
            Fet(f"M{p}NB", "nmos", x, b, "0", "0", WN)]


def nor2(a, b, out, y, p=""):
    return [Fet(f"M{p}PA", "pmos", y, a, "vdd", "vdd", WP),     # top of the series stack
            Fet(f"M{p}PB", "pmos", out, b, y, "vdd", WP),
            Fet(f"M{p}NA", "nmos", out, a, "0", "0", WN),
            Fet(f"M{p}NB", "nmos", out, b, "0", "0", WN)]


# name -> (inputs, internal nodes recorded, transistor list). AND2/OR2 are a
# NAND2/NOR2 driving an INV; "n" is the NAND2/NOR2 output (doc section 5).
GATES = {
    "INV":   (("a",), (), inv("a", "out")),
    "NAND2": (("a", "b"), ("x",), nand2("a", "b", "out", "x")),
    "NOR2":  (("a", "b"), ("y",), nor2("a", "b", "out", "y")),
    "AND2":  (("a", "b"), ("x", "n"), nand2("a", "b", "n", "x") + inv("n", "out", "I")),
    "OR2":   (("a", "b"), ("y", "n"), nor2("a", "b", "n", "y") + inv("n", "out", "I")),
}


def logic(gate, bits):
    a = bits[0]
    b = bits[1] if len(bits) > 1 else None
    return {"INV": lambda: 1 - a, "NAND2": lambda: 1 - (a & b), "NOR2": lambda: 1 - (a | b),
            "AND2": lambda: a & b, "OR2": lambda: a | b}[gate]()


def models_text(corner, cards):
    """The corner's NMOS (nch) and PMOS (pch) model cards."""
    n = cards["nmos"].with_overrides({p: corner[f"n_{p}"] for p in PROCESS_PARAMS}, name="nch")
    p = cards["pmos"].with_overrides({p: corner[f"p_{p}"] for p in PROCESS_PARAMS}, name="pch")
    return n.to_spice() + p.to_spice()


def width(f, corner):
    """The corner's Wn / Wp for this FET, or its default width."""
    key = "wn" if f.devtype == "nmos" else "wp"
    return corner[key] if corner is not None and key in corner else f.w


def fet_lines(fets, corner=None):
    model = {"nmos": "nch", "pmos": "pch"}
    return [f"{f.name} {f.d} {f.g} {f.s} {f.b} {model[f.devtype]} W={_num(width(f, corner))} L={_num(L)}"
            for f in fets]


def build(title, fets, inputs, input_values, probe_nodes, corner, cards):
    """Netlist of a transistor-level circuit at one corner and one input vector."""
    vdd = corner["vdd"]
    lines = [f"* {title}", OPTIONS, f".options temp={_num(corner['temp'])}",
             models_text(corner, cards), *fet_lines(fets, corner), f"Vdd vdd 0 {_num(vdd)}"]
    lines += [f"V{n} {n} 0 {_num(vdd * v)}" for n, v in zip(inputs, input_values)]
    probes = " ".join([f"i(v{n})" for n in inputs] + [f"v({n})" for n in probe_nodes])
    lines += [".control", "set numdgt=12", "op", f"print i(vdd) {probes}", ".endc", ".end", ""]
    return "\n".join(lines)


_VAL_RE = re.compile(r"^([iv]\([^)]+\))\s*=\s*([-+0-9.eE]+)", re.M)


def total_leakage(vdd_current, pins, bits):
    """VDD current + the current drawn by every input held at VDD."""
    return vdd_current + sum(p * b for p, b in zip(pins, bits))


def run(text):
    """Run a gate/circuit netlist; return ({'leak': VDD current in A, 'pin_<input>': A,
    <node>: V, ...}, error). Pin current = current drawn from the input source."""
    out, err = run_text(text)
    vals = {k: float(v) for k, v in _VAL_RE.findall(out)}
    if "i(vdd)" not in vals:
        return None, err or "no result"
    res = {"leak": -vals.pop("i(vdd)")}
    for k, v in vals.items():
        if k.startswith("i(v"):
            res[f"pin_{k[3:-1]}"] = -v
        else:
            res[k[2:-1]] = v
    return res, None


_CARDS = {}


def cards():
    if "c" not in _CARDS:
        _CARDS["c"] = load_cards(MODEL_FILE)
    return _CARDS["c"]
