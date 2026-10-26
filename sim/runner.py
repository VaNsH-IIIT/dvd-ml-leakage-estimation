"""Build single-device netlists and run them through ngspice.

Each terminal has its own voltage source (doc section 4). Sign convention:
`id`, `ig`, `is`, `ib` are ngspice `i(Vd)` ... `i(Vb)`, i.e. the current
through each terminal's source from its + node to ground. Positive therefore
means current flowing OUT of the device terminal (same as the professor's
sample netlist 22n_NMOS_ON.net). Kirchhoff: id + ig + is + ib = 0.
"""
import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

from .modelcard import load_cards
from .spec import MODEL_FILE, OPTIONS, PROCESS_PARAMS

NGSPICE = os.environ.get("NGSPICE", "/usr/bin/ngspice")
CURRENT_NAMES = ("id", "ig", "is", "ib")
_PRINT_RE = re.compile(r"^i\(v([dgsb])\)\s*=\s*([-+0-9.eE]+)", re.M)


@dataclass
class Sample:
    devtype: str           # "nmos" or "pmos"
    vd: float              # terminal voltages, V
    vg: float
    vs: float
    vb: float
    temp: float            # degC
    w: float               # m
    l: float               # m
    process: dict = field(default_factory=dict)  # {param: absolute value}


def model_for(devtype, process, cards, name="dut"):
    """Model card of `devtype` with process parameters replaced (absolute values)."""
    unknown = set(process) - set(PROCESS_PARAMS)
    if unknown:
        raise KeyError(f"unsupported process parameters {sorted(unknown)}")
    return cards[devtype].with_overrides(process, name=name)


def _num(x):
    """Plain decimal text for SPICE (numpy scalars would print as np.float64(...))."""
    return repr(float(x))


def build_netlist(s, cards, options=OPTIONS, title="single-device leakage sample"):
    return "\n".join([
        f"* {s.devtype.upper()} {title}",
        options,
        f".options temp={_num(s.temp)}",
        model_for(s.devtype, s.process, cards).to_spice(),
        f"M1 d g s b dut W={_num(s.w)} L={_num(s.l)}",
        f"Vd d 0 {_num(s.vd)}",
        f"Vg g 0 {_num(s.vg)}",
        f"Vs s 0 {_num(s.vs)}",
        f"Vb b 0 {_num(s.vb)}",
        ".control",
        "op",
        "print i(vd) i(vg) i(vs) i(vb)",
        ".endc",
        ".end",
        "",
    ])


def run_text(text, timeout=120):
    """Run a netlist in ngspice batch mode; return (stdout, error-or-None)."""
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        proc = subprocess.run([NGSPICE, "-b", path], capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "", "timeout"
    finally:
        os.unlink(path)
    return proc.stdout, (None if proc.returncode == 0 else proc.stderr.strip()[-300:])


def run_netlist(text, timeout=60):
    """Run one single-device netlist; return ({id, ig, is, ib}, error-or-None)."""
    out, err = run_text(text, timeout)
    found = dict(_PRINT_RE.findall(out))
    if len(found) != 4:
        tail = out.strip().splitlines()[-3:]
        return None, err or ("no result: " + " | ".join(tail))
    return {f"i{t}": float(val) for t, val in found.items()}, None


_CARDS = {}


def _cards(model_file):
    if model_file not in _CARDS:
        _CARDS[model_file] = load_cards(model_file)
    return _CARDS[model_file]


def _simulate(args):
    sample, model_file, options = args
    return run_netlist(build_netlist(sample, _cards(model_file), options))


def simulate(samples, model_file=MODEL_FILE, workers=None, options=OPTIONS):
    """Simulate many samples in parallel; results come back in input order."""
    workers = workers or os.cpu_count()
    jobs = [(s, str(model_file), options) for s in samples]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_simulate, jobs, chunksize=16))


def parallel_map(fn, items, workers=None, chunksize=1):
    """Run fn over items in worker processes (used by gate/circuit drivers)."""
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        return list(pool.map(fn, items, chunksize=chunksize))
