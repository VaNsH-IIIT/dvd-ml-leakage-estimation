"""Terminal voltages of a single device for each of the four cases.

The gate and body are fixed by the case (spec.CASES); drain and source are
free inputs drawn from 0..VDD.
"""
from .spec import CASES


def devtype(case):
    return CASES[case][0]


def terminal_voltages(case, vdd, vd, vs):
    """Return (vd, vg, vs, vb) for a case at supply vdd."""
    _, g, b = CASES[case]
    return vd, g * vdd, vs, b * vdd
