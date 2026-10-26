"""Team 22 assignment and the sampling spec from the official project doc.

Source: DVD_ML_Project.pdf (sections 2 and 4) and the TA team sheet.
Every range here is copied from the doc; nothing is tuned by us.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Team assignment (TA sheet, team 22)
MODEL_FILE = ROOT / "models" / "45nm_MGK.pm"
VDD_NOM = 0.9            # V
L = 45e-9                # m, feature size of the node
TEST_CIRCUIT = "8-bit parity checker"

# Section 4: every parameter drawn independently and uniformly
VDD_RANGE = (0.9 * VDD_NOM, 1.1 * VDD_NOM)   # drawn first; Vd, Vs in 0..VDD
W_RANGE = (1 * L, 8 * L)
TEMP_RANGE = (-40.0, 125.0)                   # degC
PROCESS_PARAMS = ("vth0", "toxe", "ndep", "u0", "xj")
PROCESS_SPREAD = 0.10                         # nominal +/-10 %, separate draws for NMOS and PMOS

# Section 4: gate and body voltage per case, as multiples of VDD
CASES = {
    #            devtype  vg  vb
    "NMOS_ON":  ("nmos", 1, 0),
    "NMOS_OFF": ("nmos", 0, 0),
    "PMOS_ON":  ("pmos", 0, 1),
    "PMOS_OFF": ("pmos", 1, 1),
}

# Section 3, step 2: 70/15/15 random split with a fixed seed
SPLIT = (0.70, 0.15, 0.15)
SPLIT_SEED = 22

# Section 4 netlist requirement (transistor, gate and test-circuit netlists)
OPTIONS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"
