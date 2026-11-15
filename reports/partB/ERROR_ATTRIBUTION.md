# Part B: which transistors carry the gate-leakage error

Each gate run's ML error is split exactly into one part per transistor: its ML currents minus its single-transistor SPICE currents at the same terminal voltages, pushed through the same cut-based combination as the leakage (`scripts/d5_error_attribution.py`). Parts are in % of the gate's SPICE leakage; shares are of the summed |part| over all 3,600 runs.
Check: the parts add up to the gate error (median |error| 0.37 %, mean 0.51 %).

## By transistor

| transistor | share of the error |
|---|---|
| PMOS ON | 29 % |
| PMOS OFF, full VDD across | 24 % |
| NMOS OFF, full VDD across | 24 % |
| NMOS ON | 22 % |
| NMOS OFF, small Vds (stacked) | 0 % |
| PMOS OFF, small Vds (stacked) | 0 % |

## By current

| current | share of the error |
|---|---|
| gate (ig) | 54 % |
| channel (id + is) | 46 % |
| body (ib) | 0 % |

## Transistor with the largest error, per gate case (share of the 200 corners)

| gate | inputs | most often largest | how often |
|---|---|---|---|
| AND2 | 00 | PMOS ON | 54 % |
| AND2 | 01 | PMOS OFF, full VDD across | 39 % |
| AND2 | 10 | PMOS OFF, full VDD across | 48 % |
| AND2 | 11 | PMOS OFF, full VDD across | 36 % |
| INV | 0 | PMOS ON | 56 % |
| INV | 1 | PMOS OFF, full VDD across | 66 % |
| NAND2 | 00 | PMOS ON | 88 % |
| NAND2 | 01 | PMOS ON | 43 % |
| NAND2 | 10 | PMOS ON | 64 % |
| NAND2 | 11 | PMOS OFF, full VDD across | 68 % |
| NOR2 | 00 | PMOS ON | 53 % |
| NOR2 | 01 | PMOS OFF, full VDD across | 40 % |
| NOR2 | 10 | PMOS OFF, full VDD across | 57 % |
| NOR2 | 11 | NMOS ON | 84 % |
| OR2 | 00 | PMOS OFF, full VDD across | 51 % |
| OR2 | 01 | PMOS ON | 48 % |
| OR2 | 10 | NMOS OFF, full VDD across | 32 % |
| OR2 | 11 | PMOS ON | 36 % |
