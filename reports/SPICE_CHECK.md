# SPICE reference check: unique operating points

Each run solved again from every non-rail node at 0 V and at VDD (`scripts/d4_op_check.py`).

| set | runs | restarts that failed | largest difference from the reference | runs differing by > 0.01 % |
|---|---|---|---|---|
| gates | 3,600 | 0 | 9.8e-05 % | 0 |
| parity checker | 597 | 0 | 4.8e-06 % | 0 |

Parity checker: the 300 runs with the largest ML error plus 300 random runs (overlaps removed).
