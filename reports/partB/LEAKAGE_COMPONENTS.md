# Part B: what the gate leakage consists of

Leakage = VDD supply current + current drawn by the inputs held at VDD. Gate-tunnelling share = 1 − leakage with the model's gate tunnelling switched off (igcmod = igbmod = 0) / leakage of the full model; the rest is sub-threshold channel current plus junction current. Drawn by inputs at VDD = the part of the leakage that flows in through the inputs held at VDD rather than the VDD supply pin. Medians over 200 corners.

| gate | inputs | leakage (A) | gate-tunnelling share | sub-threshold + junction share | drawn by inputs at VDD |
|---|---|---|---|---|---|
| INV | 0 | 1.098e-07 | 65 % | 35 % | 0 % |
| INV | 1 | 1.129e-07 | 46 % | 54 % | 46 % |
| NAND2 | 00 | 1.008e-07 | 97 % | 3 % | 0 % |
| NAND2 | 01 | 1.579e-07 | 75 % | 25 % | 23 % |
| NAND2 | 10 | 9.068e-08 | 71 % | 29 % | 0 % |
| NAND2 | 11 | 2.256e-07 | 46 % | 54 % | 46 % |
| NOR2 | 00 | 2.189e-07 | 64 % | 36 % | 0 % |
| NOR2 | 01 | 1.668e-07 | 65 % | 35 % | 31 % |
| NOR2 | 10 | 9.702e-08 | 51 % | 49 % | 52 % |
| NOR2 | 11 | 8.966e-08 | 92 % | 8 % | 92 % |
| AND2 | 00 | 2.420e-07 | 73 % | 27 % | 0 % |
| AND2 | 01 | 2.823e-07 | 60 % | 40 % | 14 % |
| AND2 | 10 | 2.128e-07 | 55 % | 45 % | 0 % |
| AND2 | 11 | 3.466e-07 | 52 % | 48 % | 30 % |
| OR2 | 00 | 3.478e-07 | 55 % | 45 % | 0 % |
| OR2 | 01 | 2.875e-07 | 62 % | 38 % | 17 % |
| OR2 | 10 | 2.230e-07 | 54 % | 46 % | 21 % |
| OR2 | 11 | 2.299e-07 | 76 % | 24 % | 40 % |

All 3,600 simulations: gate tunnelling is 67 % of the leakage (median; 17–98 % for 90 % of runs).

