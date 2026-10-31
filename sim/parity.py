"""8-bit parity checker (team 22 test circuit) built from the Part B gates.

XOR is not one of the five gates, so each XOR2 is the standard four-NAND2
circuit:  n1 = NAND(a, b), n2 = NAND(a, n1), n3 = NAND(b, n1),
out = NAND(n2, n3). Seven XOR2s form a balanced tree over inputs i0..i7, so
the output p is 1 when an odd number of inputs are 1. 28 NAND2 = 112 FETs.
"""
from .gates import nand2

INPUTS = tuple(f"i{k}" for k in range(8))


def xor_tree():
    """List of (gate name, input a net, input b net, output net) for all NAND2s."""
    nands = []
    level, nets, k = 0, list(INPUTS), 0
    while len(nets) > 1:
        nxt = []
        for a, b in zip(nets[0::2], nets[1::2]):
            out = "p" if len(nets) == 2 else f"x{level}_{len(nxt)}"
            t = f"X{k}_"
            nands += [(f"{t}N1", a, b, f"{t}n1"), (f"{t}N2", a, f"{t}n1", f"{t}n2"),
                      (f"{t}N3", b, f"{t}n1", f"{t}n3"), (f"{t}N4", f"{t}n2", f"{t}n3", out)]
            nxt.append(out)
            k += 1
        nets, level = nxt, level + 1
    return nands


NANDS = xor_tree()


def fets():
    out = []
    for name, a, b, y in NANDS:
        out += nand2(a, b, y, f"{name}_x", p=f"{name}_")
    return out


def net_values(bits):
    """Logic value of every net for an input vector (tuple of 8 bits)."""
    v = dict(zip(INPUTS, bits))
    for _, a, b, y in NANDS:
        v[y] = 1 - (v[a] & v[b])
    return v


def nand_inputs(bits):
    """Per NAND2: its input pattern string 'ab' for this input vector."""
    v = net_values(bits)
    return [f"{v[a]}{v[b]}" for _, a, b, _ in NANDS]
