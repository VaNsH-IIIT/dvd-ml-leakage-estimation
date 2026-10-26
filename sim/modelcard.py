"""Read PTM/BSIM4 model cards and write them back with modified parameters.

The project may only use the professor-supplied model files (models/*.pm).
Process variation is applied by scaling parameters of those cards, never by
swapping in other models.
"""
import re
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_FILE = MODELS_DIR / "45nm_MGK.pm"  # team 22 assignment

_MODEL_RE = re.compile(r"^\.model\s+(\S+)\s+(\S+)\s+level\s*=\s*(\S+)", re.I)
_PARAM_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+0-9.eE]+)")


class ModelCard:
    """One `.model` statement: name, device type (nmos/pmos), level, parameters."""

    def __init__(self, name, devtype, level, params):
        self.name = name
        self.devtype = devtype.lower()
        self.level = level
        self.params = params  # dict, insertion-ordered as in the file

    def with_overrides(self, overrides, name=None):
        """Return a copy with some parameters replaced (absolute values)."""
        unknown = set(overrides) - set(self.params)
        if unknown:
            raise KeyError(f"parameters not in {self.name} card: {sorted(unknown)}")
        params = dict(self.params)
        params.update(overrides)
        return ModelCard(name or self.name, self.devtype, self.level, params)

    def to_spice(self):
        lines = [f".model {self.name} {self.devtype} level = {self.level}"]
        items = [f"{k} = {float(v)!r}" for k, v in self.params.items()]
        for i in range(0, len(items), 4):
            lines.append("+ " + "   ".join(items[i:i + 4]))
        return "\n".join(lines) + "\n"


def load_cards(path=DEFAULT_MODEL_FILE):
    """Parse a .pm file into {model name: ModelCard}."""
    cards, current = {}, None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        m = _MODEL_RE.match(line)
        if m:
            name, devtype, level = m.groups()
            current = ModelCard(name, devtype, level, {})
            cards[name] = current
        elif line.startswith("+") and current is not None:
            for key, val in _PARAM_RE.findall(line):
                current.params[key.lower()] = float(val)
    return cards
