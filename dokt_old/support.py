from typing import NamedTuple


class MultiWalk(NamedTuple):
    probs: float = {}
    stepSizes: float = {}
    positionsNormal: int = {}
    positionsTurban: float = {}
    positionMe: int = {}
    steps: int = None
