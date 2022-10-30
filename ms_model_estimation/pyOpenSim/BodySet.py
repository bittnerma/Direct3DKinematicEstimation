from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Body:
    name: str
    mesh: [str]
    mass: float
    massCenter: float
    inertia: [float]
    scale: [float] = field(default_factory=lambda: [1, 1, 1])
    representation: Optional[int] = 3


class BodySet:

    def __init__(
            self,
            bodies: [Body]
    ):
        self.bodies = bodies
        self.bodiesDict = {}
        self.update_bodiesDict()

    def update_bodiesDict(self):
        self.bodiesDict = {}
        for b in self.bodies:
            self.bodiesDict[b.name] = b
