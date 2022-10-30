from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class Frame:
    parentFrame: str
    parentLoc: [float]
    parentOrientation: [float]
    childFrame: str
    childLoc: [float]
    childOrientation: [float]


@dataclass
class Coordinate:
    name: str
    coordinateType: int
    defaultValue: float
    minValue: float
    maxValue: float
    locked: bool = False


@dataclass
class SpatialTransform:
    transformName: str
    axis: [float, float, float]
    coordinateName: Optional[str] = ""
    funcType: str = "Constant"
    funcParameters: Any = field(default_factory=lambda: [0, 1])


@dataclass
class Joint:
    name: str
    jointType: str
    frames: Frame
    coordinates: [Coordinate] = None
    spatialTransform: [SpatialTransform] = None
    otherProperties: dict = None


class JointSet:

    def __init__(
            self,
            joints: [Joint],
    ):
        self.joints = joints
        self.jointsDict = {}
        self.coordinatesDict = {}
        self.update_jointsDict()
        self.update_coordinatesDict()

    def update_jointsDict(self):
        self.jointsDict = {}
        for j in self.joints:
            self.jointsDict[j.name] = j

    def update_coordinatesDict(self):
        self.coordinatesDict = {}
        for j in self.joints:
            if j.coordinates:
                for k, c in enumerate(j.coordinates):
                    self.coordinatesDict[c.name] = [j, k]
