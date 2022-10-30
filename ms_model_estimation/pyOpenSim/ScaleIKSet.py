from dataclasses import dataclass


@dataclass
class Scale:
    name: str
    markerPairSet: [[str, str]]
    bodies: [str]
    axes: [[float]] = None

    def create_axes(self):

        if self.axes is None:
            self.axes = []
            for _ in range(len(self.bodies)):
                self.axes.append([0, 1, 2])
        elif len(self.bodies) != len(self.axes) and len(self.axes) == 1:
            self.axes = [self.axes[0]] * len(self.bodies)

        elif len(self.bodies) != len(self.axes):
            assert False


@dataclass
class ScaleSet:
    scales: [Scale]

    def copy(self):
        copyClass = ScaleSet(self.scales.copy())
        return copyClass


@dataclass
class MarkerWeight:
    name: str
    weight: float = None


@dataclass
class IKSet:
    markerWeight: [MarkerWeight]
    markerWeightDict: dict = None

    def update_markerWeightDict(self):
        self.markerWeightDict = {}
        for m in self.markerWeight:
            self.markerWeightDict[m.name] = m.weight

    def copy(self):
        copyClass = IKSet(self.markerWeight.copy())
        copyClass.update_markerWeightDict()
        return copyClass
