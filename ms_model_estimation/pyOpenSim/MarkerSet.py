from dataclasses import dataclass
from typing import Optional

@dataclass
class Marker:
    name: str
    parentFrame: str
    relativeLoc: [float]
    fixed: bool = False


class MarkerSet:

    def __init__(
        self,
        markers: [Marker]
    ):
        self.markers=markers
        self.markerDict={}
        self.update_markerDict()

    def update_markerDict(self):
        self.markerDict={}
        for m in self.markers:
            self.markerDict[m.name]=m