from dataclasses import dataclass
from ms_model_estimation.pyOpenSim.BodySet import BodySet
from ms_model_estimation.pyOpenSim.JointSet import JointSet
from ms_model_estimation.pyOpenSim.MarkerSet import MarkerSet
from ms_model_estimation.pyOpenSim.ConstraintSet import ConstraintSet


@dataclass
class PyOpenSimModel:
    bodySet: BodySet
    jointSet: JointSet
    markerSet: MarkerSet
    constraintSet: ConstraintSet = None
