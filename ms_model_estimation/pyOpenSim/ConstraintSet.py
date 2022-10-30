from dataclasses import dataclass
from typing import  Any


@ dataclass
class Constraint:
    # name of the constraint
    name: str
    # use this constraint?
    isEnforced: bool
    # only support SimmSpline
    funcType: str
    funcParameters: Any
    funcName: str
    independent_coordinate_names: [str]
    dependent_coordinate_name: str

@ dataclass
class ConstraintSet:
    constraints: [Constraint]
