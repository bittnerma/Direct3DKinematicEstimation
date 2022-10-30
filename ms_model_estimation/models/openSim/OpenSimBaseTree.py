from abc import ABC

import torch
import torch.nn as nn
import collections
import numpy as np
from ms_model_estimation.models.openSim.OpenSimNode import OpenSimNode


class OpenSimBaseTree(nn.Module):

    def __init__(self, pyBaseModel, rootJointName="ground_pelvis"):
        super(OpenSimBaseTree, self).__init__()
        self.pyBaseModel = pyBaseModel
        self.rootJointName = rootJointName

        self.__construct_link_table()
        self.__construct_joint_name_to_coordinate_name_table()
        self.__construct_constraint_table()
        self.__construct_joint_name_to_anchor_markers()
        self.__construct_tree()

    def __construct_link_table(self):

        # the dict mapping from joint name to its parent joint name
        parentTable = collections.defaultdict(str)

        # the dict mapping from joint name to its child joint name
        childTable = collections.defaultdict(list)

        # the dict mapping from body name to its joint name
        bodyNameJointNameTable = {}

        # the dict mapping from joint name to its body name
        jointNameBodyNameTable = {}

        for idx, joint in enumerate(self.pyBaseModel.jointSet.joints):
            jointNameBodyNameTable[joint.name] = joint.frames.childFrame
            bodyNameJointNameTable[joint.frames.childFrame] = joint.name

        for joint in self.pyBaseModel.jointSet.joints:

            # the child is the joint itself.
            if "ground" not in joint.frames.parentFrame.lower():
                parentTable[joint.name] = bodyNameJointNameTable[joint.frames.parentFrame]
            else:
                # if parent is the ground, set the parent as None
                parentTable[joint.name] = None

            # add the joint to the parent's child
            if "ground" not in joint.frames.parentFrame.lower():
                childTable[bodyNameJointNameTable[joint.frames.parentFrame]].append(joint.name)

        self.parentTable = parentTable
        self.childTable = childTable
        self.bodyNameJointNameTable = bodyNameJointNameTable
        self.jointNameBodyNameTable = jointNameBodyNameTable

    def __construct_joint_name_to_coordinate_name_table(self):

        # the dict mapping from joit name to its three coordinates
        jointCoordinateNameTable = {}

        for joint in self.pyBaseModel.jointSet.joints:
            if joint.jointType == "CustomJoint":
                tmp = []
                for i in range(3):
                    tmp.append(joint.spatialTransform[i].coordinateName)
                jointCoordinateNameTable[joint.name] = tmp
            elif joint.jointType == "WeldJoint":
                jointCoordinateNameTable[joint.name] = ["", "", ""]
            else:
                assert False
        self.jointCoordinateNameTable = jointCoordinateNameTable

    def __construct_constraint_table(self):

        # the dict mapping from coordinat name to its joint name,
        # joint name of independent coordinate, independent coordinate name , and function.
        constraintSetTable = {}

        for constraint in self.pyBaseModel.constraintSet.constraints:
            if constraint.isEnforced:
                func = self.SimmSpline(constraint.funcParameters)
                constraintSetTable[constraint.dependent_coordinate_name] = [

                    # the joint name belonged by the coordinate
                    self.pyBaseModel.jointSet.coordinatesDict[constraint.dependent_coordinate_name][0].name,

                    # the joint name belonged by the independent coordinate
                    self.pyBaseModel.jointSet.coordinatesDict[constraint.independent_coordinate_names[0]][0].name,

                    # independent coordinate name
                    constraint.independent_coordinate_names[0],

                    # functions for constraints
                    # dependent coordinate value = func(independent coordinate value)
                    # for example, scapula rotation2 = func (elobw flexion)
                    func
                ]
        self.constraintSetTable = constraintSetTable

    def __construct_joint_name_to_anchor_markers(self):

        jointNameToAnchoredMarkersTable = {}
        for marker in self.pyBaseModel.markerSet.markers:
            jointName = self.bodyNameJointNameTable[marker.parentFrame]
            if jointName not in jointNameToAnchoredMarkersTable:
                jointNameToAnchoredMarkersTable[jointName] = [[], []]
            jointNameToAnchoredMarkersTable[jointName][0].append(marker.name)
            jointNameToAnchoredMarkersTable[jointName][1].append(marker.relativeLoc)

        for jointName, (_, relativeLocArray) in jointNameToAnchoredMarkersTable.items():
            jointNameToAnchoredMarkersTable[jointName][1] = np.array(relativeLocArray)

        self.jointNameToAnchoredMarkersTable = jointNameToAnchoredMarkersTable

    def __construct_tree(self):

        # Build opensim Node
        nodeTable = {}
        for idx, joint in enumerate(self.pyBaseModel.jointSet.joints):

            if joint.spatialTransform:
                rot1Axis = joint.spatialTransform[0].axis
                rot2Axis = joint.spatialTransform[1].axis
                rot3Axis = joint.spatialTransform[2].axis
            else:
                rot1Axis = [0, 0, 0]
                rot2Axis = [0, 0, 0]
                rot3Axis = [0, 0, 0]

            parentOrient = joint.frames.parentOrientation
            parentLoc = joint.frames.parentLoc
            childLoc = joint.frames.childLoc
            childOrient = joint.frames.childOrientation

            anchoredMarkers = self.jointNameToAnchoredMarkersTable.get(joint.name, None)
            node = OpenSimNode(
                joint.name,
                parentOrient=parentOrient, parentLoc=parentLoc,
                childLoc=childLoc, childOrient=childOrient,
                rot1Axis=rot1Axis, rot2Axis=rot2Axis, rot3Axis=rot3Axis,
                anchoredMarkers=anchoredMarkers[1] if anchoredMarkers is not None else None
            )

            nodeTable[joint.name] = node

        # connect the link
        for jointName, parentJointName in self.parentTable.items():
            if parentJointName:
                # connect the node to its parent
                nodeTable[jointName].parent = nodeTable[parentJointName]

        self.nodeTable = nn.ModuleDict(nodeTable)

    def forward(self, x):
        pass

    @staticmethod
    def SimmSpline(funcParameters):
        xRange = funcParameters[0].split()
        yRange = funcParameters[1].split()
        xRange = [float(p) for p in xRange]
        yRange = [float(p) for p in yRange]

        def inner(value):
            return (value - xRange[0]) * (yRange[1] - yRange[0]) / (xRange[1] - xRange[0]) + yRange[0]

        return inner
