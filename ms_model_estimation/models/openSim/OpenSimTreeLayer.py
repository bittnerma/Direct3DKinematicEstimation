import collections
import torch
from ms_model_estimation.models.openSim.OpenSimBaseTree import OpenSimBaseTree


class OpenSimTreeLayer(OpenSimBaseTree):

    def __init__(
            self, pyBaseModel, predictedBody, predictedJointCoordinates, predictedJoint,
            predeict_marker=True, predictedMarker=None,
            leafJoints=["radiocarpal", "radiocarpal_l"]

    ):
        super(OpenSimTreeLayer, self).__init__(pyBaseModel)

        # body shape
        self.predictedBody = predictedBody
        # joint angle
        self.predictedJointCoordinates = predictedJointCoordinates
        # joints for position prediction
        self.predictedJoint = predictedJoint
        # leaf joints
        self.leafJoints = leafJoints

        # UNIT_AXIS = torch.ones(4).float().requires_grad_(False)
        # UNIT_AXIS[:3] = UNIT_AXIS[:3] / (3 ** 0.5)
        # self.register_buffer("UNIT_AXIS", UNIT_AXIS)

        # self.predict_projection = predict_projection
        self.predeict_marker = predeict_marker
        self.predictedMarker = predictedMarker

        oneBodyRatioTensor = torch.ones(1, 3).float().requires_grad_(False)
        self.register_buffer("oneBodyRatioTensor", oneBodyRatioTensor)

        zeroJointAngleTensor = torch.zeros((1, 1)).float().requires_grad_(False)
        self.register_buffer("zeroJointAngleTensor", zeroJointAngleTensor)

        self.__construct_predictedBone_table()
        self.__construct_predictedCoordinate_table()
        self.__construct_original_body_scale()
        self.__construct_treeoutput_to_prediction_Table()
        # self.__construct_original_body_scale()
        # if predeict_marker:
        #   self.__construct_predictedMarker_Index_table()

    def __construct_predictedBone_table(self):

        # the dict mapping body name to the index of predicted body scale ratio vector
        bodyName_BodyScaleIndex_Table = {}
        for idx, bodyName in enumerate(self.predictedBody):
            if bodyName in self.pyBaseModel.bodySet.bodiesDict:
                bodyName_BodyScaleIndex_Table[bodyName] = idx
            else:
                assert False, print("body is not defined in opensim model.")

        # The scale of bodies are not predicted and their ratios will be set to 1.
        for body in self.pyBaseModel.bodySet.bodies:
            if body.name not in bodyName_BodyScaleIndex_Table:
                bodyName_BodyScaleIndex_Table[body.name] = -1

        self.bodyName_BodyScaleIndex_Table = bodyName_BodyScaleIndex_Table

    def __construct_predictedCoordinate_table(self):

        coordinateName_Index_Table = {}
        for idx, coordinate in enumerate(self.predictedJointCoordinates):
            coordinateName_Index_Table[coordinate] = idx

        self.coordinateName_Index_Table = coordinateName_Index_Table

    '''
    def __construct_predJoint_Index_table(self):
        predictedJoint_Index_Table = {}
        for idx, jointName in enumerate(self.predictedJoint):
            predictedJoint_Index_Table[jointName] = idx
        for jointName in self.leafJoints:
            predictedJoint_Index_Table[jointName] = len(predictedJoint_Index_Table)

        self.predictedJoint_Index_Table = predictedJoint_Index_Table

        if self.rootJointName != self.predictedJoint[0]:
            raise Exception("Make sure the first predicted joint is the "
                            "root(pelvis).")'''

    def __construct_original_body_scale(self):

        originalBodyScale = torch.empty(len(self.predictedBody), 3).float().requires_grad_(False)
        for idx, bodyName in enumerate(self.predictedBody):
            scale = self.pyBaseModel.bodySet.bodiesDict[bodyName].scale
            originalBodyScale[idx, 0] = scale[0]
            originalBodyScale[idx, 1] = scale[1]
            originalBodyScale[idx, 2] = scale[2]
        self.register_buffer("originalBodyScale", originalBodyScale)
        # self.originalBodyScale = originalBodyScale

    def __construct_treeoutput_to_prediction_Table(self):

        predictionJointOrder = []
        predictionMarkerOrder = []
        deque = collections.deque()
        deque.append(self.nodeTable[self.rootJointName])
        while deque:
            node = deque.popleft()
            jointName = node.name
            outputMarkerNames = self.jointNameToAnchoredMarkersTable.get(jointName, None)
            predictionJointOrder.append(jointName)
            if outputMarkerNames is not None:
                outputMarkerNames = outputMarkerNames[0]
                for m in outputMarkerNames:
                    predictionMarkerOrder.append(m)

            for child in self.childTable[jointName]:
                deque.append(self.nodeTable[child])


        predictionJointIndex = []
        for name in self.predictedJoint:
            predictionJointIndex.append(predictionJointOrder.index(name))
        for name in self.leafJoints:
            predictionJointIndex.append(predictionJointOrder.index(name))

        if self.predeict_marker:
            predictionMarkerIndex = []
            for name in self.predictedMarker:
                predictionMarkerIndex.append(predictionMarkerOrder.index(name))
        else:
            predictionMarkerIndex = None

        self.predictionJointIndex = predictionJointIndex
        self.predictionMarkerIndex = predictionMarkerIndex

    '''
    def __construct_predictedMarker_Index_table(self):

        if self.predictedMarker is None:
            raise Exception("predictedMarker is not defined!")

        markerName_Index_table = {}
        original_marker_location = []
        jointName_markerIndex_table = collections.defaultdict(list)

        for idx, markerName in enumerate(self.predictedMarker):
            markerName_Index_table[markerName] = idx
            marker = self.pyBaseModel.markerSet.markerDict[markerName]
            jointName = self.bodyNameJointNameTable[marker.parentFrame]
            jointName_markerIndex_table[jointName].append(idx)
            original_marker_location.append(marker.relativeLoc)

        original_marker_location = torch.Tensor(original_marker_location).float().requires_grad_(False)
        self.register_buffer("original_marker_location", original_marker_location)

        self.jointName_markerIndex_table = jointName_markerIndex_table
        self.markerName_Index_table = markerName_Index_table'''

    def forward(self, x):

        predBoneScale = x["predBoneScale"]
        predRotation = x["predRot"]
        rootRot = x.get("rootRot", None)

        # assume having the batch size
        assert len(predBoneScale.shape) == 3
        assert len(predRotation.shape) == 2

        device = predBoneScale.device

        # calculate the ratio to original scale
        predBoneRatio = torch.einsum('bij,ij->bij', predBoneScale, 1 / self.originalBodyScale)

        # update transformation with BFS
        deque = collections.deque()
        deque.append(self.nodeTable[self.rootJointName])
        predJointPos = None
        predMarkerPos = None
        while deque:

            node = deque.popleft()

            jointName = node.name
            if node.parent:
                parentJointName = node.parent.name
            else:
                parentJointName = ""

            bodyShape = self.construct_body_shape(jointName, predBoneRatio)
            parentBodyShape = self.construct_body_shape(parentJointName, predBoneRatio)

            if node.name == self.rootJointName and rootRot is not None:
                jointPos, markerPos = node(
                    None, None, None,
                    bodyShape, parentBodyShape, root=True,
                    rootRot=rootRot, child=True if node.name not in self.leafJoints else False
                )

            else:

                C = self.construct_coordinae_value_tensor(jointName, predRotation)

                jointPos, markerPos = node(C[0], C[1], C[2],
                                           bodyShape, parentBodyShape, root=False,
                                           child=True if node.name not in self.leafJoints else False
                                           )

            if predJointPos is None:
                predJointPos = jointPos
            else:
                predJointPos = torch.cat((predJointPos, jointPos), dim=1)

            if self.predeict_marker and predMarkerPos is None and markerPos is not None:
                predMarkerPos = markerPos
            elif self.predeict_marker and  markerPos is not None:
                predMarkerPos = torch.cat((predMarkerPos, markerPos), dim=1)

            '''
            # predict the joint position
            if jointName in self.predictedJoint_Index_Table:

                idx = self.predictedJoint_Index_Table[jointName]
                if jointName in self.leafJoints:
                    Pos = node.get_joint_position(B, child=False)
                else:
                    Pos = node.get_joint_position(B, child=True)
                predJointPos[..., idx, :] = Pos

                #if self.predict_projection:
                #    #predict the local position
                #    predLocalPos[..., idx, :] = node.get_rotation_projection(B, self.UNIT_AXIS)

                if self.predeict_marker:
                    markerLoc = self.construct_marker_tensor(jointName, device)
                    if markerLoc is not None:
                        markerPos = node.get_marker_position_in_ground(markerLoc, bodyShape)
                        predMarker[..., self.jointName_markerIndex_table[jointName], :] = markerPos'''

            for child in self.childTable[jointName]:
                deque.append(self.nodeTable[child])

        #print(predJointPos.shape)
        predDict = {
            "predBoneScale": predBoneScale,
            "predRot": predRotation,
            "predJointPos": predJointPos[:, self.predictionJointIndex, :],
            "rootRot": rootRot,
        }
        # if self.predict_projection:
        #    predDict["predLocalPos"] = predLocalPos
        if self.predeict_marker:
            predDict["predMarkerPos"] = predMarkerPos[:, self.predictionMarkerIndex, :]

        return predDict

    def construct_coordinae_value_tensor(self, jointName, predRotation):
        '''
        return a list of tensor with len=3 to store the coordinate value for the joint
        '''
        # create a list of tensor with len=3 to store the coordinate value for the joint
        C = []
        for coordinateName in self.jointCoordinateNameTable[jointName]:

            if coordinateName in self.coordinateName_Index_Table:
                idx = self.coordinateName_Index_Table[coordinateName]
                C.append(predRotation[..., idx:idx + 1])

            elif coordinateName in self.constraintSetTable:
                # constrained coordinate
                _, _, indCoodrinateName, func = self.constraintSetTable[coordinateName]

                if indCoodrinateName in self.coordinateName_Index_Table:
                    idx = self.coordinateName_Index_Table[indCoodrinateName]
                    val = predRotation[..., idx:idx + 1]

                else:
                    assert False, print(" the independent coordinate name is not defined in " + coordinateName)

                C.append(func(val))

            else:
                # create an zero tensor regardless of the device
                C.append(self.zeroJointAngleTensor.repeat(predRotation.shape[0],1))

        return C

    def construct_body_shape(self, jointName, predBoneRatio):
        #  Todo : not detach and clone not predicted body shape ratio and set it to the default size
        #   (prevent from the original body shape not equal to 1)
        '''
        get tensor of body ratio for the joint.
        if the body ratio is not predicted, its ratio will be set to [1,1,1]
        '''
        idx = -1
        if jointName:
            bodyName = self.jointNameBodyNameTable[jointName]
            idx = self.bodyName_BodyScaleIndex_Table.get(bodyName, -1)

        if jointName and idx >= 0:

            shapeRatio = predBoneRatio[..., idx, :]

        else:
            # create an tensor regardless of the device and set to 1
            # shapeRatio = predBoneRatio[..., 0, :].detach() * 0.0 + 1.0
            shapeRatio = self.oneBodyRatioTensor.repeat(predBoneRatio.shape[0], 1)

        return shapeRatio

    '''
    def construct_marker_tensor(self, jointName, device):
        # Todo : mrerge marker location to the OpenSimNode
        #
        if jointName in self.jointName_markerIndex_table:
            markerPos = torch.empty(len(self.jointName_markerIndex_table[jointName]), 3).float().to(device)

            for idx, idMarker in enumerate(self.jointName_markerIndex_table[jointName]):
                markerPos[idx, :] = self.original_marker_location[idMarker, :]

            return markerPos
        else:
            return None'''

    '''
    def check_all_device(self, device):

        for _, node in self.nodeTable.items():
            node.rot1Axis = self.check_device(node.rot1Axis, device)
            node.rot2Axis = self.check_device(node.rot2Axis, device)
            node.rot3Axis = self.check_device(node.rot3Axis, device)
            node.originalparentMat = self.check_device(node.originalparentMat, device)
            node.originalchildMat = self.check_device(node.originalchildMat, device)
            node.parentLoc = self.check_device(node.parentLoc, device)
            node.childLoc = self.check_device(node.childLoc, device)

        #self.UNIT_AXIS = self.check_device(self.UNIT_AXIS, device)
        self.originalBodyScale = self.check_device(self.originalBodyScale, device)
        # self.coordinateValueRange = self.check_device(self.coordinateValueRange, device)
        self.original_marker_location = self.check_device(self.original_marker_location, device)

    @staticmethod
    def check_device(tensor, device):
        if tensor.device != device:
            return tensor.to(device)
        else:
            return tensor'''
