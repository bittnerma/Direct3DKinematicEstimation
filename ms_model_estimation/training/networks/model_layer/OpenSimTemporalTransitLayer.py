from ms_model_estimation.training.representation6D.Representation6D import Representation6D
import torch.nn as nn
from ms_model_estimation.training.networks.model_layer.OpenSimNode import OpenSimNode
from ms_model_estimation.training.networks.model_layer.OpenSimScaleLayer import OpenSimScaleLayer


class OpenSimTransitionLayer(nn.Module):

    def __init__(self, coordinateValueRange, bodyScaleValueRange, cfg):
        super(OpenSimTransitionLayer, self).__init__()

        self.cfg = cfg
        self.numberBone = self.cfg.PREDICTION.BODY_SCALE_UNIQUE_NUMS
        self.numberCoordinates = len(self.cfg.PREDICTION.COORDINATES)
        self.coordinateValueRange = coordinateValueRange.unsqueeze(0)
        self.bodyScaleValueRange = bodyScaleValueRange.unsqueeze(0)

        # Directly Predict the coordinate angle without activation function
        self.outputSize = self.numberBone + self.numberCoordinates + 6
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.scaleLayer = OpenSimScaleLayer(cfg)

    def forward(self, x):

        assert x.shape[-1] == self.outputSize
        B = x.shape[0]

        rootRot = x[..., :6]
        rootRot = Representation6D.convert_6d_vectors_to_mat(rootRot)
        rootRot = OpenSimNode.rotMat_to_homogeneous_matrix(rootRot, B)

        if self.cfg.LAYER.OS_TRANSIT.ACT_TYPE == 0:
            # transform the value range to -1 ~ 1
            x = self.tanh(x[..., 6:])
        elif self.cfg.LAYER.OS_TRANSIT.ACT_TYPE == 1:
            # transform the value range to -1 ~ 1
            x = self.sigmoid(x[..., 6:]) * 2.0 - 1.0
        else:
            assert False

        # set the value to each body scale range
        predBoneScale = x[..., :self.numberBone] * self.bodyScaleValueRange[..., :, 1] + self.bodyScaleValueRange[
                                                                                               ..., :, 0]
        predBoneScale = self.scaleLayer(predBoneScale)
        # predBoneScale = predBoneScale * 0.6 + 1.0

        # set the values to each coordinate value range
        predRotation = x[..., self.numberBone:]
        predRotation = predRotation * self.coordinateValueRange[..., :, 1] + self.coordinateValueRange[..., :, 0]

        return {"predBoneScale": predBoneScale, "predRot": predRotation, "rootRot": rootRot}
