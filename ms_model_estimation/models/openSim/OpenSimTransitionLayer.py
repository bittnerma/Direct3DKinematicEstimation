from ms_model_estimation.models.representation6D.Representation6D import Representation6D
import torch.nn as nn
import torch
from ms_model_estimation.models.openSim.OpenSimNode import OpenSimNode
from ms_model_estimation.models.openSim.OpenSimScaleLayer import OpenSimScaleLayer


class OpenSimTransitionLayer(nn.Module):

    def __init__(self, coordinateValueRange, bodyScaleValueRange, cfg):
        super(OpenSimTransitionLayer, self).__init__()

        self.cfg = cfg
        self.numberBone = self.cfg.PREDICTION.BODY_SCALE_UNIQUE_NUMS
        self.numberCoordinates = len(self.cfg.PREDICTION.COORDINATES)
        self.coordinateValueRange = coordinateValueRange.unsqueeze(0)
        self.bodyScaleValueRange = bodyScaleValueRange.unsqueeze(0)
        self.nComponents = self.cfg.TRAINING.PCA_COMPONENTS

        self.outputSize = self.numberBone + self.numberCoordinates + 6
        self.avgPooling = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        if self.cfg.LAYER.OS_TRANSIT.ACT_TYPE == 0:
            self.tanh = torch.nn.Tanh()
        elif self.cfg.LAYER.OS_TRANSIT.ACT_TYPE == 1:
            self.sigmoid = nn.Sigmoid()
        elif self.cfg.LAYER.OS_TRANSIT.ACT_TYPE == 2:
            # softmax
            self.C = (self.cfg.MODEL.IMGSIZE[0] // self.cfg.MODEL.METRIC_SCALE.STRIDING) ** 2
            indices_kernel_C = torch.linspace(0, 1, self.C)
            self.register_buffer("indices_kernel_C", indices_kernel_C)
        else:
            assert False

        # resnet model has the average layer
        self.fullyConv = self.cfg.MODEL.FULLCONV

        self.scaleLayer = OpenSimScaleLayer(cfg)

    def forward(self, x):

        B = x.shape[0]

        rootRot = x[..., :6, :, :]
        if self.fullyConv:
            rootRot = self.avgPooling(rootRot)
        rootRot = rootRot[..., :6, 0, 0]
        rootRot = Representation6D.convert_6d_vectors_to_mat(rootRot)
        rootRot = OpenSimNode.rotMat_to_homogeneous_matrix(rootRot, B)

        if self.cfg.LAYER.OS_TRANSIT.ACT_TYPE == 0:
            # transform the value range to -1 ~ 1
            x = self.tanh(x[..., 6:, :, :])
            if self.fullyConv:
                x = self.avgPooling(x)
        elif self.cfg.LAYER.OS_TRANSIT.ACT_TYPE == 1:
            x = self.sigmoid(x[..., 6:, :, :])
            # transform the value range to -1 ~ 1
            x = x * 2.0 - 1.0
            if self.fullyConv:
                x = self.avgPooling(x)
        elif self.cfg.LAYER.OS_TRANSIT.ACT_TYPE == 2:
            x = x[..., 6:, :, :]
            x = self.softmax(x)
            x = self.soft_argmax(x)
            # transform the value range to -1 ~ 1
            x = x * 2.0 - 1.0
        else:
            assert False

        # set the value to each body scale range
        predBoneScale = x[..., :self.numberBone, 0, 0] * self.bodyScaleValueRange[..., :, 1] + self.bodyScaleValueRange[
                                                                                               ..., :, 0]
        predBoneScale = self.scaleLayer(predBoneScale)
        # predBoneScale = predBoneScale * 0.6 + 1.0

        # set the values to each coordinate value range
        predRotation = x[..., self.numberBone:, 0, 0]
        predRotation = predRotation * self.coordinateValueRange[..., :, 1] + self.coordinateValueRange[..., :, 0]

        return {"predBoneScale": predBoneScale, "predRot": predRotation, "rootRot": rootRot}

    @staticmethod
    def softmax(voxels):

        assert voxels.dim() == 4

        N, C, H, W = voxels.shape
        max_along_axis = torch.max(voxels.view(N, C, -1), dim=-1, keepdim=True)[0]
        max_along_axis = max_along_axis.view((N, C, 1, 1))
        exponentiated = torch.exp(voxels - max_along_axis)
        normalizer_denominator = torch.sum(exponentiated, dim=[2, 3], keepdim=True)
        softmax = exponentiated / normalizer_denominator

        return softmax

    def soft_argmax(self, softmaxed):
        """
        Arguments: voxel patch in shape (batch_size, channel, H, W, depth)
        Return: 3D coordinates in shape (batch_size, channel, 3)
        """
        assert softmaxed.dim() == 4
        B, C, _, _ = softmaxed.shape
        x = softmaxed.view(B, C, self.C)
        x = torch.einsum('bcv , v -> bc', x, self.indices_kernel_C)
        x = x.view(B, C, 1, 1)

        return x


class OpenSim3dTransitionLayer(nn.Module):

    def __init__(self, coordinateValueRange, bodyScaleValueRange, cfg):
        super(OpenSim3dTransitionLayer, self).__init__()

        self.cfg = cfg
        self.numberBone = self.cfg.PREDICTION.BODY_SCALE_UNIQUE_NUMS
        self.numberCoordinates = len(self.cfg.PREDICTION.COORDINATES)
        self.coordinateValueRange = coordinateValueRange.unsqueeze(0)
        self.bodyScaleValueRange = bodyScaleValueRange.unsqueeze(0)
        self.nComponents = self.cfg.TRAINING.PCA_COMPONENTS

        self.outputSize = self.numberBone + self.numberCoordinates + 6
        self.avgPooling = torch.nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))

        if self.cfg.LAYER.OS_TRANSIT.ACT_TYPE == 0:
            self.tanh = torch.nn.Tanh()
        elif self.cfg.LAYER.OS_TRANSIT.ACT_TYPE == 1:
            self.sigmoid = nn.Sigmoid()
        else:
            assert False

        # resnet model has the average layer
        self.scaleLayer = OpenSimScaleLayer(cfg)

    def forward(self, x):

        B = x.shape[0]

        rootRot = x[..., :6, :, :, :]
        rootRot = self.avgPooling(rootRot)
        rootRot = rootRot[..., :6, 0, 0, 0]
        rootRot = Representation6D.convert_6d_vectors_to_mat(rootRot)
        rootRot = OpenSimNode.rotMat_to_homogeneous_matrix(rootRot, B)

        if self.cfg.LAYER.OS_TRANSIT.ACT_TYPE == 0:
            # transform the value range to -1 ~ 1
            x = self.tanh(x[..., 6:, :, :, :])
            x = self.avgPooling(x)
        elif self.cfg.LAYER.OS_TRANSIT.ACT_TYPE == 1:
            x = self.sigmoid(x[..., 6:, :, :, :])
            # transform the value range to -1 ~ 1
            x = x * 2.0 - 1.0
            x = self.avgPooling(x)
        else:
            assert False

        # set the value to each body scale range
        predBoneScale = x[..., :self.numberBone, 0, 0, 0] * self.bodyScaleValueRange[..., :, 1] + self.bodyScaleValueRange[
                                                                                               ..., :, 0]
        predBoneScale = self.scaleLayer(predBoneScale)
        # predBoneScale = predBoneScale * 0.6 + 1.0

        # set the values to each coordinate value range
        predRotation = x[..., self.numberBone:, 0, 0, 0]
        predRotation = predRotation * self.coordinateValueRange[..., :, 1] + self.coordinateValueRange[..., :, 0]

        return {"predBoneScale": predBoneScale, "predRot": predRotation, "rootRot": rootRot}
