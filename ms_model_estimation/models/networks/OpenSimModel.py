import torch
import torch.nn as nn
from ms_model_estimation.models.networks.convolutional.ResNet50 import ResNet50
from ms_model_estimation.models.networks.model_layer.OpenSimTreeLayer import OpenSimTreeLayer
from ms_model_estimation.models.networks.model_layer.OpenSimTransitionLayer import OpenSimTransitionLayer
from ms_model_estimation.models.networks.convolutional.MetricScaleLayer import MetricScaleLayer


class OpenSimModel(nn.Module):

    def __init__(
            self, pyBaseModel, coordinateValueRange, bodyScaleValueRange, cfg , evaluation =False
    ):
        super(OpenSimModel, self).__init__()

        self.cfg = cfg

        self.resnet50Layer = ResNet50(
            next=self.cfg.MODEL.RESNET.NEXT,
            pretrained=self.cfg.MODEL.RESNET.PRETRAINED,
            fullyConv=self.cfg.MODEL.FULLCONV,
            centerStriding=self.cfg.MODEL.RESNET.CENTER_STRIDING
        )

        self.opensimTransitionLayer = OpenSimTransitionLayer(
            coordinateValueRange,bodyScaleValueRange,
            cfg
        )

        self.conv_resnet50_opensim = []
        self.bn_resnet50_opensim = []
        self.activation_resnet50_opensim = torch.nn.ReLU()


        self.opensimTreeLayer = OpenSimTreeLayer(
            pyBaseModel, self.cfg.PREDICTION.BODY, self.cfg.PREDICTION.COORDINATES, self.cfg.PREDICTION.JOINTS,
            predeict_marker=self.cfg.TRAINING.MARKER_PREDICTION, predictedMarker=self.cfg.PREDICTION.MARKER,
            leafJoints=self.cfg.PREDICTION.LEAFJOINTS,
        )


        for i in range(1, self.cfg.MODEL.CONV_LAYER + 1):
            if i == self.cfg.MODEL.CONV_LAYER:
                self.conv_resnet50_opensim.append(
                    nn.Conv2d(self.cfg.MODEL.CONV_CHANNEL_SIZE[i - 1], self.opensimTransitionLayer.outputSize,
                              (1, 1)))
                if self.cfg.LAYER.OS_TRANSIT.BN:
                    self.bn_resnet50_opensim.append(torch.nn.BatchNorm2d(self.opensimTransitionLayer.outputSize))
            else:
                self.conv_resnet50_opensim.append(
                    nn.Conv2d(self.cfg.MODEL.CONV_CHANNEL_SIZE[i - 1], self.cfg.MODEL.CONV_CHANNEL_SIZE[i],
                              (1, 1)))
                self.bn_resnet50_opensim.append(torch.nn.BatchNorm2d(self.cfg.MODEL.CONV_CHANNEL_SIZE[i]))

        self.conv_resnet50_opensim = torch.nn.ModuleList(self.conv_resnet50_opensim)
        self.bn_resnet50_opensim = torch.nn.ModuleList(self.bn_resnet50_opensim)

    def forward(self, x, otherBodyPrediction=None):

        x = self.resnet50Layer(x)

        for i in range(1, self.cfg.MODEL.CONV_LAYER + 1):
            x = self.conv_resnet50_opensim[i - 1](x)
            if i == self.cfg.MODEL.CONV_LAYER:
                if self.cfg.LAYER.OS_TRANSIT.BN:
                    x = self.bn_resnet50_opensim[i - 1](x)
                continue
            else:
                x = self.bn_resnet50_opensim[i - 1](x)
                x = self.activation_resnet50_opensim(x)

        x = self.opensimTransitionLayer(x)

        if otherBodyPrediction is not None:
            x["predBoneScale"] = otherBodyPrediction

        x = self.opensimTreeLayer(x)


        return x
