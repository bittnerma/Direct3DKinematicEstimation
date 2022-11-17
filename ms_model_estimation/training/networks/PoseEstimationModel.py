import torch.nn as nn
from ms_model_estimation.training.networks.convolutional.ResNet50 import ResNet50
from ms_model_estimation.training.networks.convolutional.MetricScaleLayer import MetricScaleLayer
import torch


class PoseEstimationModel(nn.Module):

    def __init__(
            self, cfg
    ):
        super(PoseEstimationModel, self).__init__()

        self.cfg = cfg

        self.resnet50Layer = ResNet50(
            next=self.cfg.MODEL.RESNET.NEXT,
            pretrained=self.cfg.MODEL.RESNET.PRETRAINED,
            fullyConv=True,
            centerStriding=self.cfg.MODEL.RESNET.CENTER_STRIDING,
        )
        self.metricScaleLayer = MetricScaleLayer(
            self.cfg.PREDICTION.NUMPRED,
            imgSize = self.cfg.MODEL.IMGSIZE[0],
            depth=self.cfg.MODEL.METRIC_SCALE.DEPTH,
            striding=self.cfg.MODEL.METRIC_SCALE.STRIDING,
            SC_W=self.cfg.MODEL.METRIC_SCALE.BBOX_SIZE,
            SC_H=self.cfg.MODEL.METRIC_SCALE.BBOX_SIZE,
            SC_D=self.cfg.MODEL.METRIC_SCALE.BBOX_SIZE
        )

    def forward(self, x):

        x = self.resnet50Layer(x)
        x = self.metricScaleLayer(x)

        return x

    def set_bn_momentum(self, momentum):

        for name, layer in self.resnet50Layer.named_modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.momentum = momentum

        for name, layer in self.metricScaleLayer.named_modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.momentum = momentum