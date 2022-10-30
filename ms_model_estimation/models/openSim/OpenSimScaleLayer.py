import torch.nn as nn
import torch


class OpenSimScaleLayer(nn.Module):

    def __init__(self, cfg):
        super(OpenSimScaleLayer, self).__init__()
        self.cfg = cfg
        self.mapping = self.cfg.PREDICTION.BODY_SCALE_MAPPING
        self.mappingFunc = self.create_mapping()

    def forward(self, x):

        return self.mappingFunc(x)

    def create_mapping(self):

        def inner(predictedScaleTensor):

            device = predictedScaleTensor.device
            B = predictedScaleTensor.shape[0]
            bodyScales = torch.empty(B, len(self.cfg.PREDICTION.BODY), 3).float().to(device)
            for i in range(len(self.cfg.PREDICTION.BODY)):
                for j in range(3):
                    bodyScales[:, i, j] = predictedScaleTensor[:, self.mapping[i][j]]

            return bodyScales

        return inner
