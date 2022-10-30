import math
import torch.nn as nn
from ms_model_estimation.models.PoseEstimationModel import PoseEstimationModel
from ms_model_estimation.models.temporal.TCN import TemporalModel, TemporalModelOptimized1f


class TemporalPoseEstimationModel(nn.Module):

    def __init__(
            self, cfg, evaluation=False
    ):
        super(TemporalPoseEstimationModel, self).__init__()

        self.cfg = cfg
        self.poseEstimationModel = PoseEstimationModel(cfg)
        self.filter_widths = [3] * int(math.log(self.cfg.TRAINING.RECEPTIVE_FIELD, 3))

        if evaluation:
            self.tcn = TemporalModel(self.cfg.TRAINING.NUMPRED, self.cfg.TRAINING.NUMPRED, self.filter_widths)
        else:
            self.tcn = TemporalModelOptimized1f(self.cfg.TRAINING.NUMPRED, self.cfg.TRAINING.NUMPRED,
                                                self.filter_widths)

    def forward(self, x):
        assert len(x.shape) == 5

        # input size : subject x sequence length x channel x H x W
        # Batch size = S*T
        S, T, C, H, W = x.shape
        x = x.view(S * T, C, H, W)

        # output size : B x numJointIn x 3
        x = self.poseEstimationModel(x)

        # reshape: subject x sequence length x numJointIn x 3
        x = x.view(S, T, self.numJointIn, 3)

        # set the root as original point
        x = x - x[:, :, :1, :]
        y = {"intermediatePredPos": x}

        # output size : S x sequence length - receptive field + 1 x numJointIn x 3
        x = self.tcn(x)

        if self.cfg.TRAINING.REFINE:
            x = x + y["intermediatePredPos"][:, -x.shape[1]:, :, :]

        y.update({"predPos": x})

        return y
