import torch.nn as nn
import torch
from ms_model_estimation.training.networks.PoseEstimationModel import PoseEstimationModel
from ms_model_estimation.training.networks.TemporalModel import TemporalPoseEstimationModel


class SpatialTemporalModel(nn.Module):

    def __init__(self, cfg, evaluation=False, reshapeToJoints=True):
        super(SpatialTemporalModel, self).__init__()

        self.poseEstimationModel = PoseEstimationModel(cfg)
        self.temporalPoseEstimationModel = TemporalPoseEstimationModel(cfg, evaluation=evaluation,
                                                                       reshapeToJoints=reshapeToJoints)
        self.cfg = cfg

    def forward(self, images):
        B, S, _, _, _ = images.shape
        assert len(images.shape) == 5
        assert images.shape[2] == 3

        images = images.view(B * S, 3, self.cfg.MODEL.IMGSIZE[0], self.cfg.MODEL.IMGSIZE[1])

        spatialPredPos = self.poseEstimationModel(images)

        spatialPredPos = spatialPredPos.view(B, S, self.cfg.PREDICTION.NUMPRED, 3)
        # set the root to 0
        spatialPredPos = spatialPredPos - spatialPredPos[:, :, :1, :]
        temporalPredPos = self.temporalPoseEstimationModel(spatialPredPos)

        return {"spatialPredPos": spatialPredPos, "temporalPredPos": temporalPredPos}
