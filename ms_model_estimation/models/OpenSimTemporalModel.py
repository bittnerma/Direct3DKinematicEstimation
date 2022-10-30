import math
import torch
import torch.nn as nn
from ms_model_estimation.models.TemporalModel import TemporalPoseEstimationModel
from ms_model_estimation.models.openSim.OpenSimTemporalTransitLayer import OpenSimTransitionLayer
from ms_model_estimation.models.openSim.OpenSimTreeLayer import OpenSimTreeLayer


class OpenSimTemporalModel(nn.Module):

    def __init__(
            self, pyBaseModel, coordinateValueRange, bodyScaleValueRange, cfg, evaluation=False
    ):
        super(OpenSimTemporalModel, self).__init__()

        self.cfg = cfg
        self.openSimTransitionLayer = OpenSimTransitionLayer(coordinateValueRange, bodyScaleValueRange, cfg)
        self.temporalModel = TemporalPoseEstimationModel(cfg, evaluation=evaluation, reshapeToJoints=False)
        self.opensimTreeLayer = OpenSimTreeLayer(
            pyBaseModel, self.cfg.PREDICTION.BODY, self.cfg.PREDICTION.COORDINATES, self.cfg.PREDICTION.JOINTS,
            predeict_marker=self.cfg.TRAINING.MARKER_PREDICTION, predictedMarker=self.cfg.PREDICTION.MARKER,
            leafJoints=self.cfg.PREDICTION.LEAFJOINTS,
        )

    def forward(self, x):

        B, S, _, _ = x["predPos"].shape


        if self.cfg.MODEL.POS:
            # reshape to 1d vector( B x S x feature sizes)
            predPos = x["predPos"]
            predPos1d = predPos.view((B, S, -1))

            predMarkerPos = x["predMarkerPos"]
            predMarkerPos1d = predMarkerPos.view((B, S, -1))

        predBoneScale = x["predBoneScale"]
        predBoneScale1d = predBoneScale.view((B, S, -1))

        rootRot = x["predRootRot"]
        rootRot1d = rootRot.view((B, S, 9))

        predRot = x["predRot"]

        if self.cfg.MODEL.POS:
            inputFeatures = torch.cat((predPos1d, predMarkerPos1d, predBoneScale1d, rootRot1d, predRot), dim=2)
        else:
            inputFeatures = torch.cat((predBoneScale1d, rootRot1d, predRot), dim=2)

        outputFeatures = self.temporalModel(inputFeatures)

        if self.cfg.MODEL.TYPE == 1:
            intermediatePred = outputFeatures["intermediatePos"]
            y = outputFeatures["predPos"]
            y = torch.cat((intermediatePred, y), dim=1)
            S = S + 1
        else:
            y = outputFeatures
            S = y.shape[1]

        # (BxS) x Channel Size
        y = y.contiguous()
        y = y.view(B * S, self.cfg.MODEL.OUTFEATURES)
        y = self.openSimTransitionLayer(y)
        outputs = self.opensimTreeLayer(y)

        # Reshape back to B x S x ....
        outputs["predJointPos"] = outputs["predJointPos"].view(B, S, self.cfg.PREDICTION.NUM_JOINTS, 3)
        outputs["predMarkerPos"] = outputs["predMarkerPos"].view(B, S, self.cfg.PREDICTION.NUM_MARKERS, 3)
        outputs["predBoneScale"] = outputs["predBoneScale"].view(B, S, len(self.cfg.PREDICTION.BODY), 3)
        outputs["predRot"] = outputs["predRot"].view(B, S, len(self.cfg.PREDICTION.COORDINATES))
        outputs["rootRot"] = outputs["rootRot"].view(B, S, 4, 4)

        return outputs
