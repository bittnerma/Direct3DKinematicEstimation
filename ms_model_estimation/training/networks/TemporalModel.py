import math
import torch.nn as nn
from ms_model_estimation.training.networks.sequential.TCN_FEATURES import TemporalModel, TemporalModelOptimized1f
from ms_model_estimation.training.networks.sequential.Transformer import StridedTransformerEncoder, VanillaTransformerEncoder, \
    TransformerEncoder
from ms_model_estimation.training.networks.sequential.LSTM import LSTM

class TemporalPoseEstimationModel(nn.Module):

    def __init__(
            self, cfg, evaluation=False, reshapeToJoints=True
    ):
        super(TemporalPoseEstimationModel, self).__init__()

        self.cfg = cfg
        self.filter_widths = [3] * math.ceil(math.log(self.cfg.MODEL.RECEPTIVE_FIELD, 3))

        if self.cfg.MODEL.TYPE == 0:

            if evaluation:
                self.temporalModel = TemporalModel(self.cfg.MODEL.INFEATURES, self.cfg.MODEL.OUTFEATURES,
                                                   self.filter_widths, causal=self.cfg.MODEL.CAUSAL)
            else:
                self.temporalModel = TemporalModelOptimized1f(
                    self.cfg.MODEL.INFEATURES, self.cfg.MODEL.OUTFEATURES,
                    self.filter_widths, causal=self.cfg.MODEL.CAUSAL
                )
            print("Use Temporal Convolutional Network")
            print(f'Receptive Field: {self.temporalModel.receptive_field()}')
            print(f'Filter Widths: {self.filter_widths}')

        elif self.cfg.MODEL.TYPE == 1:

            self.temporalModel = TransformerEncoder(
                self.cfg.MODEL.INFEATURES,
                self.cfg.MODEL.OUTFEATURES,
                self.cfg.TRANSFORMER.D_MODEL,
                self.cfg.TRANSFORMER.N_HEADS,
                self.cfg.TRANSFORMER.STRIDES,
                N1=self.cfg.TRANSFORMER.N1,
                d_channel=self.cfg.TRANSFORMER.D_CHANNEL,
                batchNorm=self.cfg.TRANSFORMER.BATCHNORM,
                dropOut=self.cfg.TRANSFORMER.DROPOUT,
                causal=self.cfg.MODEL.CAUSAL,
                positionEncodingEveryBlock=self.cfg.TRANSFORMER.POS_ENCODNIG_EVERY_BLOCK,
            )
            print(f'Use Transformer, Receptive Field : {self.cfg.MODEL.RECEPTIVE_FIELD}')

        elif self.cfg.MODEL.TYPE == 2:

            self.temporalModel = StridedTransformerEncoder(
                self.cfg.MODEL.INFEATURES,
                self.cfg.MODEL.OUTFEATURES,
                self.cfg.TRANSFORMER.D_MODEL,
                self.cfg.TRANSFORMER.N_HEADS,
                self.cfg.TRANSFORMER.STRIDES,
                d_channel=self.cfg.TRANSFORMER.D_CHANNEL,
                batchNorm=self.cfg.TRANSFORMER.BATCHNORM,
                dropOut=self.cfg.TRANSFORMER.DROPOUT,
                causal=self.cfg.MODEL.CAUSAL,
            )
            print(f'Use Strided Transformer, Receptive Field : {self.cfg.MODEL.RECEPTIVE_FIELD}')

        elif self.cfg.MODEL.TYPE == 3:

            self.temporalModel = VanillaTransformerEncoder(
                self.cfg.MODEL.INFEATURES,
                self.cfg.MODEL.OUTFEATURES,
                self.cfg.TRANSFORMER.D_MODEL,
                self.cfg.TRANSFORMER.N_HEADS,
                self.cfg.TRANSFORMER.STRIDES,
                N1=self.cfg.TRANSFORMER.N1,
                d_channel=self.cfg.TRANSFORMER.D_CHANNEL,
                dropOut=self.cfg.TRANSFORMER.DROPOUT,
                positionEncodingEveryBlock=self.cfg.TRANSFORMER.POS_ENCODNIG_EVERY_BLOCK,
            )
            print(f'Use Vanilla Transformer, Receptive Field : {self.cfg.MODEL.RECEPTIVE_FIELD}')

        elif self.cfg.MODEL.TYPE == 4:
            self.temporalModel = LSTM(
                self.cfg.MODEL.INFEATURES,
                self.cfg.LSTM.HIDDEN_STATE,
                self.cfg.MODEL.OUTFEATURES,
                self.cfg.LSTM.NUM_LAYERS,
                self.cfg.LSTM.DROPOUTPROB,
                bidirectional=self.cfg.LSTM.BIDIRECTIONAL
            )
            print(f'Use LSTM, Receptive Field : {self.cfg.MODEL.RECEPTIVE_FIELD}')
        else:
            assert False

        self.reshapeToJoints = reshapeToJoints

    def forward(self, x):

        B, S = x.shape[0], x.shape[1]

        if len(x.shape) == 4:
            x = x.view(x.shape[0], x.shape[1], self.cfg.MODEL.INFEATURES)
        assert x.shape[-1] == self.cfg.MODEL.INFEATURES
        assert x.shape[1] == self.cfg.MODEL.RECEPTIVE_FIELD

        y = self.temporalModel(x)

        if self.reshapeToJoints:
            if self.cfg.MODEL.TYPE == 0 or self.cfg.MODEL.TYPE == 2:
                y = y.view(B, 1, self.cfg.PREDICTION.NUMPRED, 3)
            elif self.cfg.MODEL.TYPE == 3 or self.cfg.MODEL.TYPE == 4:
                y = y.view(B, S, self.cfg.PREDICTION.NUMPRED, 3)
            elif self.cfg.MODEL.TYPE == 1:
                y["predPos"] = y["predPos"].view(B, 1, self.cfg.PREDICTION.NUMPRED, 3)
                y["intermediatePos"] = y["intermediatePos"].view(B, S, self.cfg.PREDICTION.NUMPRED, 3)

        return y
