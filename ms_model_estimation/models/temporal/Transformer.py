import math

import torch.nn as nn
from torch.nn import MultiheadAttention
import torch


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # batch size x sequence length x d_model 
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.max_len = max_len

    def forward(self, x):
        assert x.shape[1] < self.max_len
        x = x + self.pe[:, :x.shape[1], :]
        return x


class Encoder(nn.Module):

    def __init__(self, positionEncodingLayer, inFeatures, outFeatures, d_model, num_heads, strides, d_channel=512,
                 batchNorm=False,  dropOut=False):
        super(Encoder, self).__init__()
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.d_model = d_model
        self.num_heads = num_heads
        self.strides = strides
        self.d_channel = d_channel
        self.batchNorm = batchNorm
        self.dropOut = dropOut

        # position embedding
        self.RECEPTIVE_FIELD = self.receptive_field()
        self.positionEncodingLayer = positionEncodingLayer

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 1
        for s in self.strides:
            frames *= s
        return frames


class VanillaEncoder(Encoder):

    def __init__(
            self, positionEncodingLayer, inFeatures, outFeatures,
            d_model, num_heads, strides, N1=3, d_channel=512, dropOut=False, positionEncodingEveryBlock=False
    ):
        super(VanillaEncoder, self).__init__(positionEncodingLayer, inFeatures, outFeatures, d_model, num_heads,
                                             strides, d_channel=d_channel,
                                             batchNorm=False, dropOut=dropOut)

        self.N1 = N1
        encoderLayers = []
        for _ in range(N1):
            encoderLayers.append(
                Block(d_model, num_heads, 1, 0, d_channel=d_channel, dropOut=dropOut, batchNorm=False))
        self.encoderLayers = nn.ModuleList(encoderLayers)
        self.positionEncodingEveryBlock = positionEncodingEveryBlock

    def forward(self, x):


        if not self.positionEncodingEveryBlock:
            # position embedding
            x = self.positionEncodingLayer(x)

        # vanilla encoder layer
        for i in range(len(self.encoderLayers)):

            if self.positionEncodingEveryBlock:
                x = self.positionEncodingLayer(x)

            x = self.encoderLayers[i](x)

        return x


class StridedEncoder(Encoder):

    def __init__(self, positionEncodingLayer, inFeatures, outFeatures, d_model, num_heads, strides, d_channel=512,
                 dropOut=False,
                 batchNorm=False, causal=False):
        super(StridedEncoder, self).__init__(positionEncodingLayer, inFeatures, outFeatures, d_model, num_heads,
                                             strides, d_channel=d_channel,
                                             batchNorm=batchNorm,
                                             dropOut=dropOut)

        # causal shift
        self.causal_shift = []
        for i in range(len(strides)):
            self.causal_shift.append((strides[i] // 2) if causal else 0)

        strideEncoderLayers = []
        for idx, s in enumerate(self.strides):
            strideEncoderLayers.append(
                Block(d_model, num_heads, s, self.causal_shift[idx], d_channel=d_channel, dropOut=dropOut,
                      batchNorm=batchNorm, bias=False))
        self.strideEncoderLayers = nn.ModuleList(strideEncoderLayers)

    def set_bn_momentum(self, momentum):
        if self.batchNorm:
            for i in range(len(self.strideEncoderLayers)):
                self.strideEncoderLayers[i].bn1.momentum = momentum

    def forward(self, x):

        # stride encoder layer
        for i in range(len(self.strideEncoderLayers)):
            # position embedding
            x = self.positionEncodingLayer(x)
            x = self.strideEncoderLayers[i](x)

        return x


class StridedTransformerEncoder(nn.Module):

    def __init__(self, inFeatures, outFeatures, d_model, num_heads, strides, d_channel=512, batchNorm=False,
                 dropOut=False, causal=False):
        super(StridedTransformerEncoder, self).__init__()

        positionEncodingLayer = PositionalEncoding(d_model)
        self.stridedEncoder = StridedEncoder(positionEncodingLayer, inFeatures, outFeatures, d_model, num_heads,
                                             strides, d_channel=d_channel,
                                             dropOut=dropOut, batchNorm=batchNorm, causal=causal)
        self.expandingLayer = nn.Conv1d(inFeatures, d_model, 1)
        self.stridedShrinkLayer = nn.Conv1d(d_model, outFeatures, 1)

    def forward(self, x):
        assert len(x.shape) == 3
        assert x.shape[-1] == self.stridedEncoder.inFeatures

        x = x.permute(0, 2, 1)
        x = self.expandingLayer(x)
        x = x.permute(0, 2, 1)

        x = self.stridedEncoder(x)

        x = x.permute(0, 2, 1)
        x = self.stridedShrinkLayer(x)
        x = x.permute(0, 2, 1)

        return x

    def set_bn_momentum(self, momentum):
        self.stridedEncoder.set_bn_momentum(momentum)


class VanillaTransformerEncoder(nn.Module):

    def __init__(self, inFeatures, outFeatures, d_model, num_heads, strides, N1=3, d_channel=512,
                 dropOut=False, positionEncodingEveryBlock=False):
        super(VanillaTransformerEncoder, self).__init__()
        positionEncodingLayer = PositionalEncoding(d_model)
        self.vanillaEncoder = VanillaEncoder(positionEncodingLayer, inFeatures, outFeatures, d_model, num_heads,
                                             strides, N1=N1, d_channel=d_channel, dropOut=dropOut, positionEncodingEveryBlock=positionEncodingEveryBlock)
        self.expandingLayer = nn.Conv1d(inFeatures, d_model, 1)
        self.stridedShrinkLayer = nn.Conv1d(d_model, outFeatures, 1)

    def forward(self, x):
        assert len(x.shape) == 3
        assert x.shape[-1] == self.vanillaEncoder.inFeatures

        x = x.permute(0, 2, 1)
        x = self.expandingLayer(x)
        x = x.permute(0, 2, 1)

        x = self.vanillaEncoder(x)

        x = x.permute(0, 2, 1)
        x = self.stridedShrinkLayer(x)
        x = x.permute(0, 2, 1)

        return x

    def set_bn_momentum(self, momentum):
        pass


class TransformerEncoder(nn.Module):

    def __init__(self, inFeatures, outFeatures, d_model, num_heads, strides, N1=3, d_channel=512, batchNorm=False,
                 dropOut=False, causal=False, positionEncodingEveryBlock=False):
        super(TransformerEncoder, self).__init__()

        positionEncodingLayer = PositionalEncoding(d_model)

        self.vanillaEncoder = VanillaEncoder(positionEncodingLayer, inFeatures, outFeatures, d_model, num_heads,
                                             strides, N1=N1,
                                             d_channel=d_channel, dropOut=dropOut, positionEncodingEveryBlock=positionEncodingEveryBlock)
        self.stridedEncoder = StridedEncoder(positionEncodingLayer, inFeatures, outFeatures, d_model, num_heads,
                                             strides, d_channel=d_channel,
                                             dropOut=dropOut, batchNorm=batchNorm, causal=causal)
        self.expandingLayer = nn.Conv1d(inFeatures, d_model, 1)
        self.vanillaShrinkLayer = nn.Conv1d(d_model, outFeatures, 1)
        self.stridedShrinkLayer = nn.Conv1d(d_model, outFeatures, 1)

    def forward(self, x):
        assert len(x.shape) == 3
        assert x.shape[-1] == self.vanillaEncoder.inFeatures

        x = x.permute(0, 2, 1)
        x = self.expandingLayer(x)
        x = x.permute(0, 2, 1)

        x = self.vanillaEncoder(x)

        intermediatePos = x.permute(0, 2, 1)
        intermediatePos = self.vanillaShrinkLayer(intermediatePos)
        intermediatePos = intermediatePos.permute(0, 2, 1)

        x = self.stridedEncoder(x)

        x = x.permute(0, 2, 1)
        x = self.stridedShrinkLayer(x)
        x = x.permute(0, 2, 1)

        return {
            "predPos": x,
            "intermediatePos": intermediatePos
        }

    def set_bn_momentum(self, momentum):
        self.stridedEncoder.set_bn_momentum(momentum)


class Block(nn.Module):

    def __init__(self, d_model, num_heads, stride, causal_shift, d_channel=512, dropOut=False, batchNorm=False,
                 bias=True):
        super(Block, self).__init__()

        self.selfAttention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.conv1 = nn.Conv1d(d_model, d_channel, 1, stride=1, bias=bias)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(d_channel, d_model, 1, stride=stride, bias=bias)
        self.layerNorm1 = nn.LayerNorm(d_model)
        self.layerNorm2 = nn.LayerNorm(d_model)
        self.stride = stride
        self.causal_shift = causal_shift
        self.batchNorm = batchNorm
        if batchNorm:
            self.bn1 = nn.BatchNorm1d(d_channel)
            # self.bn2 = nn.BatchNorm1d(d_model)

        self.dropOut = dropOut
        if dropOut:
            self.dropout1 = nn.Dropout(0.1)
            self.dropout2 = nn.Dropout(0.25)

    def forward(self, src):

        # attention layer
        x = self.selfAttention(src, src, src)[0]
        if self.dropOut:
            x = self.dropout1(x)
        x = src + x
        src2 = self.layerNorm1(x)

        # 1d conv
        src3 = src2.permute(0, 2, 1)

        x = self.conv1(src3[:, :, self.causal_shift + self.stride // 2:])
        if self.batchNorm:
            x = self.bn1(x)
        x = self.relu(x)
        if self.dropOut:
            x = self.dropout2(x)

        x = self.conv2(x)
        '''
        if self.batchNorm:
            x = self.bn2(x)'''

        x = src2[:, self.causal_shift + self.stride // 2::self.stride, :] + x.permute(0, 2, 1)

        x = self.layerNorm2(x)

        return x
