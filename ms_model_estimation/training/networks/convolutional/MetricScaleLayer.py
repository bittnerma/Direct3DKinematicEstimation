import torch
import torch.nn as nn


class MetricScaleLayer(nn.Module):
    def __init__(
            self, numPred, imgSize=256, depth=8, striding=32, SC_W=2.2, SC_H=2.2, SC_D=2.2
    ):
        super(MetricScaleLayer, self).__init__()

        self.depth = depth
        self.striding = striding
        self.numPred = numPred
        self.outputCHNum = self.numPred * self.depth
        self.resnetlastlayer_jointpos = nn.Conv2d(2048, self.outputCHNum, (1, 1))
        # self.bn_resnetlastlayer_jointpos = nn.BatchNorm2d(self.outputCHNum)
        self.SC_W = SC_W
        self.SC_H = SC_H
        self.SC_D = SC_D

        self.W = imgSize // striding
        self.H = imgSize // striding
        self.D = depth
        indices_kernel_W = torch.linspace(0, 1, self.W) * ((self.W - 1) / self.W) * SC_W

        indices_kernel_H = torch.linspace(0, 1, self.H) * ((self.H - 1) / self.H) * SC_H

        indices_kernel_D = torch.linspace(0, 1, self.D) * ((self.D - 1) / self.D) * SC_D

        self.register_buffer("indices_kernel_W", indices_kernel_W)
        self.register_buffer("indices_kernel_H", indices_kernel_H)
        self.register_buffer("indices_kernel_D", indices_kernel_D)

    def forward(self, x):
        jointpos = self.resnetlastlayer_jointpos(x)
        # jointpos = self.bn_resnetlastlayer_jointpos(x)

        # reshape from BCHW to BJHWD
        B, C, H, W = jointpos.shape
        jointpos = jointpos.view(B, C // self.depth, H, W, self.depth)
        softmaxedJointPos = self.softmax(jointpos)
        intermediateJointPos = self.soft_argmax(softmaxedJointPos)

        return intermediateJointPos

    @staticmethod
    def softmax(voxels):
        assert voxels.dim() == 5

        N, C, H, W, D = voxels.shape
        max_along_axis = torch.max(voxels.view(N, C, -1), dim=-1, keepdim=True)[0]
        max_along_axis = max_along_axis.view((N, C, 1, 1, 1))
        exponentiated = torch.exp(voxels - max_along_axis)
        normalizer_denominator = torch.sum(exponentiated, dim=[2, 3, 4], keepdim=True)
        softmax = exponentiated / normalizer_denominator

        return softmax

    def soft_argmax(self, softmaxed):
        """
        Arguments: voxel patch in shape (batch_size, channel, H, W, depth)
        Return: 3D coordinates in shape (batch_size, channel, 3)
        """
        assert softmaxed.dim() == 5

        assert softmaxed.shape[2] == self.H
        assert softmaxed.shape[3] == self.W
        assert softmaxed.shape[4] == self.D

        x = torch.sum(softmaxed, axis=[2, 4])
        x = torch.einsum('bnw , w -> bn', x, self.indices_kernel_W)

        y = torch.sum(softmaxed, axis=[3, 4])
        y = torch.einsum('bnh , h -> bn', y, self.indices_kernel_H)

        z = torch.sum(softmaxed, axis=[2, 3])
        z = torch.einsum('bnd , d -> bn', z, self.indices_kernel_D)

        coords = torch.stack([x, y, z], dim=2)

        return coords
