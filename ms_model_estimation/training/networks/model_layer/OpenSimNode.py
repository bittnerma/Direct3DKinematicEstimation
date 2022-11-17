import torch
import numpy as np
import torch.nn as nn


class OpenSimNode(nn.Module):

    def __init__(
            self, name,
            parentOrient=torch.Tensor([0, 0, 0]).float(), parentLoc=torch.Tensor([0, 0, 0]).float(),
            childOrient=torch.Tensor([0, 0, 0]).float(), childLoc=torch.Tensor([0, 0, 0]).float(),
            rot1Axis=torch.Tensor([0, 0, 0]).float(), rot2Axis=torch.Tensor([0, 0, 0]).float(),
            rot3Axis=torch.Tensor([0, 0, 0]).float(), anchoredMarkers=None
    ):
        '''

        :param name: body name
        :param parentOrient: parent orientation
        :param parentLoc: child orientation
        :param childOrient: parent relative location
        :param childLoc: child relative location
        :param rot1Axis: rotation axis for the first coordinate
        :param rot2Axis: rotation axis for the second coordinate
        :param rot3Axis: rotation axis for the third coordinate
        '''
        super(OpenSimNode, self).__init__()

        self.name = name

        parentOrient = torch.Tensor(parentOrient).float()
        childOrient = torch.Tensor(childOrient).float()
        parentLoc = torch.Tensor(parentLoc).float()
        childLoc = torch.Tensor(childLoc).float()
        rot1Axis = torch.Tensor(rot1Axis).float()
        rot2Axis = torch.Tensor(rot2Axis).float()
        rot3Axis = torch.Tensor(rot3Axis).float()
        if anchoredMarkers is not None:
            anchoredMarkers = torch.Tensor(anchoredMarkers).float()

        self.register_buffer("parentOrient", parentOrient)
        self.register_buffer("childOrient", childOrient)
        self.register_buffer("parentLoc", parentLoc)
        self.register_buffer("childLoc", childLoc)
        self.register_buffer("rot1Axis", rot1Axis)
        self.register_buffer("rot2Axis", rot2Axis)
        self.register_buffer("rot3Axis", rot3Axis)
        if anchoredMarkers is not None:
            self.register_buffer("anchoredMarkers", anchoredMarkers)
            if len(self.anchoredMarkers.shape) == 1:
                self.anchoredMarkers = self.anchoredMarkers.view(-1, 3)
            else:
                assert self.anchoredMarkers.shape[-1] == 3
        else:
            self.anchoredMarkers = None

        self.parent = None
        self.transformInGround = None
        self.R = None

        # Calculate transfromation for parent frame and child frame without translation
        self.calculate_frame_transform()

        self.rot1Axis.requires_grad = False
        self.rot2Axis.requires_grad = False
        self.rot3Axis.requires_grad = False
        self.originalparentMat.requires_grad = False
        self.originalchildMat.requires_grad = False
        self.parentLoc.requires_grad = False
        self.childLoc.requires_grad = False
        self.parentOrient.requires_grad = False
        self.childOrient.requires_grad = False
        if anchoredMarkers is not None:
            self.anchoredMarkers.requires_grad = False

    def calculate_frame_transform(self):

        # Convert euler angle to 4x4 rotation matrix without translation
        mat = OpenSimNode.euler2mat(self.parentOrient[0], self.parentOrient[1], self.parentOrient[2])
        parentFrame = torch.from_numpy(mat).float()

        mat = OpenSimNode.euler2mat(self.childOrient[0], self.childOrient[1], self.childOrient[2])
        childFrame = torch.from_numpy(mat).float()

        self.register_buffer("originalparentMat", parentFrame)
        self.register_buffer("originalchildMat", childFrame)

        # self.originalparentMat = parentFrame
        # self.originalchildMat = childFrame

    def forward(
            self, C1, C2, C3, bodyShapeRatio, parentBodyShapeRatio,
            root=False, rootRot=None, child=True
    ):
        '''
        Update the transformation matrix, R and transformInGround.
        :param C1: the first coordinate angle value
        :param C2: the second coordinate angle value
        :param C3: the third coordinate angle value
        :param bodyShapeRatio: the ratio to original body scale
        :param parentBodyShapeRatio: the ratio to original parent body scale
        :param root: whether the node is the root of the tree (pelvis)
        :param rootRot: the rotation matrix from gournd to the root
        :return:
        '''
        if len(bodyShapeRatio.shape) == 1:
            B = 0
        else:
            B = bodyShapeRatio.shape[0]

        self.get_frame_transform(B, bodyShapeRatio, parentBodyShapeRatio)

        rot1Axis = self.rot1Axis
        rot2Axis = self.rot2Axis
        rot3Axis = self.rot3Axis

        # eyeMatrix = torch.eye(3).float().to(device)

        if B > 0:
            if root and rootRot is not None:
                # R = self.rotMat_to_homogeneous_matrix(rootRot, B)
                self.R = rootRot

                # transformation from parent to itself
                self.transformInGround = torch.einsum('bij , bjk -> bik', self.R, self.childFrame)

            else:
                # The rotation mat of first coordinate
                R1 = OpenSimNode.axangle2mat(rot1Axis.unsqueeze(0).repeat(B, 1), C1)

                # The rotation mat of second coordinate
                temp = torch.einsum('bij , j -> bi', R1, rot2Axis)
                R2 = OpenSimNode.axangle2mat(temp, C2)

                # The rotation mat of third coordinate
                temp = torch.einsum('bij , j -> bi', R1, rot3Axis)
                temp = torch.einsum('bij , bj -> bi', R2, temp)
                R3 = OpenSimNode.axangle2mat(temp, C3)

                # The overall rotation mat
                # x = torch.einsum('bij , jk -> bik', R1, eyeMatrix)
                x = torch.matmul(R2, R1)
                x = torch.matmul(R3, x)
                R = OpenSimNode.rotMat_to_homogeneous_matrix(x, B)

                self.R = R

                # transformation from parent to itself
                tmp = torch.einsum('bij , bjk -> bik', R, self.childFrame)
                self.transformInGround = torch.einsum(' bij, bjk -> bik', self.parentFrame, tmp)

        else:
            if root and rootRot is not None:
                self.R = rootRot
                # transformation from parent to itself
                self.transformInGround = torch.matmul(self.R, self.childFrame)
            else:

                # The rotation mat of first coordinate
                R1 = OpenSimNode.axangle2mat(rot1Axis, C1)

                # The rotation mat of second coordinate
                temp = torch.matmul(R1, rot2Axis)
                R2 = OpenSimNode.axangle2mat(temp, C2)

                # The rotation mat of third coordinate
                temp = torch.matmul(R1, rot3Axis)
                temp = torch.matmul(R2, temp)
                R3 = OpenSimNode.axangle2mat(temp, C3)

                # The overall rotation mat
                # x = torch.matmul(R1, eyeMatrix)
                x = torch.matmul(R2, R1)
                x = torch.matmul(R3, x)
                R = OpenSimNode.rotMat_to_homogeneous_matrix(x, B)

                self.R = R

                # transformation from parent to itself
                self.transformInGround = torch.matmul(self.parentFrame, torch.matmul(R, self.childFrame))

        # transformation from ground to itself
        if self.parent is not None:
            self.transformInGround = torch.matmul(self.parent.transformInGround, self.transformInGround)

        if self.anchoredMarkers is not None:
            return self.get_joint_position(child=child), self.get_marker_position_in_ground(bodyShapeRatio)
        else:
            return self.get_joint_position(child=child), None

    @staticmethod
    def rotMat_to_homogeneous_matrix(mat, batchSize=0):

        device = mat.device
        assert mat.shape[-1] == 3 and mat.shape[-2] == 3

        if len(mat.shape) == 3:
            batchSize = mat.shape[0]
            result = torch.empty((batchSize, 4, 4), dtype=mat.dtype).to(device)
        elif len(mat.shape) == 2:
            result = torch.empty((4, 4), dtype=mat.dtype).to(device)
        else:
            assert False

        result[..., :3, :3] = mat
        result[..., -1, -1] = 1
        result[..., :3, -1] = 0
        result[..., -1, :3] = 0

        return result

    @staticmethod
    def axangle2mat(axis, angle):
        '''
        modify the code from https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/axangles.py
        '''

        device = angle.device

        if len(angle.shape) == 1:
            result = torch.empty((3, 3), dtype=angle.dtype).to(device)
        elif len(angle.shape) == 2:
            B = angle.shape[0]
            result = torch.empty((B, 3, 3), dtype=angle.dtype).to(device)
        else:
            assert False

        axis = axis / (torch.linalg.norm(axis, ord=2, dim=-1, keepdim=True) + 10 ** -15)
        if len(angle.shape) == 1:
            x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
        else:
            x, y, z = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]

        c = torch.cos(angle)
        s = torch.sin(angle)

        C = 1 - c
        xs = x * s
        ys = y * s
        zs = z * s
        xC = x * C
        yC = y * C
        zC = z * C
        xyC = x * yC
        yzC = y * zC
        zxC = z * xC

        if len(angle.shape) == 1:
            result[..., 0, 0] = x * xC + c
            result[..., 0, 1] = xyC - zs
            result[..., 0, 2] = zxC + ys
            result[..., 1, 0] = xyC + zs
            result[..., 1, 1] = y * yC + c
            result[..., 1, 2] = yzC - xs
            result[..., 2, 0] = zxC - ys
            result[..., 2, 1] = yzC + xs
            result[..., 2, 2] = z * zC + c
        else:

            result[..., 0, 0] = (x * xC + c)[..., 0]  # .view((B,))
            result[..., 0, 1] = (xyC - zs)[..., 0]  # .view((B,))
            result[..., 0, 2] = (zxC + ys)[..., 0]  # .view((B,))
            result[..., 1, 0] = (xyC + zs)[..., 0]  # .view((B,))
            result[..., 1, 1] = (y * yC + c)[..., 0]  # .view((B,))
            result[..., 1, 2] = (yzC - xs)[..., 0]  # .view((B,))
            result[..., 2, 0] = (zxC - ys)[..., 0]  # .view((B,))
            result[..., 2, 1] = (yzC + xs)[..., 0]  # .view((B,))
            result[..., 2, 2] = (z * zC + c)[..., 0]  # .view((B,))

        return result

    @staticmethod
    def euler2mat(x, y, z):

        rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(x), -np.sin(x), 0],
            [0, np.sin(x), np.cos(x), 0],
            [0, 0, 0, 1],
        ])

        ry = np.array([
            [np.cos(y), 0, np.sin(y), 0],
            [0, 1, 0, 0],
            [-np.sin(y), 0, np.cos(y), 0],
            [0, 0, 0, 1],
        ])

        rz = np.array([
            [np.cos(z), -np.sin(z), 0, 0],
            [np.sin(z), np.cos(z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        return rx.dot(ry.dot(rz))

    def get_frame_transform(self, batchSize, bodyShapeRatio, parentBodyShapeRatio):
        '''
            Update the transform matrix of parentFrame and childFrame.
            Parent location and child location in opensim's frames are shape-dependent.
            The new location = the predefined location * the scaling ratio
        '''
        # device = bodyShapeRatio.device

        if batchSize > 0:
            parentFrame = self.originalparentMat.clone().unsqueeze(0).repeat(batchSize, 1, 1)
            childFrame = self.originalchildMat.clone().unsqueeze(0).repeat(batchSize, 1, 1)

            if self.parent is not None:
                parentFrame[..., :3, -1] = torch.einsum(' bj , j -> bj', parentBodyShapeRatio, self.parentLoc)
            else:
                parentFrame[..., :3, -1] = self.parentLoc

            childFrame[..., :3, -1] = torch.einsum(' bj , j -> bj', bodyShapeRatio, self.childLoc)

        else:
            parentFrame = self.originalparentMat.clone()
            childFrame = self.originalchildMat.clone()
            if self.parent is not None:
                parentFrame[..., :3, -1] = self.parentLoc * parentBodyShapeRatio
            else:
                parentFrame[..., :3, -1] = self.parentLoc
            childFrame[..., :3, -1] = self.childLoc * bodyShapeRatio

        childFrame = torch.inverse(childFrame)

        self.parentFrame = parentFrame
        self.childFrame = childFrame
        return
        # return parentFrame, childFrame

    def get_marker_position_in_ground(self, bodyShapeRatio):
        '''
            The relative location of marker to the attached body is shape-dependent.
            The new location = the predefined location * the scaling ratio
        '''

        '''
        device = markerLoc.device

        if len(markerLoc.shape) == 1 and len(bodyShapeRatio.shape) == 1:
            # single marker, no batch size
            markerLoc = markerLoc * bodyShapeRatio
            # markerLoc = torch.cat([markerLoc, torch.ones(1).float().to(device)], -1)
            return torch.matmul(self.transformInGround[:3, :3], markerLoc) + self.transformInGround[:3, -1]

        elif len(markerLoc.shape) == 2 and len(bodyShapeRatio.shape) == 1:
            # multiple markers, no batch size
            markerLoc = torch.einsum('nj,j->nj', markerLoc, bodyShapeRatio)
            markerLoc = torch.cat([markerLoc, torch.ones(markerLoc.shape[0], 1).float().to(device)], -1)
            markerLoc = torch.einsum('ij,nj->ni', self.transformInGround, markerLoc)
            return markerLoc[:, :-1]

        elif len(markerLoc.shape) == 3 and len(bodyShapeRatio.shape) == 2:
            # having batch size
            markerLoc = torch.einsum('bnj,bj->bnj', markerLoc, bodyShapeRatio)
            markerLoc = torch.cat([markerLoc, torch.ones(markerLoc.shape[0], markerLoc.shape[1], 1).float().to(device)],
                                  -1)
            markerLoc = torch.einsum('bij,bnj->bni', self.transformInGround, markerLoc)
            return markerLoc[:, :, :-1]

        elif len(markerLoc.shape) == 2 and len(bodyShapeRatio.shape) == 2:
            markerLoc = torch.einsum('nj,bj->bnj', markerLoc, bodyShapeRatio)
            markerLoc = torch.cat([markerLoc, torch.ones(markerLoc.shape[0], markerLoc.shape[1], 1).float().to(device)],
                                  -1)
            markerLoc = torch.einsum('bij,bnj->bni', self.transformInGround, markerLoc)
            return markerLoc[:, :, :-1]'''

        if len(self.transformInGround.shape) == 3:
            # have batch size
            markerLoc = torch.einsum('nj,bj->bnj', self.anchoredMarkers, bodyShapeRatio)
            markerLoc = torch.einsum('bij,bnj->bni', self.transformInGround[:, :3, :3],
                                     markerLoc) + self.transformInGround[:, :3, -1:].permute(0, 2, 1)
            return markerLoc
        elif len(self.transformInGround.shape) == 2:
            # no batch size
            markerLoc = torch.einsum('nj,j->nj', self.anchoredMarkers, bodyShapeRatio)
            markerLoc = torch.einsum('ij,nj->ni', self.transformInGround[:3, :3], markerLoc) + self.transformInGround[
                                                                                               :3, -1:].permute(1, 0)
            return markerLoc
        else:
            assert False

    def get_joint_position(self, child=True):

        if child or self.parent is None:
            if len(self.transformInGround.shape) == 2:
                Pos = torch.matmul(self.transformInGround, torch.inverse(self.childFrame)[:, -1])[:3].view(1, 3)
            elif len(self.transformInGround.shape) == 3:
                Pos = torch.einsum('bij,bj->bi', self.transformInGround, torch.inverse(self.childFrame)[:, :, -1])[...,
                      :3].view(self.transformInGround.shape[0], 1, 3)
            else:
                assert False
        else:
            if len(self.transformInGround.shape) == 2:
                Pos = torch.matmul(self.parent.transformInGround, self.parentFrame[:, -1])[:3].view(1, 3)
            elif len(self.transformInGround.shape) == 3:
                Pos = torch.einsum('bij,bj->bi', self.parent.transformInGround, self.parentFrame[:, :, -1])[...,
                      :3].view(self.transformInGround.shape[0], 1, 3)
            else:
                assert False
        return Pos

    '''
    def get_local_position(self, B, tensor):

        if B == 0:
            pos = torch.einsum('ij,j->i', self.R, tensor)[:3]
        else:
            pos = torch.einsum('bij,j->bi', self.R, tensor)[..., :3]

        return pos'''
