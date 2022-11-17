'''
This is the implemtation of "Revisiting the Continuity of Rotation Representations in Neural Networks"
paper url : https://arxiv.org/abs/2006.06234
'''
import torch


class Representation6D:

    @staticmethod
    def get_rotation_matrix(c1, c2):
        c3 = torch.cross(c1, c2, dim=-1)
        return torch.stack([c1, c2, c3], dim=-1)

    @staticmethod
    def convert_6d_vectors_to_3d_vectors(x):
        assert x.shape[-1] == 6

        c1 = x[..., :3] / torch.linalg.norm(x[..., :3], ord=2, dim=-1, keepdim=True)

        c2 = x[..., 3:]
        temp = x[..., :3] / torch.linalg.norm(x[..., :3], ord=2, dim=-1, keepdim=True)
        temp2 = ((c1 * c2).sum(axis=-1))
        for i in range(3):
            temp[..., i] *= temp2
        c2 = c2 - temp
        c2 = c2 / torch.linalg.norm(c2, ord=2, dim=-1, keepdim=True)

        return c1, c2

    @staticmethod
    def convert_6d_vectors_to_mat(x):
        c1, c2 = Representation6D.convert_6d_vectors_to_3d_vectors(x)
        mat = Representation6D.get_rotation_matrix(c1, c2)
        return mat

    @staticmethod
    def get_projection_result(rot, X):
        assert rot.shape[-2:] == (3, 3)
        return torch.matmul(rot, X)
