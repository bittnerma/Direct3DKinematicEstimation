import math
import trimesh
from ms_model_estimation.smplh.BodyModel import BodyModel
from ms_model_estimation.smplh.omni_tools import copy2cpu as c2c
from ms_model_estimation import BM_PATH_M, BM_PATH_N, BM_PATH_F, NUM_BETAS, NUM_DMPLS, DMPL_PATH_M, DMPL_PATH_N, DMPL_PATH_F
import torch
import numpy as np
from transforms3d.euler import euler2mat

COMP_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SMPLHModel:

    maleBodyModel = BodyModel(
        BM_PATH_M, num_betas=NUM_BETAS, num_dmpls=NUM_DMPLS, path_dmpl=DMPL_PATH_M
    ).to(COMP_DEVICE)


    neutralBodyModel = BodyModel(
        BM_PATH_N, num_betas=NUM_BETAS, num_dmpls=NUM_DMPLS, path_dmpl=DMPL_PATH_N
    ).to(COMP_DEVICE)

    femaleBodyModel = BodyModel(
        BM_PATH_F, num_betas=NUM_BETAS, num_dmpls=NUM_DMPLS, path_dmpl=DMPL_PATH_F
    ).to(COMP_DEVICE)

    # the average beta values in AMASS
    averageAmassShape = np.array([
        0.17528129, 0.19750505, 0.15343938, -0.20799128, 0.06805605,
        -0.06525175, 0.32532827, -0.90301506, 0.14500127, -0.27420981,
        -0.10608279, -0.02575251, 0.04653148, -0.30032272, 0.61680406,
        -0.49946938
    ])

    @staticmethod
    def get_smplH_vertices_position(
            bdata: np.array, defaultpose=True, frame=0, faces=False, translation=False,
            eulerAngle=None, handDown=False, DMPL=False, zeroShape=-1, gender="N"
    ):
        '''
        :param bdata: numpy dictionary.
        :param defaultpose:T-pose.
        :param frame: the index of the frame. 
        :param faces: return faces for visualization.
        :param translation: include translation.
        :param eulerAngle: rotate vertice positions with euler angles.
        :param handDown: Hand is straight down.
        :param DMPL: dynamic soft-tissue deformations. Default is False. This will influence the results.
        :param zeroShape: 0: set body parameters to 0, 1: use average body parameters, -1: use the data
        :param gender: N: neutral, F: female and M: male
        :return: joint positions and vertex positions.
        '''
        # Load Data
        fId = frame  # frame id of the mocap sequence
        root_orient = torch.Tensor(bdata['poses'][fId:fId + 1, :3]).to(
            COMP_DEVICE)  # controls the global root orientation
        pose_body = torch.Tensor(bdata['poses'][fId:fId + 1, 3:66]).to(COMP_DEVICE)  # controls the body
        pose_hand = torch.Tensor(bdata['poses'][fId:fId + 1, 66:]).to(COMP_DEVICE)  # controls the finger articulation
        betas = torch.Tensor(bdata['betas'][:NUM_BETAS][np.newaxis]).to(COMP_DEVICE)  # controls the body shape
        dmpls = torch.Tensor(bdata['dmpls'][fId:fId + 1]).to(COMP_DEVICE)  # controls soft tissue dynamics
        trans = bdata['trans'][fId:fId + 1, :3]

        if not DMPL:
            dmpls = torch.zeros(dmpls.shape).to(COMP_DEVICE)

        if zeroShape == 0:
            # Make shape parameters zero
            betas = torch.zeros(betas.shape).to(COMP_DEVICE)

        elif zeroShape == 1:
            # the average shape
            betas = SMPLHModel.averageAmassShape
            betas = np.reshape(betas, (1, NUM_BETAS))
            betas = torch.Tensor(betas).to(COMP_DEVICE)

        elif zeroShape != -1:
            assert False

        if defaultpose:
            # Set to static pose
            pose_body = torch.zeros(pose_body.shape).to(COMP_DEVICE)
            root_orient = torch.zeros(root_orient.shape).to(COMP_DEVICE)
            pose_hand = torch.zeros(pose_hand.shape).to(COMP_DEVICE)
            dmpls = torch.zeros(dmpls.shape).to(COMP_DEVICE)
            if handDown:
                pose_body[0, 47] = -90 / 180 * math.pi
                pose_body[0, 50] = 90 / 180 * math.pi

        return SMPLHModel.get_body_model_output(
            pose_body, pose_hand, betas, dmpls, root_orient,
            trans, translation=translation, eulerAngle=eulerAngle, faces=faces, gender=gender
        )

    @staticmethod
    def get_body_model_output(pose_body, pose_hand, betas, dmpls, root_orient, trans=None, translation=False,
                              eulerAngle=None, faces=False, gender="N"):

        if gender == "N":
            body = SMPLHModel.neutralBodyModel(pose_body=pose_body, pose_hand=pose_hand, betas=betas, dmpls=dmpls,
                                                root_orient=root_orient)
            f = SMPLHModel.neutralBodyModel.f
        elif gender == "F":
            body = SMPLHModel.femaleBodyModel(pose_body=pose_body, pose_hand=pose_hand, betas=betas, dmpls=dmpls,
                                               root_orient=root_orient)
            f = SMPLHModel.femaleBodyModel.f
        elif gender == "M":
            body = SMPLHModel.maleBodyModel(pose_body=pose_body, pose_hand=pose_hand, betas=betas, dmpls=dmpls,
                                             root_orient=root_orient)
            f = SMPLHModel.maleBodyModel.f
        else:
            raise Exception(" Gender is not defined!")

        vertices = body.v[0].detach().cpu().numpy().squeeze()
        joints = body.Jtr[0].detach().cpu().numpy().squeeze()
        
        # include translation
        if translation:
            assert trans is not None
            vertices += trans
            joints += trans
        
        # rotate vertices
        if eulerAngle is not None:
            assert len(eulerAngle) == 3
            rotationMat = euler2mat(eulerAngle[0], eulerAngle[1], eulerAngle[2])
            vertices = np.dot(rotationMat, vertices.T).T
            joints = np.dot(rotationMat, joints.T).T

        if faces:
            return vertices, joints, c2c(f)
        else:
            return vertices, joints

    @staticmethod
    def get_height(bdata: np.array, millimeter=True, gender="N"):

        vertices, _ = SMPLHModel.get_smplH_vertices_position(bdata, gender=gender)

        height = max(vertices[:, 1]) - min(vertices[:, 1])
        height = 1000 * height if millimeter else height
        return height

    @staticmethod
    def get_time_range(bdata: np.array, videoFrameRate):
        NumFrames = bdata['poses'].shape[0]

        return "0 " + str(1 / videoFrameRate * NumFrames)

    @staticmethod
    def get_assumed_body_weight(bdata, gender="N"):

        vertices, _, faces = SMPLHModel.get_smplH_vertices_position(bdata, faces=True, defaultpose=True, gender=gender)
        tri_mesh = trimesh.Trimesh(vertices, faces)

        # 0.07745817923272397 is the volume under the average beta paramters
        # 78.11652702058652 is the assumed body weight under the average beta paramters
        bodyWeight = 78.11652702058652 / 0.07745817923272397 * tri_mesh.volume

        return bodyWeight
