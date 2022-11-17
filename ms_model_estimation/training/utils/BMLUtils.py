import numpy as np
import torch

CAMERA_TABLE = {
    "PG1": {
        # from SMPL+H model space to the camera space
        "extrinsic": np.array([[0.99942183, -0.0305191, 0.01498661, -0.17723154],
                               [0.01303452, -0.06318672, -0.9979166, 1.03055751],
                               [0.03140247, 0.99753497, -0.06275239, 4.99931781]], dtype=np.float32),
        "intrinsic": np.array([[9.791788901134800e+02, 0., 408.0273103],
                               [0., 9.781017930509296e+02, 291.16967878],
                               [0, 0, 1.]], dtype=np.float32),
        "radialDisortionCoeff": np.array([-0.18236467, 0.18388686], dtype=np.float32),
        'res_w': 800,
        'res_h': 600,
        'center': np.array([408.0273103, 291.16967878], dtype=np.float32),
        'focal_length': np.array([979.17889011, 978.10179305], dtype=np.float32),
        'camera_params': np.array([979.17889011, 978.10179305, 408.0273103, 291.16967878, -0.18236467, 0.18388686],
                                  dtype=np.float32),

    },
    "PG2": {
        # from SMPL+H model space to the camera space
        "extrinsic": np.array([[0.04569302, -0.9988672, -0.01328404, -0.24368025],
                               [-0.17839144, 0.00492513, -0.98394727, 0.67257332],
                               [0.98289808, 0.04732928, -0.17796432, 4.05087267]], dtype=np.float32),

        "intrinsic": np.array([[980.04337094, 0., 392.27529092],
                               [0., 980.69881345, 309.91125524],
                               [0, 0, 1.]], dtype=np.float32),
        "radialDisortionCoeff": np.array([-0.22932292, 0.35464834], dtype=np.float32),
        'res_w': 800,
        'res_h': 600,
        'center': np.array([392.27529092, 309.91125524], dtype=np.float32),
        'focal_length': np.array([980.04337094, 980.69881345], dtype=np.float32),
        'camera_params': np.array([980.04337094, 980.69881345, 392.27529092, 309.91125524, -0.22932292, 0.35464834],
                                  dtype=np.float32),
    },
}

TrainingSubjects = [
    78,
    59,
    79,
    44,
    56,
    88,
    75,
    61,
    53,
    47,
    40,
    41,
    32,
    70,
    28,
    39,
    69,
    11,
    3,
    90,
    71,
    54,
    20,
    85,
    37,
    34,
    33,
    46,
    67,
    63,
    55,
    36,
    42,
    50,
    24,
    35,
    84,
    45,
    60,
    16,
    74,
    82,
    27,
    89,
    57,
    5,
    49,
    22,
    4,
    66,
    31,
    83,
    43,
    48,
    52,
    68,
    25,
    9,
    18,
    1,
    81,
    58,
    23,
    62,
    64,
    8,
    87]
ValidationSubjects = [14, 86, 15, 30]
TestSubjects = [2, 29, 19, 21, 51, 26, 7, 72, 77, 17, 65, 80, 6, 76, 10, 73, 13, 38]

PredictedMarkers = [

    # Pelvis
    'LPSI',
    'RPSI',
    'RASI',
    'LASI',

    # Spine
    'T12',
    'T10',
    'T8',
    "T1",

    # Neck
    'C7',

    # HIP
    'RKNE',
    'RKNI',
    'LKNE',
    'LKNI',

    # shrank
    'LANK',
    'RANK',
    'LANI',
    'RANI',

    # Foot
    'LHEE',
    'RHEE',
    'RTOE',
    'LTOE',
    'RMT1',
    'LMT1',
    'RMT5',
    'LMT5',

    # Calvicle
    'LSHO',
    'RSHO',

    # Thoratic
    'STRN',
    'CLAV',
    'RCLAV',
    'LCLAV',

    # Arm
    'LELB',
    'RELB',
    'LELBIN',
    'RELBIN',

    # Fore Arm
    'LIWR',
    'RIWR',
    'LOWR',
    'ROWR',

    # Head
    'LFHD',
    'RFHD',
    'LBHD',
    'RBHD',

    # Virtual Markers
    "Virtual_Ground_RAJC",
    "Virtual_Ground_LAJC",
    "Virtual_Ground_RFOOT",
    "Virtual_Ground_LFOOT",
    "Virtual_Ground_RTOE",
    "Virtual_Ground_LTOE",
    "Virtual_Ground_RMT1",
    "Virtual_Ground_LMT1",
    "Virtual_Ground_RMT5",
    "Virtual_Ground_LMT5",
    "Virtual_Ground_RHEE",
    "Virtual_Ground_LHEE",

]
smplHJoint = [
    "ROOT",
    "LHIPJ",
    "RHIPJ",
    "Spine1",
    "LKJC",
    "RKJC",
    "Spine2",
    "LAJC",
    "RAJC",
    "Spine3",
    "LFOOT",
    "RFOOT",
    'NECK',
    'LCOL',
    'RCOL',
    'HEAD',
    "LSHOULDER",
    "RSHOULDER",
    "LELC",
    "RELC",
    "LWRIST",
    "RWRIST",
]

Joints_Inside_Cylinder_Body = {

    "ROOT": "Torso",
    "RHIPJ": "Torso",
    "LHIPJ": "Torso",
    "Spine1": "Torso",
    "Spine2": "Torso",
    "Spine3": "Torso",
    'LCOL': "Torso",
    'RCOL': "Torso",
    'NECK': "Torso",

    "HEAD": "HEAD",

    # HIP
    'RKNE': "RightLeg",
    'RKNI': "RightLeg",
    'LKNE': "LeftLeg",
    'LKNI': "LeftLeg",

    # shrank
    'LANK': "LeftShrank",
    'RANK': "RightShrank",
    'LANI': "LeftShrank",
    'RANI': "RightShrank",

    # Foot
    'RTOE': "RightFOOT",
    'LTOE': "LeftFOOT",
    'RMT1': "RightFOOT",
    'LMT1': "LeftFOOT",
    'RMT5': "RightFOOT",
    'LMT5': "LeftFOOT",

    # Arm
    'LELB': "LeftArm",
    'RELB': "RightArm",
    'LELBIN': "LeftArm",
    'RELBIN': "RightArm",

    # Fore Arm
    'LIWR': "LeftForearm",
    'RIWR': "RightForearm",
    'LOWR': "LeftForearm",
    'ROWR': "RightForearm",

}
SELECTED_JOINTS = smplHJoint + PredictedMarkers
MIRROR_JOINTS = []
for joint in SELECTED_JOINTS:
    if joint[0] == "R" and joint != "ROOT":
        joint = "L" + joint[1:]
        MIRROR_JOINTS.append(SELECTED_JOINTS.index(joint))
    elif joint[0] == "L":
        joint = "R" + joint[1:]
        MIRROR_JOINTS.append(SELECTED_JOINTS.index(joint))
    elif "Ground_R" in joint:
        joint = joint.replace("Ground_R", "Ground_L")
        MIRROR_JOINTS.append(SELECTED_JOINTS.index(joint))
    elif "Ground_L" in joint:
        joint = joint.replace("Ground_L", "Ground_R")
        MIRROR_JOINTS.append(SELECTED_JOINTS.index(joint))
    else:
        MIRROR_JOINTS.append(SELECTED_JOINTS.index(joint))

PairMarkers = [
    ["ROOT", "Spine1"],
    ["Spine1", "Spine2"],
    ["Spine2", "Spine3"],
    ["ROOT", "LHIPJ"],
    ["ROOT", "RHIPJ"],
    ["LHIPJ", "LKJC"],
    ["RHIPJ", "RKJC"],
    ["LKJC", "LAJC"],
    ["RKJC", "RAJC"],
    ["LAJC", "LFOOT"],
    ["RAJC", "RFOOT"],
    ["Spine3", "LCOL"],
    ["Spine3", "RCOL"],
    ["Spine3", "NECK"],
    ["NECK", "HEAD"],
    ["LCOL", "LSHOULDER"],
    ["RCOL", "RSHOULDER"],
    ["LSHOULDER", "LELC"],
    ["RSHOULDER", "RELC"],
    ["LELC", "LWRIST"],
    ["RELC", "RWRIST"],
]

SymmetricPairMarkers = [
    [["ROOT", "LHIPJ"], ["ROOT", "RHIPJ"]],
    [["LHIPJ", "LKJC"], ["RHIPJ", "RKJC"]],
    [["LELC", "LAJC"], ["RELC", "RAJC"]],
    [["LAJC", "LFOOT"], ["RAJC", "RFOOT"]],
    [["LCOL", "LSHOULDER"], ["RCOL", "RSHOULDER"]],
    [["LSHOULDER", "LELC"], ["RSHOULDER", "RELC"]],
    [["LELC", "LWRIST"], ["RELC", "RWRIST"]],
    [["RIWR", "ROWR"], ["LIWR", "LOWR"]],
    [["LKNI", "LKNE"], ["RKNI", "RKNE"]],
    [["LANK", "LANI"], ["RANK", "RANI"]],
]


def project_to_camera_space(X, extrinsic):
    translation = extrinsic[..., :3, -1]
    rotation = extrinsic[..., :3, :3]

    if len(extrinsic.shape) == 2 and len(X.shape) == 2:
        XX = torch.einsum('ik,jk -> ji', rotation, X)
        XX = XX + translation
    elif len(extrinsic.shape) == 2 and len(X.shape) == 3:
        XX = torch.einsum('ik,bjk -> bji', rotation, X)
        XX = XX + translation
    elif len(extrinsic.shape) == 3 and len(X.shape) == 3:

        XX = torch.einsum('bik,bjk -> bji', rotation, X)
        XX = XX + translation
    else:
        assert False

    return XX


'''
def project_to_2d(X, intrinsic, distortion):
    XX = torch.div(X, X[..., -1:])
    
    if len(X.shape) == 2 and len(distortion.shape) == 1:
        # radialDisortion
        r2 = XX[..., 0] ** 2 + XX[..., 1] ** 2
        XX[..., 0] = XX[..., 0] * (1 + distortion[..., 0] * r2 + distortion[..., 1] * (r2 ** 2))
        XX[..., 1] = XX[..., 1] * (1 + distortion[..., 0] * r2 + distortion[..., 1] * (r2 ** 2))

    else:
        r2 = XX[..., 0:1] ** 2 + XX[..., 1:2] ** 2
        XX[..., 0:1] = XX[..., 0:1] * (1 + distortion[..., 0:1] * r2 + distortion[..., 1:] * (r2 ** 2))
        XX[..., 1:2] = XX[..., 1:2] * (1 + distortion[..., 0:1] * r2 + distortion[..., 1:] * (r2 ** 2))


    if len(intrinsic.shape) == 2 and len(X.shape) == 2:
        XXX = torch.einsum('ik,jk -> ji', intrinsic, XX)
    elif len(intrinsic.shape) == 2 and len(X.shape) == 3:
        XXX = torch.einsum('ik,bjk -> bji', intrinsic, XX)
    elif len(intrinsic.shape) == 3 and len(X.shape) == 3:
        XXX = torch.einsum('bik,bjk -> bji', intrinsic, XX)
    else:
        assert False

    XXX = XXX[..., :2]

    return XXX'''


def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 6
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2] ** 2, dim=len(XX.shape) - 1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2 ** 2), dim=len(r2.shape) - 1), dim=len(r2.shape) - 1,
                           keepdim=True)

    XXX = XX * radial

    return f * XXX + c
