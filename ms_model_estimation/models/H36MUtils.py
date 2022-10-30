# The script refers to the paper, "3D human pose estimation in video with temporal convolutions and
# semi-supervised training".
# script url: https://github.com/facebookresearch/VideoPose3D/tree/master/common
# paper url : https://arxiv.org/abs/1811.11742
import h5py
import torch
import numpy as np

USED_JOINTS = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
SELECTED_JOINTS = [
    "PELVIS",
    "RHIP",
    "RKNE",
    "RANK",
    "LHIP",
    "LKNE",
    "LANK",
    "BACK",
    "THORAX",
    "NECK",
    "HEAD",
    "LSHOULDER",
    "LELB",
    "LWRIST",
    "RSHOULDER",
    "RELB",
    "RWRIST",
]
MIRROR_JOINTS = []
for joint in SELECTED_JOINTS:
    if joint[0] == "R":
        joint = "L" + joint[1:]
        MIRROR_JOINTS.append(SELECTED_JOINTS.index(joint))
    elif joint[0] == "L":
        joint = "R" + joint[1:]
        MIRROR_JOINTS.append(SELECTED_JOINTS.index(joint))
    else:
        MIRROR_JOINTS.append(SELECTED_JOINTS.index(joint))

assert len(USED_JOINTS) == len(SELECTED_JOINTS)

CONNECTION_TABLE = [
    ["PELVIS", "RHIP"],
    ["PELVIS", "LHIP"],
    ["RHIP", "RKNE"],
    ["LHIP", "LKNE"],
    ["RKNE", "RANK"],
    ["LKNE", "LANK"],
    ["PELVIS", "BACK"],
    ["BACK", "THORAX"],
    ["THORAX", "NECK"],
    ["NECK", "HEAD"],
    ["THORAX", "LSHOULDER"],
    ["THORAX", "RSHOULDER"],
    ["RSHOULDER", "RELB"],
    ["LSHOULDER", "LELB"],
    ["RELB", "RWRIST"],
    ["LELB", "LWRIST"],
]
CONNECTION_TABLE_INDEX_LIST = []
for k, v in CONNECTION_TABLE:
    CONNECTION_TABLE_INDEX_LIST.append([SELECTED_JOINTS.index(k), SELECTED_JOINTS.index(v)])

CAMERA_NAMES = ['54138969', '55011271', '58860488', '60457274']

INTRINSIC_TABLE = {
    '54138969': {'id': '54138969',
                 'center': np.array([0.02508307, 0.02890298], dtype=np.float32),
                 'focal_length': np.array([2.290099, 2.2875624], dtype=np.float32),
                 'radial_distortion': np.array([-0.20709892, 0.24777518, -0.00307515], dtype=np.float32),
                 'tangential_distortion': np.array([-0.0009757, -0.00142447], dtype=np.float32),
                 'res_w': 1000,
                 'res_h': 1002,
                 'azimuth': np.array(70., dtype=np.float32),
                 'intrinsic': np.array([2.2900989e+00, 2.2875624e+00, 2.5083065e-02, 2.8902981e-02,
                                        -2.0709892e-01, 2.4777518e-01, -3.0751503e-03, -9.7569887e-04,
                                        -1.4244716e-03], dtype=np.float32),
                 'localCenter': np.array([-3.6321580e-08, 3.5390258e-08, 0.0000000e+00]),
                 'imageCenterRay': np.array([-0.01095131, -0.01263308, 0.99986017]),

                 },
    '55011271': {'id': '55011271',
                 'center': np.array([0.01769722, 0.01612985], dtype=np.float32),
                 'focal_length': np.array([2.2993512, 2.2951834], dtype=np.float32),
                 'radial_distortion': np.array([-0.19421363, 0.24040854, 0.00681998], dtype=np.float32),
                 'tangential_distortion': np.array([-0.00161903, -0.00274089], dtype=np.float32),
                 'res_w': 1000,
                 'res_h': 1000,
                 'azimuth': np.array(-70., dtype=np.float32),
                 'intrinsic': np.array([2.2993512e+00, 2.2951834e+00, 1.7697215e-02, 1.6129851e-02,
                                        -1.9421363e-01, 2.4040854e-01, 6.8199756e-03, -1.6190266e-03,
                                        -2.7408944e-03], dtype=np.float32),
                 'localCenter': np.array([-5.4016709e-08, 4.4703484e-08, 0.0000000e+00]),
                 'imageCenterRay': np.array([-0.0076959, -0.00702697, 0.99994564]),
                 },
    '58860488': {'id': '58860488',
                 'center': np.array([0.03963172, 0.00280535], dtype=np.float32),
                 'focal_length': np.array([2.2982814, 2.297598], dtype=np.float32),
                 'radial_distortion': np.array([-0.20833819, 0.255488, -0.0024605], dtype=np.float32),
                 'tangential_distortion': np.array([0.00148439, -0.00076], dtype=np.float32),
                 'res_w': 1000,
                 'res_h': 1000,
                 'azimuth': np.array(110., dtype=np.float32),
                 'intrinsic': np.array([2.2982814e+00, 2.2975979e+00, 3.9631724e-02, 2.8053522e-03,
                                        -2.0833819e-01, 2.5548801e-01, -2.4604974e-03, 1.4843870e-03,
                                        -7.5999933e-04], dtype=np.float32),
                 'localCenter': np.array([4.8428774e-08, -1.3832003e-05, 0.0000000e+00]),
                 'imageCenterRay': np.array([-0.01724347, -0.00120171, 0.99985063])
                 },
    '60457274': {'id': '60457274',
                 'center': np.array([0.02993643, 0.00176403], dtype=np.float32),
                 'focal_length': np.array([2.2910228, 2.289548], dtype=np.float32),
                 'radial_distortion': np.array([-0.19838409, 0.21832368, -0.00894781], dtype=np.float32),
                 'tangential_distortion': np.array([-0.00058721, -0.00181336], dtype=np.float32),
                 'res_w': 1000,
                 'res_h': 1002,
                 'azimuth': np.array(-110., dtype=np.float32),
                 'intrinsic': np.array([2.2910228e+00, 2.2895479e+00, 2.9936433e-02, 1.7640333e-03,
                                        -1.9838409e-01, 2.1832368e-01, -8.9478074e-03, -5.8720558e-04,
                                        -1.8133620e-03], dtype=np.float32),
                 'localCenter': np.array([2.2351742e-08, -2.7279835e-05, 0.0000000e+00]),
                 'imageCenterRay': np.array([-1.3065962e-02, -7.4283913e-04, 9.9991441e-01])
                 }
}


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]


def image_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Reverse camera frame normalization
    return (X + [1, h / w]) * w / 2


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
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2] ** 2, dim=len(XX.shape) - 1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2 ** 2, r2 ** 3), dim=len(r2.shape) - 1), dim=len(r2.shape) - 1,
                           keepdim=True)
    tan = torch.sum(p * XX, dim=len(XX.shape) - 1, keepdim=True)

    XXX = XX * (radial + tan) + p * r2

    return f * XXX + c


def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result


def read_bbox(gtBBoxPath):
    with h5py.File(gtBBoxPath, 'r') as f:
        refs = f['Masks'][:, 0]
        bboxes = np.empty([len(refs), 4], dtype=np.float32)
        for i, ref in enumerate(refs):
            mask = np.array(f['#refs#'][ref]).T
            try:
                xmin, xmax = np.nonzero(np.any(mask, axis=0))[0][[0, -1]]
                ymin, ymax = np.nonzero(np.any(mask, axis=1))[0][[0, -1]]
                bboxes[i] = [xmin, ymin, xmax - xmin + 1, ymax - ymin + 1]
            except IndexError:
                bboxes[i] = [0, 0, 0, 0]

        return bboxes


def correct_S9_bbox_and_position(cdfFile, cam, positions, bboxes):
    if 'S9' in cdfFile and ('SittingDown 1' in cdfFile or 'Waiting 1' in cdfFile or 'Greeting.' in cdfFile):
        positive = 1 if ('58860488' in cdfFile or '54138969' in cdfFile) else -1

        positionsWrong = positions.copy()
        positions[:, :, 0] += 0.2 * positive

        pos_2d_wrong = wrap(project_to_2d, positionsWrong, cam['intrinsic'], unsqueeze=True)
        pos_2d_pixel_space_wrong = image_coordinates(pos_2d_wrong, w=cam['res_w'], h=cam['res_h'])

        pos_2d = wrap(project_to_2d, positions, cam['intrinsic'], unsqueeze=True)
        pos_2d_pixel_space = image_coordinates(pos_2d, w=cam['res_w'], h=cam['res_h'])

        diff = (pos_2d_pixel_space[:, 0, 0] - pos_2d_pixel_space_wrong[:, 0, 0])

        bboxes[:, 0] += diff
        bboxes[:, 2] += diff

    return positions, bboxes


def correct_S9(cdfFile, positions):
    if 'S9' in cdfFile and ('SittingDown 1' in cdfFile or 'Waiting 1' in cdfFile or 'Greeting.' in cdfFile):
        positive = 1 if ('58860488' in cdfFile or '54138969' in cdfFile) else -1
        positions[:, :, 0] += 0.2 * positive
        print(f'Correct {cdfFile} plus {0.2 * positive} meter in x-axis.')
    return positions


def correct_boxes(bboxes, path, world_coords, camera):
    """Three activties for subject S9 have erroneous bounding boxes, they are horizontally shifted.
    This function corrects them. Use --dataset=h36m-incorrect-S9 to use the erroneous annotation."""

    def correct_image_coords(bad_imcoords):
        root_depths = camera.world_to_camera(world_coords[:, 0])[:, 2:]
        bad_worldcoords = camera.image_to_world(bad_imcoords, camera_depth=root_depths)
        good_worldcoords = bad_worldcoords + np.array([-200, 0, 0])
        good_imcoords = camera.world_to_image(good_worldcoords)
        return good_imcoords

    if 'S9' in path and ('SittingDown 1' in path or 'Waiting 1' in path or 'Greeting.' in path):
        toplefts = correct_image_coords(bboxes[:, :2])
        bottomrights = correct_image_coords(bboxes[:, :2] + bboxes[:, 2:])
        return np.concatenate([toplefts, bottomrights - toplefts], axis=-1)

    return bboxes


def correct_world_coords(coords, path):
    """Three activties for subject S9 have erroneous coords, they are horizontally shifted.
    This corrects them. Use --dataset=h36m-incorrect-S9 to use the erroneous annotation."""
    if 'S9' in path and ('SittingDown 1' in path or 'Waiting 1' in path or 'Greeting.' in path):
        coords = coords.copy()
        coords[:, :, 0] -= 200
    return coords
