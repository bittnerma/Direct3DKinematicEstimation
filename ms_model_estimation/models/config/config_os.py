from ms_model_estimation.models.config.config_bml import get_cfg_defaults as get_img_defaults
from ms_model_estimation.models.utils.OSUtils import PredictedBones, PredictedOSJoints, PredictedCoordinates, LeafJoints, \
    MIRROR_BONES, PredictedMarkersType, PredictedMarkers, Joints_Inside_Cylinder_Body
from yacs.config import CfgNode as CN

_C = get_img_defaults()

# Postfix for the model name
_C.POSTFIX = "OS"

# model architecture

# use layers before average pooling
_C.MODEL.FULLCONV = True

# more convolutional layers between the last layer of resnet model
# and the opensim transition layer
_C.MODEL.CONV_LAYER = 1
_C.MODEL.CONV_CHANNEL_SIZE = [2048]
_C.MODEL.INTERMEDIATE_POSE_ESTIMATION = False
_C.MODEL.NUM_PRED = len(PredictedOSJoints) + len(PredictedMarkers) + len(LeafJoints)
_C.MODEL.NUMPRED = len(PredictedOSJoints) + len(PredictedMarkers) + len(LeafJoints)

# Use simulation for training
_C.TRAINING.USE_OS_SIMULATION = False

# Predict Marker
_C.TRAINING.MARKER_PREDICTION = True

# coordinate angle weights
_C.TRAINING.COORDINATE_MAX_WEIGHT = 2

# use body weights
_C.TRAINING.USE_BODY_WEIGHTS = False

# marker weights
_C.TRAINING.MARKER_TYPE = PredictedMarkersType
_C.TRAINING.MARKER_WEIGHT = [i / 10 for i in PredictedMarkersType]

# hyperparameter
_C.HYP = CN()
_C.HYP.POS = 1.0
_C.HYP.MARKER = 2.0
_C.HYP.ANGLE = 0.01
_C.HYP.BODY = 0.1
_C.HYP.SY_BODY = 0.05
_C.HYP.INTERMEDIATE_POS = 1.5
_C.HYP.INTERMEDIATE_MARKER = 1.5
_C.HYP.ROTMAT = 0.1

# Learning rate and the number of epoch
_C.TRAINING.TRAINING_STEPS = 2
_C.TRAINING.START_LR = [5 * (10 ** (-4)), 3.33 * (10 ** (-6))]
_C.TRAINING.END_LR = [3.33 * (10 ** (-5)), (10 ** (-6))]
_C.TRAINING.EPOCH = [28, 2]

# use each n-th frame for validation and testing
_C.DATASET.USEDEACHFRAME = 1
_C.DATASET.TRAINING_USEDEACHFRAME = 2

# awared occlusion
_C.TRAINING.AWARED_OCCLUSION = CN()
_C.TRAINING.AWARED_OCCLUSION.GT = False
_C.TRAINING.AWARED_OCCLUSION.USE = False

# prediction
_C.PREDICTION = CN()
_C.PREDICTION.BODY = PredictedBones
_C.PREDICTION.JOINTS = PredictedOSJoints
_C.PREDICTION.COORDINATES = PredictedCoordinates
_C.PREDICTION.MARKER = PredictedMarkers
_C.PREDICTION.LEAFJOINTS = LeafJoints
_C.PREDICTION.MIRROR_BONES = MIRROR_BONES
_C.PREDICTION.BODY_SCALE_MAPPING = [[0, 1, 2],
                                    [10, 3, 9],
                                    [10, 3, 9],
                                    [10, 3, 9],
                                    [10, 3, 9],
                                    [10, 3, 9],
                                    [10, 3, 9],
                                    [10, 4, 9],
                                    [10, 4, 9],
                                    [10, 5, 9],
                                    [10, 5, 9],
                                    [10, 6, 9],
                                    [10, 6, 9],
                                    [10, 6, 9],
                                    [10, 6, 9],
                                    [10, 6, 9],
                                    [10, 6, 9],
                                    [10, 6, 9],
                                    [11, 7, 9],
                                    [16, 14, 16],
                                    [20, 18, 20],
                                    [20, 18, 20],
                                    [22, 23, 24],
                                    [28, 30, 30],
                                    [17, 15, 17],
                                    [21, 19, 21],
                                    [21, 19, 21],
                                    [25, 26, 27],
                                    [29, 31, 31],
                                    [11, 7, 9],
                                    [11, 7, 9],
                                    [11, 7, 9],
                                    [11, 7, 9],
                                    [11, 7, 9],
                                    [11, 7, 9],
                                    [12, 8, 13],
                                    [12, 8, 13],
                                    [32, 33, 34],
                                    [37, 35, 41],
                                    [39, 36, 41],
                                    [45, 43, 45],
                                    [49, 47, 49],
                                    [49, 47, 49],
                                    [38, 35, 42],
                                    [40, 36, 42],
                                    [46, 44, 46],
                                    [50, 48, 50],
                                    [50, 48, 50]]
_C.PREDICTION.BODY_SCALE_UNIQUE_NUMS = 51
_C.PREDICTION.BODY_AVG_VALUE = [1.21267658, 1.19278473, 1.09221309, 0.82231784, 0.65399921,
                                0.86740023, 1.23414558, 1.13399419, 1.00263798, 1.01988775,
                                1.04674897, 0.92795962, 0.80902216, 0.94262153, 0.96230629,
                                0.95413619, 1.04046246, 1.03910667, 0.68939832, 0.71175453,
                                0.74281865, 0.76454479, 0.9076589, 1.02388528, 0.64942223,
                                0.89473501, 0.89226717, 0.61283192, 0.7215803, 0.68092433,
                                1.08504841, 1.17846495, 1.14104316, 1.14461103, 0.74801221,
                                0.92795962, 1.14461103, 0.89621243, 0.8843008, 1.13165459,
                                1.13165459, 0.73835266, 0.74087435, 1.07924646, 1.08077973,
                                1.01209745, 0.99989337, 1.0244289, 1.03203484, 0.82801554,
                                0.7559647]
_C.PREDICTION.BODY_AMPITUDE_VALUE = [0.38330534, 0.11688988, 0.27979052, 0.18767838, 0.12217734,
                                     0.17559392, 0.25814359, 0.19320878, 0.09488225, 0.23721306,
                                     0.3308586, 0.22396367, 0.08659548, 0.000001, 0.1634575,
                                     0.18397586, 0.23942574, 0.23136499, 0.14410597, 0.14576365,
                                     0.17185328, 0.18178275, 0.15676707, 0.19965802, 0.12167494,
                                     0.14225374, 0.19253547, 0.08705724, 0.13519439, 0.09673029,
                                     0.16227082, 0.25709417, 0.27539152, 0.21485549, 0.11401759,
                                     0.22396367, 0.21485549, 0.1838405, 0.18214334, 0.43287996,
                                     0.43287996, 0.13003099, 0.13036308, 0.2175328, 0.20085876,
                                     0.30206298, 0.3034652, 0.2025831, 0.20302752, 0.1691131,
                                     0.22653959]
_C.PREDICTION.BODY_WEIGHTS = [[2., 2., 2.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [2., 1., 1.],
                              [2., 1., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [2., 1., 1.],
                              [2., 1., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 1., 2.],
                              [1., 1., 2.],
                              [2., 1., 1.],
                              [1., 1., 2.],
                              [1., 1., 2.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 1., 2.],
                              [1., 1., 2.],
                              [1., 2., 1.],
                              [1., 2., 1.],
                              [1., 2., 1.]]
_C.PREDICTION.NUM_JOINTS = len(PredictedOSJoints) + len(LeafJoints)
_C.PREDICTION.NUM_MARKERS = len(PredictedMarkers)
_C.PREDICTION.NUM_SMPL_JOINTS = 22
_C.PREDICTION.POS_NAME = PredictedOSJoints + LeafJoints + PredictedMarkers
_C.PREDICTION.Joints_Inside_Cylinder_Body_LIST = [[k, v] for k, v in Joints_Inside_Cylinder_Body.items()]

# Model Architecture
_C.LAYER = CN()

# opensim transition layer
_C.LAYER.OS_TRANSIT = CN()

# 0 : tanh , 1 : sigmoid , 2 : soft argmax
_C.LAYER.OS_TRANSIT.ACT_TYPE = 1
_C.LAYER.OS_TRANSIT.BN = False

# Loss
_C.LOSS = CN()

_C.LOSS.POS = CN()
_C.LOSS.POS.USE = True
_C.LOSS.POS.TYPE = 1
_C.LOSS.POS.BETA = 0.04

_C.LOSS.MARKER = CN()
_C.LOSS.MARKER.USE = True
_C.LOSS.MARKER.TYPE = 1
_C.LOSS.MARKER.BETA = 0.04

# coordinate angle
_C.LOSS.ANGLE = CN()
_C.LOSS.ANGLE.USE = True
_C.LOSS.ANGLE.USEMASK = True
_C.LOSS.ANGLE.MASKMINVALUE = 0
_C.LOSS.ANGLE.TYPE = 3
_C.LOSS.ANGLE.EVAL_TYPE = 1
_C.LOSS.ANGLE.BETA = 5
_C.LOSS.ANGLE.H = 0.1

# body scale
_C.LOSS.BODY = CN()
_C.LOSS.BODY.USE = True
_C.LOSS.BODY.TYPE = 3
_C.LOSS.BODY.EVAL_TYPE = 1
_C.LOSS.BODY.BETA = 0.05
_C.LOSS.BODY.H = 0.1

# symmetric body scale
_C.LOSS.ROTMAT = CN()
_C.LOSS.ROTMAT.USE = False
_C.LOSS.ROTMAT.TYPE = 1
_C.LOSS.ROTMAT.EVAL_TYPE = 1
_C.LOSS.ROTMAT.BETA = 0.02

# symmetric body scale
_C.LOSS.SY_BODY = CN()
_C.LOSS.SY_BODY.USE = False
_C.LOSS.SY_BODY.TYPE = 1
_C.LOSS.SY_BODY.EVAL_TYPE = 1

# Paths and data folder
_C.PASCAL_PATH = None
_C.BML_FOLDER = None
_C.MODEL_FOLDER = None
_C.PCA_PATH = None
_C.RESNETPATH = None

# Use PCA
_C.TRAINING.PCA = False
_C.TRAINING.PCA_COMPONENTS = 51

# Geometric Augmentation
_C.DATASET.GEOM.AUG = True
_C.DATASET.GEOM.HFLIP = False

_C.DATASET.INTERPOLATION.TEST = 'nearest'
_C.DATASET.ANTIALIAS.TEST = 1


def get_cfg_defaults():
    return _C.clone()


def update_config(config):
    if config.DATASET.OCCLUSION.PROB > 0:
        assert config.PASCAL_PATH is not None

    assert config.BML_FOLDER is not None
    assert config.MODEL_FOLDER is not None

    if config.TRAINING.PCA:
        assert config.PCA_PATH is not None

    config.TRAINING.EPOCH = [int(c) for c in config.TRAINING.EPOCH]
    config.TRAINING.START_LR = [float(c) for c in config.TRAINING.START_LR]
    config.TRAINING.END_LR = [float(c) for c in config.TRAINING.END_LR]

    config.MODEL.CONV_CHANNEL_SIZE = [int(i) for i in config.MODEL.CONV_CHANNEL_SIZE]

    return config
