from ms_model_estimation.models.config.config_default_img_training import get_cfg_defaults as get_img_defaults
from ms_model_estimation.models.utils.BMLUtils import TrainingSubjects, ValidationSubjects, TestSubjects, Joints_Inside_Cylinder_Body, \
    PairMarkers, SymmetricPairMarkers, PredictedMarkers, smplHJoint
from yacs.config import CfgNode as CN

_C = get_img_defaults()

# subjects ID
_C.TRAINING_SUBJECTS = TrainingSubjects
_C.VALID_SUBJECTS = ValidationSubjects
_C.TEST_SUBJECTS = TestSubjects

# Postfix for the model name
_C.POSTFIX = "BML_IMG"

_C.PREDICTION = CN()
_C.PREDICTION.POS_NAME = smplHJoint + PredictedMarkers
_C.PREDICTION.Joints_Inside_Cylinder_Body_LIST = [[k, v] for k, v in Joints_Inside_Cylinder_Body.items()]
# The number of prediction
_C.PREDICTION.NUM_JOINTS = len(smplHJoint)
_C.PREDICTION.NUM_MARKER = len(PredictedMarkers)
_C.PREDICTION.NUMPRED = _C.PREDICTION.NUM_JOINTS + _C.PREDICTION.NUM_MARKER

PAIR_MARKERS = []
for (a, b) in PairMarkers:
    PAIR_MARKERS.append([_C.PREDICTION.POS_NAME.index(a), _C.PREDICTION.POS_NAME.index(b)])
_C.PREDICTION.PAIR_MARKERS = PAIR_MARKERS

SY_MARKERS = []
for (a, b), (c, d) in SymmetricPairMarkers:
    SY_MARKERS.append(
        [_C.PREDICTION.POS_NAME.index(a), _C.PREDICTION.POS_NAME.index(b), _C.PREDICTION.POS_NAME.index(c),
         _C.PREDICTION.POS_NAME.index(d)])
_C.PREDICTION.SY_MARKERS = SY_MARKERS

# hyperparameter
_C.HYP = CN()
_C.HYP.POS = 1
_C.HYP.MARKER = 1


# Learning rate and the number of epoch
_C.TRAINING.TRAINING_STEPS = 2
_C.TRAINING.START_LR = [10 ** (-4), 3.33 * (10 ** (-6))]
_C.TRAINING.END_LR = [3.33 * (10 ** (-5)), (10 ** (-6))]
_C.TRAINING.EPOCH = [25, 2]

# use each n-th frame for validation and testing
_C.DATASET.USEDEACHFRAME = 1
_C.DATASET.TRAINING_USEDEACHFRAME = 2

# 2D projection loss
_C.TRAINING.LOSS2D = False
_C.TRAINING.LOSSTYPE = 1

# Paths and data folder
_C.PASCAL_PATH = None
_C.BML_FOLDER = None
_C.MODEL_FOLDER = None
_C.DATASET.INTERPOLATION.TEST = 'nearest'
_C.DATASET.ANTIALIAS.TEST = 1


def get_cfg_defaults():
    return _C.clone()


def update_config(config):
    if config.DATASET.OCCLUSION.PROB > 0:
        assert config.PASCAL_PATH is not None

    assert config.BML_FOLDER is not None
    assert config.MODEL_FOLDER is not None

    for subject in config.TEST_SUBJECTS:
        assert subject not in config.TRAINING_SUBJECTS and subject not in config.VALID_SUBJECTS
    for subject in config.VALID_SUBJECTS:
        assert subject not in config.TRAINING_SUBJECTS and subject not in config.TEST_SUBJECTS

    return config
