from ms_model_estimation.models.config.config_default_img_training import get_cfg_defaults as get_img_defaults
from ms_model_estimation.models.MPIIUtils import create_table, ARMS_PELVIS_LEGS
from yacs.config import CfgNode as CN
import h5py

_C = get_img_defaults()

# Postfix for the model name
_C.POSTFIX = "H36M_IMG"

# The number of prediction
_C.TRAINING.NUMPRED = 17

# protocal for training
_C.TRAINING.PROTOCAL = 1

# ratio for validation set
_C.TRAINING.VALIDATIONRATIO = 0.05

# hyperparameter
_C.TRAINING.HYP = [1, 0.1]

# Learning rate and the number of epoch
_C.TRAINING.TRAINING_STEPS = 2
_C.TRAINING.START_LR = [10 ** (-4), 3.33 * (10 ** (-6))]
_C.TRAINING.END_LR = [3.33 * (10 ** (-5)), (10 ** (-6))]
_C.TRAINING.EPOCH = [25, 2]

# use each n-th frame for validation and testing
_C.DATASET.USEDEACHFRAME = 64
_C.DATASET.TRAINING_USEDEACHFRAME = 32

# perspective correction
_C.DATASET.PERSPECTIVE_CORRECTION = False

# train with MPII and Human 3.6M together
_C.TRAINING.MIX2D = True
if _C.TRAINING.MIX2D:
    _C.TRAINING.H36M_BATCHSIZE = _C.TRAINING.BATCHSIZE // 2
    _C.TRAINING.MPII_BATCHSIZE = _C.TRAINING.BATCHSIZE // 2
else:
    _C.TRAINING.H36M_BATCHSIZE = _C.TRAINING.BATCHSIZE
_C.TRAINING.MPII_USED_INDEX = ARMS_PELVIS_LEGS
_C.TRAINING.MAPPING_H36M_INDEX_TO_MPII = create_table(usedJoints=_C.TRAINING.MPII_USED_INDEX)

# Use all of H36M data in ine epoch. Since MPII data is less than H36M data, one epoch will have same MPII data multiple times.
_C.TRAINING.ALL_H36M = False

_C.TRAINING.H36M_2DPROJECTION = False

# 2D projection loss
_C.TRAINING.LOSS2D = _C.TRAINING.H36M_2DPROJECTION or _C.TRAINING.MIX2D
_C.TRAINING.LOSSTYPE = 1


# Paths and data folder
_C.PASCAL_PATH = None
_C.H36M_FOLDER = None
_C.MPII_FOLDER = None
_C.MODEL_FOLDER = None
_C.CAMERA_PATH = None
_C.STARTMODELPATH = None

def get_cfg_defaults():
    return  _C.clone()

def update_config(config):
    config.TRAINING.LOSS2D = config.TRAINING.H36M_2DPROJECTION or config.TRAINING.MIX2D
    config.TRAINING.MAPPING_H36M_INDEX_TO_MPII = create_table(usedJoints=config.TRAINING.MPII_USED_INDEX)
    config.TRAINING.H36M_BATCHSIZE = config.TRAINING.BATCHSIZE // 2
    config.TRAINING.MPII_BATCHSIZE = config.TRAINING.BATCHSIZE // 2

    if config.DATASET.OCCLUSION.PROB > 0:
        assert config.PASCAL_PATH is not None

    if config.TRAINING.MIX2D:
        assert config.MPII_FOLDER is not None

    assert config.H36M_FOLDER is not None
    assert config.MODEL_FOLDER is not None
    assert config.CAMERA_PATH is not None

    config.TRAINING.START_LR = [float(lr) for lr in config.TRAINING.START_LR]
    config.TRAINING.END_LR = [float(lr) for lr in config.TRAINING.END_LR]


    return config