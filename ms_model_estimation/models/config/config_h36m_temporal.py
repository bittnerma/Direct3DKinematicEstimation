from ms_model_estimation.models.config.config_h36m import get_cfg_defaults as get_img_defaults
from yacs.config import CfgNode as CN

_C = get_img_defaults()

# Postfix for the model name
_C.POSTFIX = "H36M_TEMPORAL"

_C.TRAINING.BATCHSIZE = 128
_C.TRAINING.EVALUATION_BATCHSIZE = 128

_C.TRAINING.TRAINING_STEPS = 1
_C.TRAINING.EPOCH = [80]
_C.TRAINING.START_LR = [0.001]
_C.TRAINING.END_LR = [0.0000182]
#_C.TRAINING.LR_DECAY_RATE = 0.966
_C.TRAINING.INITIAL_MOMENTUM = 0.1
_C.TRAINING.END_MOMENTUM = 0.001
_C.TRAINING.RECEPTIVE_FIELD = 243

# _C.TRAINING.USE_POSE_ESTIMATION = [False]
_C.TRAINING.REFINE = False
_C.TRAINING.CAUSAL = False
_C.TRAINING.USE_2D = True
_C.TRAINING.USE_GT = True

_C.TRAINING.AWARED_OCCLUSION = CN()
_C.TRAINING.AWARED_OCCLUSION.GT = False
_C.TRAINING.AWARED_OCCLUSION.USE = False
_C.TRAINING.AWARED_OCCLUSION.USE_PREDICTION = False


_C.DATASET.TRAINING_USEDEACHFRAME = 1
_C.DATASET.USEDEACHFRAME = 1
# Paths and data folder
_C.MODEL_FOLDER = None
_C.H36M_FOLDER = None
_C.CAMERA_PATH = None

# _C.PRETRAINED_POSE_MODEL = None

_C.HYP = CN()
_C.HYP.POS = 1
_C.HYP.MARKER = 1

def get_cfg_defaults():
    return  _C.clone()

def update_config(config):

    assert config.MODEL_FOLDER is not None
    assert config.H36M_FOLDER is not None
    assert config.CAMERA_PATH is not None

    return config
