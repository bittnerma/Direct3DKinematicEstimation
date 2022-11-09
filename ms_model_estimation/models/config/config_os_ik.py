from ms_model_estimation.models.config.config_os import get_cfg_defaults as get_img_defaults
from yacs.config import CfgNode as CN

_C = get_img_defaults()

# Postfix for the model name
_C.POSTFIX = "OS_IK"

# model architecture
_C.TRAINING.IK_CONV_LAYER = 3
_C.TRAINING.IK_CONV_CHANNEL_SIZE = [1024, 1024, 1024]
_C.TRAINING.TRAINING_STEPS = 1
_C.TRAINING.START_LR = [5 * (10 ** (-4))]
_C.TRAINING.END_LR = [3.33 * (10 ** (-5))]
_C.TRAINING.EPOCH = [15]


def get_cfg_defaults():
    return _C.clone()


def update_config(config):
    assert config.BML_FOLDER is not None
    assert config.MODEL_FOLDER is not None
    # assert  config.PRETRAINED_POSE_MODEL is not None
    # assert len(config.TRAINING.USE_POSE_ESTIMATION) ==  config.TRAINING.TRAINING_STEPS
    return config
