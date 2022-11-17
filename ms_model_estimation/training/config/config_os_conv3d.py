from ms_model_estimation.training.config.config_os import get_cfg_defaults
from yacs.config import CfgNode as CN

_C = get_cfg_defaults()

# Postfix for the model name
_C.POSTFIX = "OS_CONV3D"

# model architecture

# use layers before average pooling
_C.MODEL.RECEPTIVE_FIELD = 64
_C.MODEL.FINE_TUNE_LAST_LAYER = True
_C.MODEL.CAUSAL = False

# Image Size
_C.MODEL.IMGSIZE = (224, 224)

_C.PRETRAINED_I3D_PATH = None

def get_cfg_defaults():
    return _C.clone()


def update_config(config):
    if config.DATASET.OCCLUSION.PROB > 0:
        assert config.PASCAL_PATH is not None

    assert config.BML_FOLDER is not None
    assert config.MODEL_FOLDER is not None

    config.TRAINING.EPOCH = [int(c) for c in config.TRAINING.EPOCH]
    config.TRAINING.START_LR = [float(c) for c in config.TRAINING.START_LR]
    config.TRAINING.END_LR = [float(c) for c in config.TRAINING.END_LR]

    config.MODEL.CONV_CHANNEL_SIZE = [int(i) for i in config.MODEL.CONV_CHANNEL_SIZE]

    return config
