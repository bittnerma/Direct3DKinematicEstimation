from ms_model_estimation.training.config.config_os_temporal import get_cfg_defaults
from yacs.config import CfgNode as CN

_C = get_cfg_defaults()

# Postfix for the model name
_C.POSTFIX = "OS_TEMPORAL"
_C.STARTPOSMODELPATH = None
_C.STARTTEMPORALMODELPATH = None

def get_cfg_defaults():
    return _C.clone()


def update_config(config):
    if config.DATASET.OCCLUSION.PROB > 0:
        assert config.PASCAL_PATH is not None

    assert config.BML_FOLDER is not None
    assert config.MODEL_FOLDER is not None

    return config
