from ms_model_estimation.models.config.config_os import get_cfg_defaults
from yacs.config import CfgNode as CN

_C = get_cfg_defaults()

# Postfix for the model name
_C.POSTFIX = "OS_TEMPORAL"

# transformer architecture
_C.TRANSFORMER = CN()
_C.TRANSFORMER.D_MODEL = 256
_C.TRANSFORMER.D_CHANNEL = 512
_C.TRANSFORMER.N_HEADS = 8
_C.TRANSFORMER.STRIDES = [3, 9, 9]
_C.TRANSFORMER.BATCHNORM = False
_C.TRANSFORMER.DROPOUT = False
_C.TRANSFORMER.N1 = 3
_C.TRANSFORMER.POS_ENCODNIG_EVERY_BLOCK = False


# LSTM
_C.LSTM = CN()
_C.LSTM.HIDDEN_STATE = 512
_C.LSTM.NUM_LAYERS = 3
_C.LSTM.DROPOUTPROB = 0.1
_C.LSTM.BIDIRECTIONAL = True


_C.TRAINING.BATCHSIZE = 128
_C.TRAINING.EVALUATION_BATCHSIZE = 128
_C.TRAINING.TRAINING_STEPS = 1
_C.TRAINING.EPOCH = [50]
_C.TRAINING.START_LR = [0.001]
_C.TRAINING.END_LR = [0.000005]
_C.TRAINING.INITIAL_MOMENTUM = 0.1
_C.TRAINING.END_MOMENTUM = 0.001
_C.TRAINING.AMSGRAD = False

# use marker position and joint position as the input
_C.MODEL.POS = True
_C.MODEL.TYPE = 1
_C.MODEL.INFEATURES = 3 * 3 + _C.MODEL.NUMPRED * 3 + len(_C.PREDICTION.BODY) * 3 + len(_C.PREDICTION.COORDINATES)
_C.MODEL.OUTFEATURES = _C.PREDICTION.BODY_SCALE_UNIQUE_NUMS + len(_C.PREDICTION.COORDINATES) + 6
_C.MODEL.CAUSAL = False
_C.MODEL.RECEPTIVE_FIELD = 243


_C.LOSS.POS.USE = True
_C.LOSS.POS.TYPE = 2
_C.LOSS.POS.BETA = 0.04

_C.LOSS.MARKER.USE = True
_C.LOSS.MARKER.TYPE = 2
_C.LOSS.MARKER.BETA = 0.04


# coordinate angle
_C.LOSS.ANGLE.USE = True
_C.LOSS.ANGLE.USEMASK = True
_C.LOSS.ANGLE.MASKMINVALUE = 0
_C.LOSS.ANGLE.TYPE = 3
_C.LOSS.ANGLE.EVAL_TYPE = 1
_C.LOSS.ANGLE.BETA = 5
_C.LOSS.ANGLE.H = 0.1

# body scale
_C.LOSS.BODY.USE = True
_C.LOSS.BODY.TYPE = 3
_C.LOSS.BODY.EVAL_TYPE = 1
_C.LOSS.BODY.BETA = 0.05
_C.LOSS.BODY.H = 0.1

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

    if config.MODEL.POS:
        config.MODEL.INFEATURES = 3 * 3 + config.MODEL.NUMPRED * 3 + len(config.PREDICTION.BODY) * 3 + len(config.PREDICTION.COORDINATES)
    else:
        config.MODEL.INFEATURES = 3 * 3 + len(config.PREDICTION.BODY) * 3 + len(config.PREDICTION.COORDINATES)

    if config.MODEL.TYPE == 0:
        config.TRAINING.AMSGRAD = True
    else:
        config.TRAINING.AMSGRAD = False

    return config
