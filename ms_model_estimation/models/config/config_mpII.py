from ms_model_estimation.models.config.config_default_img_training import get_cfg_defaults as get_img_defaults
from ms_model_estimation.models.utils.MPIIUtils import MAPPING_H36M_INDEX_TO_MPII

_C = get_img_defaults()

# Postfix for the model name
_C.POSTFIX = "MPII_PETRAINED"

# The number of prediction
_C.TRAINING.NUMPRED = 17


# ratio for validation set
_C.TRAINING.VALIDATIONRATIO = 0.05

# 2D projection loss
_C.TRAINING.LOSS2D = True
_C.TRAINING.LOSSTYPE = 1

# hyperparameter
# _C.TRAINING.HYP = [1]

# Learning rate and the number of epoch
_C.TRAINING.TRAINING_STEPS = 2
_C.TRAINING.START_LR = [10 ** (-4), 3.33 * (10 ** (-6))]
_C.TRAINING.END_LR = [3.33 * (10 ** (-5)), (10 ** (-6))]
_C.TRAINING.EPOCH = [25, 2]

_C.TRAINING.MAPPING_H36M_INDEX_TO_MPII = MAPPING_H36M_INDEX_TO_MPII


def get_cfg_defaults():
    return _C.clone()
