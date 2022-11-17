from ms_model_estimation.models.config.config_bml import get_cfg_defaults
from ms_model_estimation.models.utils.OSUtils import PredictedOSJoints , LeafJoints
from ms_model_estimation.models.utils.BMLUtils import PredictedMarkers, smplHJoint
from yacs.config import CfgNode as CN

_C = get_cfg_defaults()
_C.TRAINING.NUMPRED = len(PredictedOSJoints) + len(LeafJoints) + len(smplHJoint) + len(PredictedMarkers)
_C.TRAINING.NUM_OS_JOINT = len(PredictedOSJoints) + len(LeafJoints)
_C.TRAINING.NUM_SMPLH_MARKERS = len(smplHJoint) + len(PredictedMarkers)

_C.HYP = CN()
_C.HYP.OS = 1
_C.HYP.SMPL = 1


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
