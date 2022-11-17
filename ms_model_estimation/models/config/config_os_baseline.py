from yacs.config import CfgNode as CN
from ms_model_estimation.models.config.config_bml import get_cfg_defaults as get_bml_defaults
from ms_model_estimation.models.utils.BMLUtils import PredictedMarkers, smplHJoint
from ms_model_estimation.models.utils.OSUtils import  PredictedCoordinates

_C = get_bml_defaults()

_C.OPENSIM = CN()
_C.OPENSIM.GTFOLDER = None
_C.OPENSIM.MODELPATH = None
_C.OPENSIM.UNIT = "m"
_C.OPENSIM.prescaling_lockedCoordinates = None
_C.OPENSIM.prescaling_unlockedConstraints = None
_C.OPENSIM.prescaling_defaultValues = None
_C.OPENSIM.postscaling_lockedCoordinates = None
_C.OPENSIM.postscaling_unlockedConstraints = None
_C.OPENSIM.changingParentMarkers = None


_C.PREDICTION.COORDINATES = PredictedCoordinates

# overwrite the results of IK
_C.OVERWRITE_IK = True
# overwrite the results of body scaling
_C.OVERWRITE_BS = True

# bbox image folder
_C.BBOX_FOLDER = None
# .pkl file to save the bounding boxes
_C.BBOX_PKL_PATH = None

# best nn model
_C.bestNNModelPath = None

def get_cfg_defaults():
    return _C.clone()


def update_config(config):

    assert config.BML_FOLDER is not None
    assert config.OPENSIM.GTFOLDER is not None
    assert config.BBOX_FOLDER is not None
    assert config.OPENSIM.MODELPATH is not None
    assert config.bestNNModelPath is not None
    assert config.BBOX_PKL_PATH is not None

    config.BML_FOLDER = config.BML_FOLDER if config.BML_FOLDER.endswith("/") else config.BML_FOLDER + "/"
    config.OPENSIM.GTFOLDER = config.OPENSIM.GTFOLDER if config.OPENSIM.GTFOLDER.endswith("/") else config.OPENSIM.GTFOLDER + "/"
    config.BBOX_FOLDER = config.BBOX_FOLDER if config.BBOX_FOLDER.endswith("/") else config.BBOX_FOLDER + "/"

    return config