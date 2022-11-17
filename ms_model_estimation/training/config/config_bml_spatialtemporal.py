from ms_model_estimation.training.utils.BMLUtils import PredictedMarkers, smplHJoint
from ms_model_estimation.training.utils.BMLUtils import TrainingSubjects, ValidationSubjects, TestSubjects, Joints_Inside_Cylinder_Body, \
    PairMarkers, SymmetricPairMarkers
from yacs.config import CfgNode as CN
from ms_model_estimation.training.config.config_bml import get_cfg_defaults

_C = get_cfg_defaults()


# Postfix for the model name
_C.POSTFIX = "BML_SPATIALTEMPORAL"

# Model Architecture
_C.MODEL.TYPE = 3
_C.MODEL.INFEATURES = _C.PREDICTION.NUMPRED*3
_C.MODEL.OUTFEATURES = _C.PREDICTION.NUMPRED*3
_C.MODEL.CAUSAL = False
_C.MODEL.REFINE = False
_C.MODEL.USE_2D = False
_C.MODEL.USE_GT = False

_C.MODEL.AWARED_OCCLUSION = CN()
_C.MODEL.AWARED_OCCLUSION.GT = False
_C.MODEL.AWARED_OCCLUSION.USE = False
_C.MODEL.AWARED_OCCLUSION.USE_PREDICTION = False
_C.MODEL.RECEPTIVE_FIELD = 64
_C.MODEL.LAYER_NORM_AFFINE = True

# Training setting
_C.TRAINING.BATCHSIZE = 2
_C.TRAINING.EVALUATION_BATCHSIZE = 2
_C.TRAINING.TRAINING_STEPS = 1
_C.TRAINING.EPOCH = [5]
_C.TRAINING.START_LR = [0.00005]
_C.TRAINING.END_LR = [0.000005]
_C.TRAINING.MOMENTUM = 0.0001
_C.TRAINING.LOSSTYPE = None
_C.TRAINING.SPATIAL_LOSSTYPE = 1
_C.TRAINING.TEMPORAL_LOSSTYPE = 2
_C.TRAINING.RESNET50_LAYER = False


# user every i-th frame for training and evaluation
_C.DATASET.TRAINING_USEDEACHFRAME = _C.MODEL.RECEPTIVE_FIELD
_C.DATASET.USEDEACHFRAME = 1

# Paths and data folder
_C.PASCAL_PATH = None
_C.BML_FOLDER = None
_C.MODEL_FOLDER = None
_C.STARTPOSMODELPATH = None
_C.STARTTEMPORALMODELPATH = None
_C.STARTMODELPATH = None

# hyper paramters
_C.HYP = CN()
_C.HYP.SPATIAL_POS = 1
_C.HYP.SPATIAL_MARKER = 1
_C.HYP.TEMPORAL_POS = 1
_C.HYP.TEMPORAL_MARKER = 1

# transformer architecture
_C.TRANSFORMER = CN()
_C.TRANSFORMER.D_MODEL = 256
_C.TRANSFORMER.D_CHANNEL = 512
_C.TRANSFORMER.N_HEADS = 8
_C.TRANSFORMER.STRIDES = [3, 9, 9]
_C.TRANSFORMER.BATCHNORM = False
_C.TRANSFORMER.DROPOUT = False
_C.TRANSFORMER.N1 = 3
_C.TRANSFORMER.POS_ENCODNIG_EVERY_BLOCK = True

# Geometric Augmentation
_C.DATASET.GEOM.AUG = False
_C.DATASET.GEOM.HFLIP = True


def get_cfg_defaults():
    return _C.clone()


def update_config(config):
    assert config.BML_FOLDER is not None
    assert config.MODEL_FOLDER is not None
    # assert  config.PRETRAINED_POSE_MODEL is not None
    # assert len(config.TRAINING.USE_POSE_ESTIMATION) ==  config.TRAINING.TRAINING_STEPS

    return config
