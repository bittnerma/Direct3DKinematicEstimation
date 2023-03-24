import pickle as pkl
import numpy as np
import math
import argparse
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).absolute().parent / "ms_model_estimation"))
from ms_model_estimation.training.train_opensim import Training
from ms_model_estimation.training.config.config_os import get_cfg_defaults

from ms_model_estimation.training.train_opensim_temporal4 import Training as TemporalTraining
from ms_model_estimation.training.config.config_os_temporal import get_cfg_defaults as get_temporal_cfg_defaults

cwd = Path(__file__).absolute().parent

cfg = get_cfg_defaults()

# cfg.STARTPOSMODELPATH = str((cwd / r'checkpoints\OS_ALL_L2_ANGLE006.pt').as_posix())
# cfg.STARTTEMPORALMODELPATH = str((cwd / r'checkpoints\OS_TEMPORAL_TRANSFORMER.pt').as_posix())
cfg.MODEL_FOLDER = str((cwd / r'checkpoints').as_posix())
cfg.BML_FOLDER = str((cwd / r'_dataset_full').as_posix()) + "/"
cfg.PASCAL_PATH = str((cwd / r'_dataset/pascal.hdf5').as_posix())
cfg.TRAINING.EPOCH = [1]
cfg.TRAINING.TRAINING_STEPS = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cpu', action='store_true', default=False, help="only use cpu?")
    parser.add_argument('--evaluation', action='store_true', default=False, help="only use cpu?")    

    args = parser.parse_args()    
    
    # cfg.freeze()
    # print(cfg)

    # torch.manual_seed(cfg.SEED)
    # random.seed(cfg.SEED)
    # np.random.seed(cfg.SEED)

    # trainingProgram = Training(args, cfg)
    # trainingProgram.run()
    # trainingProgram.store_every_frame_prediction(train=True, valid=True, test=True)

    cfg = get_temporal_cfg_defaults()
    cfg.MODEL_FOLDER = str((cwd / r'checkpoints').as_posix())
    cfg.BML_FOLDER = str((cwd / r'_dataset_full').as_posix()) + "/"
    cfg.PASCAL_PATH = str((cwd / r'_dataset/pascal.hdf5').as_posix())
    cfg.TRAINING.EPOCH = [1]
    cfg.TRAINING.TRAINING_STEPS = 1

    cfg.freeze()
    print(cfg)

    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    trainingProgram = TemporalTraining(args, cfg)
    trainingProgram.run()
    trainingProgram.store_every_frame_prediction(train=True, valid=True, test=True)
    