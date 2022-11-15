import pickle as pkl
import numpy as np
import math
import argparse
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).parent / "ms_model_estimation"))
from ms_model_estimation.models.utils_os_spatialTemporal_infer import Training
from ms_model_estimation.models.config.config_os_spatialtemporal_time import get_cfg_defaults

cwd = Path(__file__).parent

cfg = get_cfg_defaults()

cfg.STARTPOSMODELPATH = str((cwd / r'checkpoints\OS_ALL_L2_ANGLE006.pt').as_posix())
cfg.STARTTEMPORALMODELPATH = str((cwd / r'checkpoints\OS_TEMPORAL_TRANSFORMER.pt').as_posix())
cfg.MODEL_FOLDER = str((cwd / r'checkpoints').as_posix())
cfg.BML_FOLDER = str((cwd / r'_dataset_full').as_posix())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cpu', action='store_true', default=False, help="only use cpu?")
    parser.add_argument('--evaluation', action='store_true', default=False, help="only use cpu?")    

    args = parser.parse_args()    
    
    cfg.freeze()
    print(cfg)

    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    trainingProgram = Training(args, cfg)
    trainingProgram.run_inference(datasplit='test')

