import argparse
import math
import random
import numpy as np
import torch
from ms_model_estimation.models.dataset.BMLImgDataSet import BMLImgDataSet
from ms_model_estimation.models.loss.CustomLoss import CustomLoss
from ms_model_estimation.models.PoseEstimationModel import PoseEstimationModel
from ms_model_estimation.models.config.config_bml import get_cfg_defaults
from torch.utils.data import DataLoader
import os
from pathlib import Path
from ms_model_estimation.models.BMLUtils import CAMERA_TABLE
from ms_model_estimation.models.camera.cameralib import Camera
from tqdm import tqdm
import time
from ms_model_estimation.models.TorchTrainingProgram import TorchTrainingProgram


class Training(TorchTrainingProgram):

    def __init__(self, args, cfg):
        super(Training, self).__init__(args, cfg)

        # create camera parameter
        cameraParamter = {}
        for cameraType in CAMERA_TABLE:
            cameraInf = CAMERA_TABLE[cameraType]
            R = cameraInf["extrinsic"][:3, :3]  # .T
            t = np.matmul(cameraInf["extrinsic"][:3, -1:].T, R) * -1
            distortion_coeffs = np.array(
                [cameraInf["radialDisortionCoeff"][0], cameraInf["radialDisortionCoeff"][1], 0, 0, 0], np.float32)
            intrinsic_matrix = cameraInf["intrinsic"].copy()
            camera = Camera(t, R, intrinsic_matrix, distortion_coeffs)
            cameraParamter[cameraType] = camera
        self.cameraParamter = cameraParamter

        # Dataset
        self.h5pyFolder = cfg.BML_FOLDER if cfg.BML_FOLDER.endswith("/") else cfg.BML_FOLDER + "/"
        self.trainSet = BMLImgDataSet(self.cfg, self.h5pyFolder, self.cameraParamter, 0, evaluation=False)

        self.validationSet = BMLImgDataSet(self.cfg, self.h5pyFolder, self.cameraParamter, 1, evaluation=True)

        self.testSet = BMLImgDataSet(self.cfg, self.h5pyFolder, self.cameraParamter, 2, evaluation=True)

        print("%d training data, %d valid data, %d test data" % (
            len(self.trainSet), len(self.validationSet), len(self.testSet)))

        self.trainLoader = DataLoader(self.trainSet, batch_size=cfg.TRAINING.BATCHSIZE, shuffle=True,
                                      num_workers=self.cfg.WORKERS)
        self.valLoader = DataLoader(self.validationSet, batch_size=cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=True,
                                    drop_last=False, num_workers=self.cfg.WORKERS)
        self.testLoader = DataLoader(self.testSet, batch_size=cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                     drop_last=False, num_workers=self.cfg.WORKERS)
        # model
        self.create_model()

        # loss
        self.numLoss = 2
        self.lossNames = [
            "total loss",
            "3d pose loss",
            "3d marker loss",
        ]

        # model path
        if self.bestModelPath is not None:
            print(f'Load Pose Estimation Model : {self.bestModelPath}')
            dicts = self.model.state_dict()
            weights = torch.load(self.bestModelPath)
            for k, w in weights.items():
                if k in dicts:
                    dicts[k] = w
                else:
                    print(f'{k} is not in model')
            self.model.load_state_dict(dicts)

    def initialize_training(self, startEpoch, endEpoch, epochIdx):

        # loss
        self.positionLoss = CustomLoss.pose3d_mpjpe(root=False, L=self.cfg.TRAINING.LOSSTYPE)

        self.optimizer = torch.optim.AdamW(self.params, lr=self.cfg.TRAINING.START_LR[epochIdx], weight_decay=0.001)

        # learning rate schedule
        self.decayRate = (self.cfg.TRAINING.END_LR[epochIdx] / self.cfg.TRAINING.START_LR[epochIdx])
        if (endEpoch - startEpoch - 1) != 0:
            self.decayRate = self.decayRate ** (1 / (endEpoch - startEpoch - 1))

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        print(
            f'Training with lr from {self.cfg.TRAINING.START_LR[epochIdx]} to {self.cfg.TRAINING.END_LR[epochIdx]} in {endEpoch - startEpoch} epochs.')
        print(f'LR decay rate : {self.decayRate}')

    def update(self, epoch, startEpoch, endEpoch, epochIdx):

        self.lr_scheduler.step()
        if epoch != (endEpoch - 1):
            print("Current LR: ", self.lr_scheduler.get_last_lr())

    def create_model(self):
        self.model = PoseEstimationModel(
            self.cfg
        ) \
            .float().to(self.COMP_DEVICE)
        print(f'Image size : {self.cfg.MODEL.IMGSIZE}')

    def model_forward(self, inf):

        # get input
        image = inf["image"].to(self.COMP_DEVICE)
        pos3d = inf["pose3d"].to(self.COMP_DEVICE)
        mask = inf["joint3d_validity_mask"].to(self.COMP_DEVICE)
        rootOffset = pos3d[:, 0:1, :].clone()

        # forward
        predPos = self.model(image)

        # move the root to original point
        predPos = predPos - predPos[:, 0:1, :]

        # aligned with the gt root
        predPos3d = predPos + rootOffset

        return predPos3d, pos3d, mask

    def model_forward_and_calculate_loss(self, inf, evaluation=False):

        predPos3d, pos3d, mask = self.model_forward(inf)

        # 3D joint position loss
        joint3dLossValue = self.positionLoss(predPos3d[:, :self.cfg.PREDICTION.NUM_JOINTS, :],
                                             pos3d[:, :self.cfg.PREDICTION.NUM_JOINTS, :],
                                             mask=mask[:, :self.cfg.PREDICTION.NUM_JOINTS] if not evaluation else None,
                                             evaluation=evaluation) * self.cfg.HYP.POS

        marker3dLossValue = self.positionLoss(predPos3d[:, self.cfg.PREDICTION.NUM_JOINTS:, :],
                                              pos3d[:, self.cfg.PREDICTION.NUM_JOINTS:, :],
                                              mask=mask[:, self.cfg.PREDICTION.NUM_JOINTS:] if not evaluation else None,
                                              evaluation=evaluation) * self.cfg.HYP.MARKER

        losses = [joint3dLossValue, marker3dLossValue]

        return losses

    def store_every_frame_prediction(self, train=False, valid=False, test=True):

        # reload the model
        print("Load ", self.bestModelPath)
        self.create_model()
        if self.bestModelPath is not None:
            print(f'Load Pose Estimation Model : {self.bestModelPath}')
            dicts = self.model.state_dict()
            weights = torch.load(self.bestModelPath)
            for k, w in weights.items():
                if k in dicts:
                    dicts[k] = w
                else:
                    print(f'{k} is not in model')
            self.model.load_state_dict(dicts)
        self.model.eval()

        self.positionLoss = CustomLoss.pose3d_mpjpe(root=False, L=2)

        if valid:
            dataset = BMLImgDataSet(self.cfg, self.h5pyFolder, self.cameraParamter, 1, evaluation=True)
            datasetLoader = DataLoader(dataset, batch_size=cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                       drop_last=False,
                                       num_workers=self.cfg.EVAL_WORKERS)

            self.save_prediction(dataset, datasetLoader, 'valid')

        if test:
            dataset = BMLImgDataSet(self.cfg, self.h5pyFolder, self.cameraParamter, 2, evaluation=True)
            datasetLoader = DataLoader(dataset, batch_size=cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                       drop_last=False,
                                       num_workers=self.cfg.EVAL_WORKERS)

            self.save_prediction(dataset, datasetLoader, 'test')

        if train:
            dataset = BMLImgDataSet(self.cfg, self.h5pyFolder, self.cameraParamter, 0, evaluation=False,
                                    useEveryFrame=True, usePredefinedSeed=True)
            datasetLoader = DataLoader(dataset, batch_size=cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                       drop_last=False,
                                       num_workers=self.cfg.EVAL_WORKERS)

            self.save_prediction(dataset, datasetLoader, 'aug_train')

    def save_prediction(self, dataset, datasetLoader, name):

        # prediction
        prediction = np.zeros((len(dataset), self.cfg.PREDICTION.NUMPRED, 3), dtype=np.float32)
        correctedPos2d = np.zeros((len(dataset), self.cfg.PREDICTION.NUMPRED, 2), dtype=np.float32)
        correctedPos3d = np.zeros((len(dataset), self.cfg.PREDICTION.NUMPRED, 3), dtype=np.float32)
        masks = np.empty((len(dataset), self.cfg.PREDICTION.NUMPRED))

        with torch.no_grad():
            if self.cfg.PROGRESSBAR:
                iterator = enumerate(tqdm(datasetLoader))
            else:
                iterator = enumerate(datasetLoader, 0)

            for _, inf in iterator:
                fileIndices = inf["fileIdx"]
                predPos, _, _ = self.model_forward(inf)
                prediction[fileIndices, :, :] = predPos.cpu().detach().numpy()
                correctedPos3d[fileIndices, :, :] = inf["pose3d"].cpu().detach().numpy()
                correctedPos2d[fileIndices, :, :] = inf["pose2d"].cpu().detach().numpy()
                masks[fileIndices, :] = inf["joint3d_validity_mask"].cpu().detach().numpy()

        output = {
            "prediction": prediction,
            "label": correctedPos3d,
            'label2d': correctedPos2d,
            'masks': masks,
        }

        np.save(self.cfg.BML_FOLDER + f'prediction_{name}.npy', output)
        jointError = np.mean(np.sum((prediction[:, :self.cfg.PREDICTION.NUM_JOINTS, :] - correctedPos3d[:,
                                                                                         :self.cfg.PREDICTION.NUM_JOINTS,
                                                                                         :]) ** 2, axis=-1) ** 0.5)
        markerError = np.mean(np.sum((prediction[:, self.cfg.PREDICTION.NUM_JOINTS:, :] - correctedPos3d[:,
                                                                                          self.cfg.PREDICTION.NUM_JOINTS:,
                                                                                          :]) ** 2, axis=-1) ** 0.5)
        print(f'Joint Error : {jointError} , Marker Error : {markerError}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cpu', action='store_true', default=False, help="only use cpu?")
    parser.add_argument('--evaluation', action='store_true', default=False, help="only use cpu?")
    parser.add_argument('--ymlFile', action='store',
                        default="", type=str,
                        help="The hdf5 folder")
    parser.add_argument('--testTime', action='store_true', default=False, help="only use cpu?")

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    if args.ymlFile:
        cfg.merge_from_file(args.ymlFile)
        # cfg = update_config(cfg)
    cfg.freeze()
    print(cfg)

    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    trainingProgram = Training(args, cfg)
    if args.evaluation:
        trainingProgram.store_every_frame_prediction(train=True, valid=True, test=True)
    elif args.testTime:
        pass
    else:
        trainingProgram.run()
