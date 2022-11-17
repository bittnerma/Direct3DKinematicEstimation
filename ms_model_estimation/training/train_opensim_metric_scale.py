import argparse
import math
import random
import time

import numpy as np
import torch
from ms_model_estimation.training.dataset.BMLImgOSDataSet import BMLImgDataSet
from ms_model_estimation.training.loss.CustomLoss import CustomLoss
from ms_model_estimation.training.networks.PoseEstimationModel import PoseEstimationModel
from ms_model_estimation.training.config.config_os_metric_scale import get_cfg_defaults
from torch.utils.data import DataLoader
import os
from pathlib import Path
from ms_model_estimation.training.utils.BMLUtils import CAMERA_TABLE
from ms_model_estimation.training.camera.cameralib import Camera
from tqdm import tqdm


class Training:

    def __init__(self, args, cfg):

        self.cfg = cfg

        if args.cpu:
            self.COMP_DEVICE = torch.device("cpu")
        else:
            self.COMP_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.modelFolder = cfg.MODEL_FOLDER if cfg.MODEL_FOLDER.endswith("/") else cfg.MODEL_FOLDER + "/"
        if not os.path.exists(self.modelFolder):
            Path(self.modelFolder).mkdir(parents=True, exist_ok=True)

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
            "3d opensim loss",
            "3d smpl loss",
        ]

        # model path
        self.bestValidLoss = math.inf
        self.bestModelPath = self.cfg.STARTMODELPATH

    def run(self):

        assert self.cfg.TRAINING.TRAINING_STEPS >= 1

        startEpoch = 0
        endEpoch = 0
        for i in range(self.cfg.TRAINING.TRAINING_STEPS):
            startEpoch += endEpoch
            endEpoch += self.cfg.TRAINING.EPOCH[i]
            self.train(startEpoch, endEpoch, i)

        print('Finished Training')

        # self.evaluate()

    def train(self, startEpoch, endEpoch, idx):

        try:
            del self.optimizer
        except:
            pass

        # loss
        self.positionLoss = CustomLoss.pose3d_mpjpe(root=False, L=self.cfg.TRAINING.LOSSTYPE)

        # optimizer
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr=self.cfg.TRAINING.START_LR[idx], weight_decay=0.001)

        # learning rate schedule
        self.decayRate = (self.cfg.TRAINING.END_LR[idx] / self.cfg.TRAINING.START_LR[idx])
        if (endEpoch - startEpoch - 1) != 0:
            self.decayRate = self.decayRate ** (1 / (endEpoch - startEpoch - 1))

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        print(
            f'Training with lr from {self.cfg.TRAINING.START_LR[idx]} to {self.cfg.TRAINING.END_LR[idx]} in {endEpoch - startEpoch} epochs.')
        print(f'LR decay rate : {self.decayRate}')

        epoch = 0

        NUM_ITERATION = len(self.trainSet) // self.cfg.TRAINING.BATCHSIZE

        for epoch in range(startEpoch, endEpoch):

            if self.cfg.TESTMODE:
                currentIter = 0
                NUM_ITERATION = 50
                startTime = time.time()

            self.model.train()
            runningLoss = [0] * (self.numLoss + 1)
            if self.cfg.PROGRESSBAR:
                iterator = enumerate(tqdm(self.trainLoader))
            else:
                iterator = enumerate(self.trainLoader, 0)

            for _, inf in iterator:
                self.optimizer.zero_grad()
                losses = self.model_forward(inf=inf, evaluation=False)
                loss = losses[0]
                for a in range(1, len(losses)):
                    loss = loss + losses[a]
                loss.backward()
                self.optimizer.step()

                runningLoss[0] += loss.item()
                for i in range(self.numLoss):
                    value = losses[i]
                    if torch.is_tensor(value):
                        value = value.item()
                    runningLoss[i + 1] += value

                if self.cfg.TESTMODE:
                    currentIter += 1
                    if currentIter == NUM_ITERATION:
                        break

            self.report(f'Train , Epoch {epoch} :', runningLoss, NUM_ITERATION)
            if self.cfg.TESTMODE:
                endTime = time.time()
                print(f'Training Time : {endTime - startTime} s with {NUM_ITERATION} iterations')

            # validation set
            runningLoss = [0] * (self.numLoss + 1)
            self.model.eval()
            numValidData = len(self.validationSet)
            with torch.no_grad():
                if self.cfg.PROGRESSBAR:
                    iterator = enumerate(tqdm(self.valLoader))
                else:
                    iterator = enumerate(self.valLoader, 0)

                for _, inf in iterator:
                    losses = self.model_forward(inf=inf, evaluation=True)
                    loss = losses[0]
                    for a in range(1, len(losses)):
                        loss = loss + losses[a]

                    runningLoss[0] += loss.item()
                    for i in range(self.numLoss):
                        value = losses[i]
                        if torch.is_tensor(value):
                            value = value.item()
                        runningLoss[i + 1] += value

            self.report(f'Valid, Epoch {epoch} :', runningLoss, len(self.validationSet))

            # print the lr if lr changes
            self.lr_scheduler.step()
            if epoch != (endEpoch - 1):
                print("Current LR: ", self.lr_scheduler.get_last_lr())

            # save the best model
            if self.bestValidLoss > (runningLoss[0] / numValidData):
                self.bestModelPath = self.save_model(epoch)
                self.bestValidLoss = (runningLoss[0] / numValidData)

        # save the model in the last epoch
        _ = self.save_model(epoch)

    def save_model(self, epoch):
        path = f'{self.modelFolder}model_{epoch}_{self.cfg.POSTFIX}.pt'
        torch.save(self.model.state_dict(), path)
        return path

    def create_model(self):
        self.model = PoseEstimationModel(
            self.cfg
        ) \
            .float().to(self.COMP_DEVICE)
        print(f'Image size : {self.cfg.TRAINING.IMGSIZE}')

    def report(self, dataset, runnigLoss, numData):
        s = ""
        s += f'{dataset} : '
        for i in range(len(self.lossNames)):
            s += f'{self.lossNames[i]} : {runnigLoss[i] / numData}   '

        print(s)

    def model_forward(self, inf=None, evaluation=False, prediction=False):

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

        # 3D joint position loss
        opensimJointLoss = self.positionLoss(predPos3d[:, :self.cfg.TRAINING.NUM_OS_JOINT, :],
                                             pos3d[:, :self.cfg.TRAINING.NUM_OS_JOINT, :],
                                             mask=mask[:, :self.cfg.TRAINING.NUM_OS_JOINT] if not evaluation else None,
                                             evaluation=evaluation) * self.cfg.HYP.OS
        smplMarkerLoss = self.positionLoss(predPos3d[:, self.cfg.TRAINING.NUM_OS_JOINT:, :],
                                           pos3d[:, self.cfg.TRAINING.NUM_OS_JOINT:, :],
                                           mask=mask[:, self.cfg.TRAINING.NUM_OS_JOINT:] if not evaluation else None,
                                           evaluation=evaluation) * self.cfg.HYP.SMPL

        losses = [opensimJointLoss, smplMarkerLoss]

        if prediction:
            return predPos3d, losses
        else:
            return losses

    def store_every_frame_prediction(self, train=False, valid=False, test=True, augmentedTrain=True):

        # reload the model
        print("Load ", self.bestModelPath)
        self.create_model()
        self.model.load_state_dict(torch.load(self.bestModelPath))
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
            dataset = BMLImgDataSet(self.cfg, self.h5pyFolder, self.cameraParamter, 0, evaluation=True)
            datasetLoader = DataLoader(dataset, batch_size=cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                       drop_last=False,
                                       num_workers=self.cfg.EVAL_WORKERS)

            self.save_prediction(dataset, datasetLoader, 'train')

    def save_prediction(self, dataset, datasetLoader, name):

        # prediction
        prediction = np.zeros((len(dataset), self.cfg.TRAINING.NUMPRED, 3), dtype=np.float32)
        correctedPos2d = np.zeros((len(dataset), self.cfg.TRAINING.NUMPRED, 2), dtype=np.float32)
        correctedPos3d = np.zeros((len(dataset), self.cfg.TRAINING.NUMPRED, 3), dtype=np.float32)
        masks = np.empty((len(dataset), self.cfg.TRAINING.NUMPRED))

        with torch.no_grad():
            if self.cfg.PROGRESSBAR:
                iterator = enumerate(tqdm(datasetLoader))
            else:
                iterator = enumerate(datasetLoader, 0)

            for _, inf in iterator:
                fileIndices = inf["fileIdx"]
                predPos, loss = self.model_forward(inf=inf, evaluation=True, prediction=True)
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

        osJointError = np.mean(np.sum((prediction[:, :self.cfg.TRAINING.NUM_OS_JOINT, :] - correctedPos3d[:,
                                                                                           :self.cfg.TRAINING.NUM_OS_JOINT,
                                                                                           :]) ** 2, axis=-1) ** 0.5)
        smplMarkerError = np.mean(np.sum((prediction[:, self.cfg.TRAINING.NUM_OS_JOINT:, :] - correctedPos3d[:,
                                                                                              self.cfg.TRAINING.NUM_OS_JOINT:,
                                                                                              :]) ** 2, axis=-1) ** 0.5)
        print(f'OS Joint Error : {osJointError} , SMPL Marker Error : {smplMarkerError}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cpu', action='store_true', default=False, help="only use cpu?")
    parser.add_argument('--evaluation', action='store_true', default=False, help="only use cpu?")
    parser.add_argument('--ymlFile', action='store',
                        default="", type=str,
                        help="The hdf5 folder")

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    if args.ymlFile:
        cfg.merge_from_file(args.ymlFile)
    cfg.freeze()
    print(cfg)

    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    trainingProgram = Training(args, cfg)
    if not args.evaluation:
        trainingProgram.run()
    # else:
    #    trainingProgram.evaluate()
    trainingProgram.store_every_frame_prediction(train=True, valid=True, test=True)
