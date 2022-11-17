import argparse
import math
import pickle
import random
import time

import h5py
import numpy as np
import torch
from ms_model_estimation.models.dataset.BMLOpenSimTemporalDataSet import BMLOpenSimTemporalDataSet
from ms_model_estimation.models.loss.CustomLoss import CustomLoss
from ms_model_estimation.models.networks.OpenSimTemporalModel import OpenSimTemporalModel
from ms_model_estimation.models.config.config_os_temporal import get_cfg_defaults, update_config
from torch.utils.data import DataLoader
import os
from pathlib import Path
from ms_model_estimation.models.utils.BMLUtils import CAMERA_TABLE
from ms_model_estimation.models.camera.cameralib import Camera
from tqdm import tqdm
from ms_model_estimation.models.dataset.data_loading import load_and_transform3d


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
        self.trainSet = BMLOpenSimTemporalDataSet(cfg, cfg.BML_FOLDER,
                                                  self.cfg.BML_FOLDER + "prediction_OS_augmented_train.npy",
                                                  0, evaluation=False)

        self.validationSet = BMLOpenSimTemporalDataSet(cfg, cfg.BML_FOLDER,
                                                       self.cfg.BML_FOLDER + "prediction_OS_valid.npy",
                                                       1, evaluation=True,
                                                       )

        self.testSet = BMLOpenSimTemporalDataSet(cfg, cfg.BML_FOLDER, self.cfg.BML_FOLDER + "prediction_OS_test.npy",
                                                  2, evaluation=True)

        print("%d training data, %d valid data, %d test data" % (
            len(self.trainSet), len(self.validationSet), len(self.testSet)))

        self.trainLoader = DataLoader(self.trainSet, batch_size=cfg.TRAINING.BATCHSIZE, shuffle=True,
                                      num_workers=self.cfg.WORKERS)
        self.valLoader = DataLoader(self.validationSet, batch_size=cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=True,
                                    drop_last=False,
                                    num_workers=self.cfg.EVAL_WORKERS)
        self.testLoader = DataLoader(self.testSet, batch_size=cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                     drop_last=False,
                                     num_workers=self.cfg.EVAL_WORKERS)

        # load pyBaseModel
        self.pyBaseModel = pickle.load(open(self.cfg.BML_FOLDER + "pyBaseModel.pkl", "rb"))

        # coordinate value range        
        coordinateValueRange = torch.zeros(len(self.cfg.PREDICTION.COORDINATES), 2).float().to(self.COMP_DEVICE)
        for idx, coordinateName in enumerate(self.cfg.PREDICTION.COORDINATES):
            joint, idxC = self.pyBaseModel.jointSet.coordinatesDict[coordinateName]
            minValue = joint.coordinates[idxC].minValue
            maxValue = joint.coordinates[idxC].maxValue
            if abs(minValue) > (2 * math.pi) or abs(maxValue) > (2 * math.pi):
                coordinateValueRange[idx, 0] = 0
                coordinateValueRange[idx, 1] = 2 * math.pi
            else:
                minValue = np.fmod(minValue, math.pi * 2)
                maxValue = np.fmod(maxValue, math.pi * 2)
                coordinateValueRange[idx, 0] = (minValue + maxValue) / 2
                coordinateValueRange[idx, 1] = maxValue - (minValue + maxValue) / 2
        self.coordinateValueRange = coordinateValueRange.requires_grad_(False)

        self.markerWeights = torch.Tensor(self.cfg.TRAINING.MARKER_WEIGHT).float().to(self.COMP_DEVICE).requires_grad_(
            False)

        # best model
        self.bestValidLoss = math.inf
        self.bestModelPath = self.cfg.STARTMODELPATH

        self.create_model()

        # model
        if self.cfg.STARTMODELPATH is not None:
            self.model.load_state_dict(torch.load(self.cfg.STARTMODELPATH))

        # loss
        self.numLoss = 2
        self.train_positionLoss = CustomLoss.pose3d_mpjpe(root=False, L=self.cfg.LOSS.POS.TYPE)
        self.infer_positionLoss = CustomLoss.pose3d_mpjpe(root=False, L=2)
        self.train_marker_positionLoss = CustomLoss.pose3d_mpjpe(root=False, L=self.cfg.LOSS.POS.TYPE,
                                                                 weights=self.markerWeights)
        self.infer_marker_positionLoss = CustomLoss.pose3d_mpjpe(root=False, L=2)
        self.lossNames = [
            "total loss",
            "3d pose loss",
            "3d marker loss",
        ]
        self.metrcIdx = 2

        if self.cfg.LOSS.ANGLE.USE:
            self.numLoss += 1
            self.metricLossIdx = [3]

            # create coordinate weight
            maxWeight = self.cfg.TRAINING.COORDINATE_MAX_WEIGHT
            minRange, maxRange = torch.min(self.coordinateValueRange[:, 1]), torch.max(self.coordinateValueRange[:, 1])
            coeff = np.log(maxWeight) / (maxRange - minRange)
            self.coordinateWeights = torch.exp((self.coordinateValueRange[:, 1] - minRange) * coeff).to(
                self.COMP_DEVICE)

            self.train_coordinate_angle_loss = CustomLoss.opensim_coordinate_angle_loss(
                coordinateWeights=self.coordinateWeights)
            self.infer_coordinate_angle_loss = CustomLoss.opensim_coordinate_angle_loss(coordinateWeights=None)
            self.lossNames.append("angle loss")

        # loss for body scales
        if self.cfg.LOSS.BODY.USE:
            self.numLoss += 1
            self.train_bone_scale_loss = CustomLoss.bone_scale_loss(self.cfg.LOSS.BODY.TYPE)
            self.infer_bone_scale_loss = CustomLoss.bone_scale_loss(self.cfg.LOSS.BODY.EVAL_TYPE)
            self.lossNames.append("bone scale loss")

        # loss for symmetric body scale
        if self.cfg.LOSS.SY_BODY.USE:
            self.numLoss += 1
            self.train_symmetric_loss = CustomLoss.bone_scale_loss(self.cfg.LOSS.SY_BODY.TYPE)
            self.infer_symmetric_loss = CustomLoss.bone_scale_loss(self.cfg.LOSS.SY_BODY.EVAL_TYPE)
            self.lossNames.append("symmetric loss")

    def swapLoss(self, evaluation=False):

        self.positionLoss = self.train_positionLoss if not evaluation else self.infer_positionLoss
        self.marker_positionLoss = self.train_marker_positionLoss if not evaluation else self.infer_marker_positionLoss

        if self.cfg.LOSS.BODY.USE:
            self.bone_scale_loss = self.train_bone_scale_loss if not evaluation else self.infer_bone_scale_loss
        if self.cfg.LOSS.ANGLE.USE:
            self.coordinate_angle_loss = self.train_coordinate_angle_loss if not evaluation else self.infer_coordinate_angle_loss
        if self.cfg.LOSS.SY_BODY.USE:
            self.symmetric_loss = self.train_symmetric_loss if not evaluation else self.infer_symmetric_loss

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
        trainingProgram.store_every_frame_prediction(train=True, valid=True, test=True)

    def train(self, startEpoch, endEpoch, idx):

        try:
            del self.optimizer
        except:
            pass

        # optimizer
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.params, lr=self.cfg.TRAINING.START_LR[idx], amsgrad=True)

        # learning rate schedule
        self.decayRate = (self.cfg.TRAINING.END_LR[idx] / self.cfg.TRAINING.START_LR[idx]) ** (
                1 / (endEpoch - startEpoch + 1))
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)
        print(f'LR starts from {self.cfg.TRAINING.START_LR[idx]}, LR decay rate : {self.decayRate}')

        epoch = 0

        NUM_ITERATION = len(self.trainSet) // self.cfg.TRAINING.BATCHSIZE

        for epoch in range(startEpoch, endEpoch):

            if self.cfg.TESTMODE:
                currentIter = 0
                NUM_ITERATION = 50
                startTime = time.time()

            self.model.train()
            self.swapLoss(evaluation=False)
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

            if self.cfg.TESTMODE:
                startTime = time.time()

            # validation set
            runningLoss = [0] * (self.numLoss + 1)
            self.optimizer.zero_grad()
            self.model.eval()
            self.swapLoss(evaluation=True)
            numValidData = len(self.validationSet)
            with torch.no_grad():

                if self.cfg.PROGRESSBAR:
                    iterator = enumerate(tqdm(self.valLoader))
                else:
                    iterator = enumerate(self.valLoader, 0)

                for _, inf in iterator:
                    self.optimizer.zero_grad()
                    losses = self.model_forward(inf=inf, evaluation=True)
                    loss = losses[0]
                    for a in range(1, len(losses)):
                        loss = loss + losses[a]

                    runningLoss[0] += loss.detach().item()
                    for i in range(self.numLoss):
                        value = losses[i].detach()
                        if torch.is_tensor(value):
                            value = value.item()
                        runningLoss[i + 1] += value

            self.report(f'Valid, Epoch {epoch} :', runningLoss, len(self.validationSet))

            if self.cfg.TESTMODE:
                endTime = time.time()
                print(f'Validation Time : {endTime - startTime} s')

            # print the lr if lr changes
            self.lr_scheduler.step()
            if epoch != (endEpoch - 1):
                print("Current LR: ", self.lr_scheduler.get_last_lr())

            # set the momentum
            momentum = self.cfg.TRAINING.INITIAL_MOMENTUM * np.exp(
                - (epoch - startEpoch) / (endEpoch - startEpoch + 1) * np.log(
                    self.cfg.TRAINING.INITIAL_MOMENTUM / self.cfg.TRAINING.END_MOMENTUM))
            self.model.tcn.set_bn_momentum(momentum)

            # save the best model
            if self.bestValidLoss > (runningLoss[self.metrcIdx] / numValidData):
                self.bestModelPath = self.save_model(epoch)
                self.bestValidLoss = (runningLoss[self.metrcIdx] / numValidData)
                print(f'Best Loss: {self.bestValidLoss}')

        # save the model in the last epoch
        _ = self.save_model(epoch)

    def create_model(self):
        self.model = OpenSimTemporalModel(
            self.pyBaseModel, self.coordinateValueRange, self.cfg
        ) \
            .float().to(self.COMP_DEVICE)
        # TODO: check
        # self.model.opensimTreeLayer.check_all_device(self.COMP_DEVICE)

        if self.bestModelPath is not None:
            self.model.load_state_dict(torch.load(self.bestModelPath))

    def report(self, dataset, runnigLoss, numData):
        s = ""
        s += f'{dataset} : '
        for i in range(len(self.lossNames)):
            s += f'{self.lossNames[i]} : {runnigLoss[i] / numData}   '

        print(s)

    def save_model(self, epoch):
        path = f'{self.modelFolder}model_{epoch}_{self.cfg.POSTFIX}.pt'
        torch.save(self.model.state_dict(), path)
        return path

    def model_forward(self, inf, evaluation=False, prediction=False, useTpose=False):

        for k, v in inf.items():
            inf[k] = v.to(self.COMP_DEVICE)

        # get input
        coordinateAngle = inf["coordinateAngle"]
        boneScale = inf["boneScale"]
        coordinateMask = inf["coordinateMask"]

        # not use masks during the evaluation
        pose3d = inf["pose3d"][:, -1, :, :]
        pose3d = pose3d - pose3d[:, 0:1, :]

        # forward

        outputs = self.model(inf)

        predBoneScale = outputs["predBoneScale"]
        predPos = outputs["predJointPos"]
        predMarkerPos = outputs["predMarkerPos"]
        predRot = outputs["predRot"]

        # substract the root
        predMarkerPos = predMarkerPos - predPos[:, 0:1, :]
        predPos = predPos - predPos[:, 0:1, :]

        # 3D joint position
        joint3dLossValue = self.positionLoss(
            predPos, pose3d[:, :self.cfg.NUM_JOINTS, :],
            evaluation=evaluation, mask=None
        ) * self.cfg.HYP.POS

        # 3D marker loss
        marker3dLossValue = self.marker_positionLoss(
            predMarkerPos, pose3d[:, self.cfg.NUM_JOINTS:, :],
            evaluation=evaluation, mask=None
        ) * self.cfg.HYP.MARKER

        losses = [joint3dLossValue, marker3dLossValue]

        # angle loss
        if self.cfg.LOSS.ANGLE.USE:
            angleLossValue = self.coordinate_angle_loss(predRot, coordinateAngle,
                                                        mask=coordinateMask if not evaluation and self.cfg.LOSS.ANGLE.USEMASK else None,
                                                        evaluation=evaluation) / math.pi * 180 * self.cfg.HYP.ANGLE
            losses.append(angleLossValue)

        # bone scale loss
        if self.cfg.LOSS.BODY.USE:
            bodyLossValue = self.bone_scale_loss(predBoneScale, boneScale, evaluation=evaluation) * self.cfg.HYP.BODY
            losses.append(bodyLossValue)

        # symmetric loss
        if self.cfg.LOSS.SY_BODY.USE:
            symmetricLossValue = self.symmetric_loss(predBoneScale,
                                                     predBoneScale[:, self.cfg.PREDICTION.MIRROR_BONES, :],
                                                     evaluation=evaluation) * self.cfg.HYP.SY_BODY
            losses.append(symmetricLossValue)

        if prediction:
            return outputs, losses
        else:
            return losses

    def store_every_frame_prediction(self, train=False, valid=False, test=True, useTpose=False):

        # reload the model
        print("Load ", self.bestModelPath)
        self.create_model()
        self.model.load_state_dict(torch.load(self.bestModelPath))
        self.model.eval()

        self.swapLoss(evaluation=True)


        if valid:
            dataset = BMLOpenSimTemporalDataSet(self.cfg, self.cfg.BML_FOLDER,
                                                       self.cfg.BML_FOLDER + "prediction_OS_valid.npy",
                                                       1, evaluation=True,
                                                       )
            datasetLoader = DataLoader(dataset, batch_size=cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                       drop_last=False,
                                       num_workers=self.cfg.EVAL_WORKERS)

            self.save_prediction(dataset, datasetLoader, 'valid', useTpose=useTpose)

        if test:
            dataset = BMLOpenSimTemporalDataSet(self.cfg, self.cfg.BML_FOLDER, self.cfg.BML_FOLDER + "prediction_OS_test.npy",
                                                 2, evaluation=True)
            datasetLoader = DataLoader(dataset, batch_size=cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                       drop_last=False,
                                       num_workers=self.cfg.EVAL_WORKERS)

            self.save_prediction(dataset, datasetLoader, 'test', useTpose=useTpose)

        if train:
            dataset = BMLOpenSimTemporalDataSet(self.cfg, self.cfg.BML_FOLDER,
                                                  self.cfg.BML_FOLDER + "prediction_OS_augmented_train.npy",
                                                  0, evaluation=True)
            datasetLoader = DataLoader(dataset, batch_size=cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                       drop_last=False,
                                       num_workers=self.cfg.EVAL_WORKERS)

            self.save_prediction(dataset, datasetLoader, 'augmented_train', useTpose=useTpose)

    def save_prediction(self, dataset, datasetLoader, name, useTpose=False):

        # prediction
        predBoneScale = np.zeros((len(dataset), len(self.cfg.PREDICTION.BODY), 3), dtype=np.float32)
        predPos = np.zeros((len(dataset), len(self.cfg.PREDICTION.JOINTS) + len(self.cfg.PREDICTION.LEAFJOINTS), 3),
                           dtype=np.float32)
        predMarkerPos = np.zeros((len(dataset), len(self.cfg.PREDICTION.MARKER), 3), dtype=np.float32)
        rootRot = np.zeros((len(dataset), 4, 4), dtype=np.float32)
        predRot = np.zeros((len(dataset), len(self.cfg.PREDICTION.COORDINATES)), dtype=np.float32)
        pose3d = np.zeros((len(dataset), len(self.cfg.PREDICTION.JOINTS) + len(self.cfg.PREDICTION.LEAFJOINTS) + len(
            self.cfg.PREDICTION.MARKER), 3), dtype=np.float32)

        with torch.no_grad():

            if self.cfg.PROGRESSBAR:
                iterator = enumerate(tqdm(datasetLoader))
            else:
                iterator = enumerate(datasetLoader, 0)

            for _, inf in iterator:
                # self.optimizer.zero_grad()
                fileIndices = inf["fileIdx"].cpu().detach().numpy()
                outputs, _ = self.model_forward(inf=inf, evaluation=True, prediction=True)
                predPos[fileIndices, :, :] = outputs["predJointPos"].cpu().detach().numpy()
                predMarkerPos[fileIndices, :, :] = outputs["predMarkerPos"].cpu().detach().numpy()
                predBoneScale[fileIndices, :, :] = outputs["predBoneScale"].cpu().detach().numpy()
                rootRot[fileIndices, :, :] = outputs["rootRot"].cpu().detach().numpy()
                predRot[fileIndices, :] = outputs["predRot"].cpu().detach().numpy()
                pose3d[fileIndices, :, :] = inf["pose3d"].cpu().detach().numpy()[:, -1, :, :]

        pose3d = pose3d - pose3d[:, :1, :]
        predMarkerPos = predMarkerPos - predPos[:, :1, :]
        predPos = predPos - predPos[:, :1, :]

        prediction = {
            "predPos": predPos,
            "predMarkerPos": predMarkerPos,
            "predBoneScale": predBoneScale,
            "rootRot": rootRot,
            "predRot": predRot,
            "pose3d": pose3d,
        }

        errorOS = np.mean(np.sum((pose3d[:, :self.cfg.NUM_JOINTS, :] - predPos) ** 2, axis=-1) ** 0.5)
        errorSMPLHJoint = np.mean(
            np.sum((pose3d[:, self.cfg.NUM_JOINTS:self.cfg.NUM_JOINTS + 22, :] - predMarkerPos[:, :22, :]) ** 2,
                   axis=-1) ** 0.5)
        errorMarker = np.mean(
            np.sum((pose3d[:, self.cfg.NUM_JOINTS + 22:, :] - predMarkerPos[:, 22:, :]) ** 2, axis=-1) ** 0.5)

        path = f'prediction_{self.cfg.POSTFIX}_{name}.npy'

        np.save(self.cfg.BML_FOLDER + path, prediction)

        bonyLandmarks = self.markerWeights.cpu().detach().numpy()
        errorBonyLandmarks = np.mean(
            np.sum(((pose3d[:, self.cfg.NUM_JOINTS:, :])[:, bonyLandmarks == 2, :] - predMarkerPos[:,
                                                                                     bonyLandmarks == 2, :]) ** 2,
                   axis=-1) ** 0.5)

        print(
            f' OS Joint error : {errorOS} , SMPLH Joint error : {errorSMPLHJoint} , Marker error : {errorMarker} , '
            f'BonyLandmark : {errorBonyLandmarks}')


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
        cfg = update_config(cfg)
    cfg.freeze()
    print(cfg)

    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    trainingProgram = Training(args, cfg)
    if not args.evaluation:
        trainingProgram.run()

    trainingProgram.store_every_frame_prediction(train=True, valid=True, test=True)
