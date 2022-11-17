import argparse
import math
import pickle
import random
import time
import numpy as np
import torch
from ms_model_estimation.training.dataset.BMLImgOpenSimDataSet import BMLImgOpenSimDataSet
from ms_model_estimation.training.dataset.BMLImgTposeDataSet import BMLImgTposeDataSet
from ms_model_estimation.training.loss.CustomLoss import CustomLoss
from ms_model_estimation.training.networks.OpenSimModel import OpenSimModel
from ms_model_estimation.training.config.config_os import get_cfg_defaults, update_config
from torch.utils.data import DataLoader
from ms_model_estimation.training.utils.BMLUtils import CAMERA_TABLE
from ms_model_estimation.training.camera.cameralib import Camera
from tqdm import tqdm
from ms_model_estimation.training.TorchTrainingProgram import TorchTrainingProgram


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
        self.trainSet = BMLImgOpenSimDataSet(cfg, cfg.BML_FOLDER, cameraParamter, 0, evaluation=False)

        self.validationSet = BMLImgOpenSimDataSet(cfg, cfg.BML_FOLDER, cameraParamter, 1, evaluation=True)

        self.testSet = BMLImgOpenSimDataSet(cfg, cfg.BML_FOLDER, cameraParamter, 2, evaluation=True)

        print("%d training data, %d valid data, %d test data" % (
            len(self.trainSet), len(self.validationSet), len(self.testSet)))

        self.trainLoader = DataLoader(self.trainSet, batch_size=cfg.TRAINING.BATCHSIZE, shuffle=True,
                                      drop_last=True, num_workers=self.cfg.WORKERS)
        self.valLoader = DataLoader(self.validationSet, batch_size=cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=True,
                                    drop_last=False, num_workers=self.cfg.EVAL_WORKERS)
        self.testLoader = DataLoader(self.testSet, batch_size=cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                     drop_last=False, num_workers=self.cfg.EVAL_WORKERS)

        # load pyBaseModel
        self.pyBaseModel = pickle.load(open(self.cfg.BML_FOLDER + "pyBaseModel.pkl", "rb"))

        # coordinate value range        
        coordinateValueRange = torch.zeros(len(self.cfg.PREDICTION.COORDINATES), 2).float()
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
        self.coordinateValueRange = coordinateValueRange.to(self.COMP_DEVICE).requires_grad_(False)

        # body scale range
        bodyScaleValueRange = torch.zeros((self.cfg.PREDICTION.BODY_SCALE_UNIQUE_NUMS, 2))
        bodyScaleValueRange[:, 0] = torch.Tensor(self.cfg.PREDICTION.BODY_AVG_VALUE).float()
        bodyScaleValueRange[:, 1] = torch.Tensor(self.cfg.PREDICTION.BODY_AMPITUDE_VALUE).float()
        self.bodyScaleValueRange = bodyScaleValueRange.float().to(self.COMP_DEVICE).requires_grad_(False)
        self.bodyWeights = torch.Tensor(self.cfg.PREDICTION.BODY_WEIGHTS).float().to(self.COMP_DEVICE).requires_grad_(
            False)
        self.bodyWeights = self.bodyWeights / torch.sum(self.bodyWeights) * (self.bodyWeights.shape[0] * 3)

        self.markerWeights = torch.Tensor(self.cfg.TRAINING.MARKER_WEIGHT).float().to(self.COMP_DEVICE).requires_grad_(
            False)

        # best model
        self.bestValidLoss = math.inf
        self.bestModelPath = self.cfg.STARTMODELPATH

        # model
        self.create_model()
        self.load_model(self.bestModelPath)

        # loss
        self.numLoss = 0
        self.lossNames = [
            "total loss",
        ]

        self.metricLossIdx = [0]

        if self.cfg.LOSS.MARKER.USE:
            self.train_marker_positionLoss = CustomLoss.pose3d_mpjpe(root=False, L=self.cfg.LOSS.MARKER.TYPE,
                                                                     weights=self.markerWeights,
                                                                     beta=self.cfg.LOSS.MARKER.BETA)
            numBonyLandmarks = torch.sum(self.markerWeights == 2) * 1.0
            self.infer_marker_positionLoss = CustomLoss.pose3d_mpjpe(root=False, L=2,
                                                                     weights=(self.markerWeights == 2) * 1.0 *
                                                                             self.markerWeights.shape[
                                                                                 0] / numBonyLandmarks)
            self.lossNames.append("Marker")
            self.numLoss += 1
            self.metricLossIdx = [1]

        if self.cfg.LOSS.POS.USE:
            self.train_positionLoss = CustomLoss.pose3d_mpjpe(root=False, L=self.cfg.LOSS.POS.TYPE,
                                                              beta=self.cfg.LOSS.POS.BETA)
            self.infer_positionLoss = CustomLoss.pose3d_mpjpe(root=False, L=2)
            self.numLoss += 1
            self.lossNames.append("Joint")

        if self.cfg.LOSS.ANGLE.USE:
            self.numLoss += 1
            self.metricLossIdx = [3]
            
            # create coordinate weight
            maxWeight = self.cfg.TRAINING.COORDINATE_MAX_WEIGHT
            minRange, maxRange = torch.min(self.coordinateValueRange[:, 1]), torch.max(self.coordinateValueRange[:, 1])
            coeff = np.log(maxWeight) / (maxRange - minRange)
            self.coordinateWeights = torch.exp((self.coordinateValueRange[:, 1] - minRange) * coeff).to(
                self.COMP_DEVICE)

            # if self.cfg.LOSS.ANGLE.TYPE == 3:
            # self.train_coordinate_angle_loss = self.create_angle_loss(weights=True)
            self.ANGLE_BETA_INRADIAN = self.cfg.LOSS.ANGLE.BETA / 180 * math.pi
            self.angle_coef = torch.div(1.0, 2 * self.coordinateValueRange[:,
                                                 1]) * self.ANGLE_BETA_INRADIAN / self.cfg.LOSS.ANGLE.H
            self.train_coordinate_angle_loss = CustomLoss.opensim_coordinate_angle_loss(
                coef=self.angle_coef, coordinateWeights=self.coordinateWeights, lossType=self.cfg.LOSS.ANGLE.TYPE,
                beta=self.ANGLE_BETA_INRADIAN
            )
            self.infer_coordinate_angle_loss = CustomLoss.opensim_coordinate_angle_loss(
                coef=self.angle_coef, coordinateWeights=None, lossType=self.cfg.LOSS.ANGLE.EVAL_TYPE,
                beta=self.ANGLE_BETA_INRADIAN
            )
            self.lossNames.append("Angle")

        # loss for body scales
        if self.cfg.LOSS.BODY.USE:
            self.numLoss += 1
            self.bodyScaleValueRangeMapping = torch.empty((len(self.cfg.PREDICTION.BODY_SCALE_MAPPING), 3)).float().to(
                self.COMP_DEVICE).requires_grad_(False)
            for i in range(len(self.cfg.PREDICTION.BODY_SCALE_MAPPING)):
                for j in range(3):
                    self.bodyScaleValueRangeMapping[i, j] = self.bodyScaleValueRange[
                        self.cfg.PREDICTION.BODY_SCALE_MAPPING[i][j], 1]
            self.body_coef = torch.div(1.0,
                                       2 * self.bodyScaleValueRangeMapping) * self.cfg.LOSS.BODY.BETA / self.cfg.LOSS.BODY.H
            self.train_bone_scale_loss = CustomLoss.bone_scale_loss(self.cfg.LOSS.BODY.TYPE,
                                                                    weights=self.bodyWeights if self.cfg.TRAINING.USE_BODY_WEIGHTS else None,
                                                                    coef=self.body_coef,
                                                                    beta=self.cfg.LOSS.BODY.BETA)
            self.infer_bone_scale_loss = CustomLoss.bone_scale_loss(self.cfg.LOSS.BODY.EVAL_TYPE,
                                                                    coef=self.body_coef,
                                                                    beta=self.cfg.LOSS.BODY.BETA)
            self.lossNames.append("body")

        if self.cfg.LOSS.ROTMAT.USE:
            self.numLoss += 1
            self.train_rotMat_loss = CustomLoss.opensim_rotation_mat_loss(
                lossType=self.cfg.LOSS.ROTMAT.TYPE, beta=self.cfg.LOSS.ROTMAT.BETA
            )
            self.infer_rotMat_loss = CustomLoss.opensim_rotation_mat_loss(
                lossType=self.cfg.LOSS.ROTMAT.EVAL_TYPE, beta=self.cfg.LOSS.ROTMAT.BETA
            )
            self.lossNames.append("rootMat")

        assert self.numLoss > 0

    def swapLoss(self, evaluation=False):

        print(f'Evaluation : {evaluation}')

        if self.cfg.LOSS.POS.USE:
            self.positionLoss = self.train_positionLoss if not evaluation else self.infer_positionLoss
        if self.cfg.LOSS.MARKER.USE:
            self.marker_positionLoss = self.train_marker_positionLoss if not evaluation else self.infer_marker_positionLoss

        if self.cfg.LOSS.BODY.USE:
            self.bone_scale_loss = self.train_bone_scale_loss if not evaluation else self.infer_bone_scale_loss
        if self.cfg.LOSS.ANGLE.USE:
            self.coordinate_angle_loss = self.train_coordinate_angle_loss if not evaluation else self.infer_coordinate_angle_loss
        if self.cfg.LOSS.SY_BODY.USE:
            self.symmetric_loss = self.train_symmetric_loss if not evaluation else self.infer_symmetric_loss
        if self.cfg.LOSS.ROTMAT.USE:
            self.rotMat_loss = self.train_rotMat_loss if not evaluation else self.infer_rotMat_loss

    def initialize_training(self, startEpoch, endEpoch, epochIdx):

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

    def load_model(self, path):

        if path is not None:
            print(f'Load Pose Estimation Model : {path}')
            dicts = self.model.state_dict()
            weights = torch.load(path)
            for k, w in weights.items():
                if k in dicts:
                    dicts[k] = w
                else:
                    print(f'{k} is not in model')
            self.model.load_state_dict(dicts)

    def create_model(self):
        self.model = OpenSimModel(
            self.pyBaseModel, self.coordinateValueRange, self.bodyScaleValueRange, self.cfg
        ) \
            .float().to(self.COMP_DEVICE)
        print(f'Image size : {self.cfg.MODEL.IMGSIZE}')
        # TODO: check
        # self.model.opensimTreeLayer.check_all_device(self.COMP_DEVICE)

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

    def model_forward(self, inf, useTpose=False):

        # get input
        for k in inf:
            if isinstance(inf[k], torch.Tensor):
                inf[k] = inf[k].to(self.COMP_DEVICE)

        inf["pose3d"] = inf["pose3d"] - inf["pose3d"][:, 0:1, :]

        # forward
        if useTpose:
            outputs = self.model(inf["image"], inf["otherBodyPrediction"].to(self.COMP_DEVICE))
        else:
            outputs = self.model(inf["image"])

        outputs["predMarkerPos"] = outputs["predMarkerPos"] - outputs["predJointPos"][:, 0:1, :]
        outputs["predJointPos"] = outputs["predJointPos"] - outputs["predJointPos"][:, 0:1, :]

        return outputs, inf

    def model_forward_and_calculate_loss(self, inf, evaluation=False):

        outputs, inf = self.model_forward(inf)
        predPos = outputs["predJointPos"]
        predRot = outputs["predRot"]
        predBoneScale = outputs["predBoneScale"]
        pose3d = inf["pose3d"]
        mask = inf["joint3d_validity_mask"] if not evaluation else None
        coordinateMask = inf["coordinateMask"] if not evaluation else None
        coordinateAngle = inf["coordinateAngle"]
        boneScale = inf["boneScale"]
        predMarkerPos = outputs["predMarkerPos"]

        losses = []

        if self.cfg.LOSS.MARKER.USE:
            # 3D marker loss
            marker3dLossValue = self.marker_positionLoss(
                predMarkerPos, pose3d[:, self.cfg.PREDICTION.NUM_JOINTS:, :],
                evaluation=evaluation, mask=mask[:, self.cfg.PREDICTION.NUM_JOINTS:] if mask is not None else None
            ) * self.cfg.HYP.MARKER
            assert not torch.isnan(marker3dLossValue)
            losses.append(marker3dLossValue)
            # losses = [joint3dLossValue, marker3dLossValue]

        # 3D joint position
        if self.cfg.LOSS.POS.USE:
            joint3dLossValue = self.positionLoss(
                predPos, pose3d[:, :self.cfg.PREDICTION.NUM_JOINTS, :],
                evaluation=evaluation, mask=mask[:, :self.cfg.PREDICTION.NUM_JOINTS] if mask is not None else None
            ) * self.cfg.HYP.POS
            assert not torch.isnan(joint3dLossValue)
            losses.append(joint3dLossValue)

        # angle loss
        if self.cfg.LOSS.ANGLE.USE:
            angleLossValue = self.coordinate_angle_loss(predRot, coordinateAngle,
                                                        mask=coordinateMask if not evaluation and self.cfg.LOSS.ANGLE.USEMASK else None,
                                                        evaluation=evaluation) / math.pi * 180 * self.cfg.HYP.ANGLE
            assert not torch.isnan(angleLossValue)
            losses.append(angleLossValue)

        # bone scale loss
        if self.cfg.LOSS.BODY.USE:
            bodyLossValue = self.bone_scale_loss(predBoneScale, boneScale, evaluation=evaluation) * self.cfg.HYP.BODY
            assert not torch.isnan(bodyLossValue)
            losses.append(bodyLossValue)

        if self.cfg.LOSS.ROTMAT.USE:
            rotMatLossValue = self.rotMat_loss(outputs["rootRot"], inf["rootRotMat"],
                                               evaluation=evaluation) * self.cfg.HYP.ROTMAT
            assert not torch.isnan(rotMatLossValue)
            losses.append(rotMatLossValue)
        '''  
        if self.cfg.TRAINING.INTERMEDIATE_POSE_ESTIMATION:
            intermediatePos = outputs["intermediatePos"]
            intermediatePos = intermediatePos - intermediatePos[:, :1, :] + \
                              pose3d[:, self.cfg.PREDICTION.NUM_JOINTS:self.cfg.PREDICTION.NUM_JOINTS + 1, :]

            # 3D marker loss
            intermediateMarker3dLossValue = self.positionLoss(
                intermediatePos, pose3d[:, self.cfg.PREDICTION.NUM_JOINTS:, :],
                evaluation=evaluation, mask=mask[:, self.cfg.PREDICTION.NUM_JOINTS:] if mask is not None else None
            ) * self.cfg.HYP.INTERMEDIATE_MARKER

            losses.append(intermediateMarker3dLossValue)'''

        return losses

    def timeCost(self):

        # reload the model
        print("Load ", self.bestModelPath)
        self.create_model()
        self.load_model(self.bestModelPath)
        self.model.eval()
        self.swapLoss(evaluation=True)

        dataset = BMLImgOpenSimDataSet(self.cfg, self.cfg.BML_FOLDER, self.cameraParamter, 1,
                                       evaluation=True)

        # validation set
        for i in range(3, 10):
            batchSize = 2 ** i

            with torch.no_grad():
                datasetLoader = DataLoader(dataset, batch_size=batchSize, shuffle=False,
                                           drop_last=False,
                                           num_workers=self.cfg.EVAL_WORKERS if batchSize >= self.cfg.EVAL_WORKERS else batchSize)

                if self.cfg.PROGRESSBAR:
                    iterator = enumerate(tqdm(datasetLoader))
                else:
                    iterator = enumerate(datasetLoader, 0)

                startTime = time.time()

                for _, inf in iterator:
                    _ = self.model_forward(inf=inf, evaluation=True)

                endTime = time.time()

            print(
                f'BatchSize: {batchSize} , Time : {(endTime - startTime)} sec, FPS: {len(dataset) / (endTime - startTime)}')

    def construct_body_scales(self, inf):
        cameraIdxTable = {"PG1": 0, "PG2": 1}
        subjects = inf["subject"]
        cameraTypes = [cameraIdxTable[c] for c in inf["cameraType"]]
        otherBodyPrediction = np.empty((len(cameraTypes), len(self.cfg.PREDICTION.BODY), 3), dtype=np.float32)
        otherBodyPrediction[:, :, :] = self.boneScalesTpose[subjects - 1, cameraTypes, :, :].copy()
        inf["otherBodyPrediction"] = torch.from_numpy(otherBodyPrediction).float()
        return inf

    def predict_T_pose(self):

        # reload the model
        print("Load ", self.bestModelPath)
        self.create_model()
        self.load_model(self.bestModelPath)
        self.model.eval()

        dataset = BMLImgTposeDataSet(self.cfg, self.cfg.BML_FOLDER, self.cameraParamter)
        dataLoader = DataLoader(dataset, batch_size=cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                drop_last=False,
                                num_workers=self.cfg.EVAL_WORKERS)

        boneScalesTpose = np.zeros((90, 2, len(self.cfg.PREDICTION.BODY), 3), dtype=np.float32)

        if self.cfg.PROGRESSBAR:
            iterator = enumerate(tqdm(dataLoader))
        else:
            iterator = enumerate(dataLoader, 0)

        with torch.no_grad():
            for _, inf in iterator:
                outputs = self.model(inf["image"].to(self.COMP_DEVICE))
                cameraIndices = inf["cameraIdx"].detach().numpy()
                subjectIDs = inf["subjectID"].detach().numpy()
                boneScalesTpose[subjectIDs - 1, cameraIndices, :, :] = outputs["predBoneScale"].cpu().detach().numpy()

        self.boneScalesTpose = boneScalesTpose
        path = f'OS_prediction_Tpose.npy'
        np.save(self.cfg.BML_FOLDER + path, self.boneScalesTpose)
        self.store_every_frame_prediction(train=True, valid=True, test=True, useTpose=True)

    def store_every_frame_prediction(self, train=False, valid=False, test=True, useTpose=False):

        if useTpose:
            assert self.boneScalesTpose is not None

        # reload the model
        print("Load ", self.bestModelPath)
        self.create_model()
        self.load_model(self.bestModelPath)
        self.model.eval()

        if valid:
            dataset = BMLImgOpenSimDataSet(self.cfg, self.cfg.BML_FOLDER, self.cameraParamter, 1,
                                           evaluation=True)
            datasetLoader = DataLoader(dataset, batch_size=cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                       drop_last=False,
                                       num_workers=self.cfg.EVAL_WORKERS)

            self.save_prediction(dataset, datasetLoader, 'valid', useTpose=useTpose)

        if test:
            dataset = BMLImgOpenSimDataSet(self.cfg, self.cfg.BML_FOLDER, self.cameraParamter, 2,
                                           evaluation=True)
            datasetLoader = DataLoader(dataset, batch_size=cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                       drop_last=False,
                                       num_workers=self.cfg.EVAL_WORKERS)

            self.save_prediction(dataset, datasetLoader, 'test', useTpose=useTpose)

        if train:
            dataset = BMLImgOpenSimDataSet(self.cfg, self.cfg.BML_FOLDER, self.cameraParamter, 0,
                                           evaluation=True, useEveryFrame=True)
            datasetLoader = DataLoader(dataset, batch_size=cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                       drop_last=False,
                                       num_workers=self.cfg.EVAL_WORKERS)

            self.save_prediction(dataset, datasetLoader, 'train', useTpose=useTpose)

    def save_prediction(self, dataset, datasetLoader, name, useTpose=False):

        # prediction
        predBoneScale = np.zeros((len(dataset), len(self.cfg.PREDICTION.BODY), 3), dtype=np.float32)
        predPos = np.zeros((len(dataset), len(self.cfg.PREDICTION.JOINTS) + len(self.cfg.PREDICTION.LEAFJOINTS), 3),
                           dtype=np.float32)
        predMarkerPos = np.zeros((len(dataset), len(self.cfg.PREDICTION.MARKER), 3), dtype=np.float32)
        predRootRot = np.zeros((len(dataset), 3, 4), dtype=np.float32)
        predRot = np.zeros((len(dataset), len(self.cfg.PREDICTION.COORDINATES)), dtype=np.float32)

        labelBoneScale = np.zeros((len(dataset), len(self.cfg.PREDICTION.BODY), 3), dtype=np.float32)
        labelRot = np.zeros((len(dataset), len(self.cfg.PREDICTION.COORDINATES)), dtype=np.float32)
        labelRootRot = np.zeros((len(dataset), 3, 4), dtype=np.float32)
        pose3d = np.zeros((len(dataset), self.cfg.MODEL.NUMPRED, 3), dtype=np.float32)

        coordinateMask = np.zeros((len(dataset), len(self.cfg.PREDICTION.COORDINATES)), dtype=np.float32)
        mask = np.zeros((len(dataset), self.cfg.MODEL.NUMPRED), dtype=np.float32)

        with torch.no_grad():

            if self.cfg.PROGRESSBAR:
                iterator = enumerate(tqdm(datasetLoader))
            else:
                iterator = enumerate(datasetLoader, 0)

            for _, inf in iterator:
                # self.optimizer.zero_grad()
                fileIndices = inf["fileIdx"].cpu().detach().numpy()
                if useTpose:
                    inf = self.construct_body_scales(inf)

                outputs, _ = self.model_forward(inf, useTpose=useTpose)

                predPos[fileIndices, :, :] = outputs["predJointPos"].cpu().detach().numpy()
                predMarkerPos[fileIndices, :, :] = outputs["predMarkerPos"].cpu().detach().numpy()
                predBoneScale[fileIndices, :, :] = outputs["predBoneScale"].cpu().detach().numpy()
                predRootRot[fileIndices, :, :] = outputs["rootRot"][:, :3, :].cpu().detach().numpy()
                predRot[fileIndices, :] = outputs["predRot"].cpu().detach().numpy()
                pose3d[fileIndices, :, :] = inf["pose3d"].cpu().detach().numpy()
                labelBoneScale[fileIndices, :, :] = inf["boneScale"].cpu().detach().numpy()
                labelRot[fileIndices, :] = inf["coordinateAngle"].cpu().detach().numpy()
                coordinateMask[fileIndices, :] = inf["coordinateMask"].cpu().detach().numpy()
                mask[fileIndices, :] = inf["joint3d_validity_mask"].cpu().detach().numpy()
                labelRootRot[fileIndices, :, :] = inf["rootRotMat"][:, :3, :].cpu().detach().numpy()

        pose3d = pose3d - pose3d[:, :1, :]
        predMarkerPos = predMarkerPos - predPos[:, :1, :]
        predPos = predPos - predPos[:, :1, :]

        prediction = {
            "predPos": predPos,
            "predMarkerPos": predMarkerPos,
            "predBoneScale": predBoneScale,
            "predRootRot": predRootRot,
            "predRot": predRot,
            "pose3d": pose3d,
            "labelRot": labelRot,
            "labelBoneScale": labelBoneScale,
            "coordinateMask": coordinateMask,
            "mask": mask,
        }

        errorOS = np.mean(np.sum((pose3d[:, :self.cfg.PREDICTION.NUM_JOINTS, :] - predPos) ** 2, axis=-1) ** 0.5)
        errorSMPLHJoint = np.mean(
            np.sum((pose3d[:, self.cfg.PREDICTION.NUM_JOINTS:self.cfg.PREDICTION.NUM_JOINTS + 22, :] - predMarkerPos[:,
                                                                                                       :22, :]) ** 2,
                   axis=-1) ** 0.5)
        errorMarker = np.mean(
            np.sum((pose3d[:, self.cfg.PREDICTION.NUM_JOINTS + 22:, :] - predMarkerPos[:, 22:, :]) ** 2,
                   axis=-1) ** 0.5)

        if useTpose:
            path = f'prediction_{self.cfg.POSTFIX}_{name}_Tpose.npy'
        else:
            path = f'prediction_{self.cfg.POSTFIX}_{name}.npy'

        np.save(self.cfg.BML_FOLDER + path, prediction)

        bonyLandmarks = self.markerWeights.cpu().detach().numpy()
        errorBonyLandmarks = np.mean(
            np.sum(((pose3d[:, self.cfg.PREDICTION.NUM_JOINTS:, :])[:, bonyLandmarks == 2, :] - predMarkerPos[:,
                                                                                                bonyLandmarks == 2,
                                                                                                :]) ** 2,
                   axis=-1) ** 0.5)

        print(
            f' OS Joint error : {errorOS} , SMPLH Joint error : {errorSMPLHJoint} , Marker error : {errorMarker} , '
            f'BonyLandmark : {errorBonyLandmarks}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cpu', action='store_true', default=False, help="only use cpu?")
    parser.add_argument('--evaluation', action='store_true', default=False, help="only use cpu?")
    parser.add_argument('--ymlFile', action='store',
                        default="", type=str,
                        help="The hdf5 folder")
    parser.add_argument('--testTime', action='store_true', default=False, help="only use cpu?")
    parser.add_argument('--Tpose', action='store_true', default=False, help="Tpose")

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
    if args.evaluation:
        trainingProgram.store_every_frame_prediction(train=True, valid=True, test=True)
    elif args.testTime:
        trainingProgram.timeCost()
    elif args.Tpose:
        trainingProgram.predict_T_pose()
    else:
        trainingProgram.run()
        trainingProgram.store_every_frame_prediction(train=True, valid=True, test=True)

    # trainingProgram.predict_T_pose()
    # trainingProgram.store_every_frame_prediction(train=False, valid=True, test=True, useTpose=True)
