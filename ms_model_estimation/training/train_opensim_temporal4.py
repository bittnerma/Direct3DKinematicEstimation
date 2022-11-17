import argparse
import math
import pickle
import random
import numpy as np
import torch
from ms_model_estimation.training.dataset.BMLOpenSimTemporalDataSet import BMLOpenSimTemporalDataSet
from ms_model_estimation.training.loss.CustomLoss import CustomLoss
from ms_model_estimation.training.networks.OpenSimTemporalModel import OpenSimTemporalModel
from ms_model_estimation.training.config.config_os_temporal import get_cfg_defaults, update_config
from ms_model_estimation.training.TorchTrainingProgram import TorchTrainingProgram
from torch.utils.data import DataLoader
from tqdm import tqdm


class Training(TorchTrainingProgram):

    def __init__(self, args, cfg):
        super(Training, self).__init__(args, cfg)

        # Dataset
        self.h5pyFolder = cfg.BML_FOLDER if cfg.BML_FOLDER.endswith("/") else cfg.BML_FOLDER + "/"

        self.trainSet = BMLOpenSimTemporalDataSet(cfg, self.h5pyFolder,
                                                  self.h5pyFolder + "os_train.npy", 0,
                                                  evaluation=False)

        self.validationSet = BMLOpenSimTemporalDataSet(cfg, self.h5pyFolder,
                                                       self.h5pyFolder + "os_valid.npy", 1, evaluation=True)

        self.testSet = BMLOpenSimTemporalDataSet(cfg, self.h5pyFolder, self.h5pyFolder + "os_test.npy",
                                                 2, evaluation=True)

        print("%d training data, %d valid data, %d test data" % (
            len(self.trainSet), len(self.validationSet), len(self.testSet)))

        self.trainLoader = DataLoader(self.trainSet, batch_size=cfg.TRAINING.BATCHSIZE, shuffle=True,
                                      num_workers=self.cfg.WORKERS)
        self.valLoader = DataLoader(self.validationSet, batch_size=cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=True,
                                    drop_last=False, num_workers=self.cfg.WORKERS)
        self.testLoader = DataLoader(self.testSet, batch_size=cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                     drop_last=False, num_workers=self.cfg.WORKERS)
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

        self.markerWeights = torch.Tensor(self.cfg.TRAINING.MARKER_WEIGHT).float().to(self.COMP_DEVICE).requires_grad_(
            False)

        # model
        self.create_model()

        if self.cfg.STARTMODELPATH is not None:
            self.load_model(self.cfg.STARTMODELPATH)

        # loss
        self.numLoss = 0
        self.lossNames = [
            "total loss",
        ]

        self.train_marker_positionLoss = CustomLoss.sequence_pose3d_mpjpe(root=False, L=self.cfg.LOSS.MARKER.TYPE,
                                                                          weights=self.markerWeights,
                                                                          beta=self.cfg.LOSS.MARKER.BETA)
        numBonyLandmarks = torch.sum(self.markerWeights == 2) * 1.0
        self.infer_marker_positionLoss = CustomLoss.sequence_pose3d_mpjpe(root=False, L=2,
                                                                          weights=(self.markerWeights == 2) * 1.0 *
                                                                                  self.markerWeights.shape[
                                                                                      0] / numBonyLandmarks)
        self.lossNames.append("marker")
        self.numLoss += 1
        self.metricLossIdx = [1]

        if self.cfg.LOSS.POS.USE:
            self.train_positionLoss = CustomLoss.sequence_pose3d_mpjpe(root=False, L=self.cfg.LOSS.POS.TYPE,
                                                                       beta=self.cfg.LOSS.POS.BETA)
            self.infer_positionLoss = CustomLoss.sequence_pose3d_mpjpe(root=False, L=2)
            self.numLoss += 1
            self.lossNames.append("pose")

        if self.cfg.LOSS.ANGLE.USE:
            self.numLoss += 1
            self.metricLossIdx = [3]

            # create coordinate weight
            maxWeight = self.cfg.TRAINING.COORDINATE_MAX_WEIGHT
            minRange, maxRange = torch.min(self.coordinateValueRange[:, 1]), torch.max(self.coordinateValueRange[:, 1])
            coeff = np.log(maxWeight) / (maxRange - minRange)
            self.coordinateWeights = torch.exp((self.coordinateValueRange[:, 1] - minRange) * coeff).to(
                self.COMP_DEVICE)

            self.ANGLE_BETA_INRADIAN = self.cfg.LOSS.ANGLE.BETA / 180 * math.pi
            self.angle_coef = torch.div(1.0, 2 * self.coordinateValueRange[:,
                                                 1]) * self.ANGLE_BETA_INRADIAN / self.cfg.LOSS.ANGLE.H
            self.train_coordinate_angle_loss = CustomLoss.opensim_sequence_coordinate_angle_loss(
                coef=self.angle_coef, coordinateWeights=self.coordinateWeights, lossType=self.cfg.LOSS.ANGLE.TYPE,
                beta=self.ANGLE_BETA_INRADIAN
            )
            self.infer_coordinate_angle_loss = CustomLoss.opensim_sequence_coordinate_angle_loss(
                coef=self.angle_coef, coordinateWeights=None, lossType=self.cfg.LOSS.ANGLE.EVAL_TYPE,
                beta=self.ANGLE_BETA_INRADIAN
            )
            self.lossNames.append("angle")

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
            self.train_bone_scale_loss = CustomLoss.sequence_bone_scale_loss(self.cfg.LOSS.BODY.TYPE,
                                                                             coef=self.body_coef,
                                                                             beta=self.cfg.LOSS.BODY.BETA)
            self.infer_bone_scale_loss = CustomLoss.sequence_bone_scale_loss(self.cfg.LOSS.BODY.EVAL_TYPE,
                                                                             coef=self.body_coef,
                                                                             beta=self.cfg.LOSS.BODY.BETA)
            self.lossNames.append("body")

        if self.cfg.MODEL.TYPE == 1 or self.cfg.MODEL.TYPE == 4:
            if self.cfg.LOSS.POS.USE:
                self.numLoss += 1
                self.lossNames.append("inter pose")
            if self.cfg.LOSS.MARKER.USE:
                self.numLoss += 1
                self.lossNames.append("inter marker")
            if self.cfg.LOSS.ANGLE.USE:
                self.numLoss += 1
                self.lossNames.append("inter angle")
            if self.cfg.LOSS.BODY.USE:
                self.numLoss += 1
                self.lossNames.append("inter body")

        '''
        if self.cfg.MODEL.TYPE == 2 or self.cfg.MODEL.TYPE == 0:

            # the output is single frame
            self.labelIndices = slice(self.cfg.MODEL.RECEPTIVE_FIELD - 1, self.cfg.MODEL.RECEPTIVE_FIELD, 1) \
                if self.cfg.MODEL.CAUSAL else slice(self.cfg.MODEL.RECEPTIVE_FIELD // 2,
                                                    self.cfg.MODEL.RECEPTIVE_FIELD // 2 + 1, 1)
            self.usedIndices = slice(0, 1, 1)

        elif self.cfg.MODEL.TYPE == 1:

            # the output is single frame
            self.labelIndices = slice(self.cfg.MODEL.RECEPTIVE_FIELD - 1, self.cfg.MODEL.RECEPTIVE_FIELD, 1) \
                if self.cfg.MODEL.CAUSAL else slice(self.cfg.MODEL.RECEPTIVE_FIELD // 2,
                                                    self.cfg.MODEL.RECEPTIVE_FIELD // 2 + 1, 1)
            self.usedIndices = slice(self.cfg.MODEL.RECEPTIVE_FIELD, self.cfg.MODEL.RECEPTIVE_FIELD + 1, 1)

        else:
            # the output is the sequence
            self.labelIndices = slice(0, self.cfg.MODEL.RECEPTIVE_FIELD, 1)
            self.usedIndices = slice(0, self.cfg.MODEL.RECEPTIVE_FIELD, 1)
        '''

        # index of target frame and label
        if self.cfg.MODEL.TYPE == 0:
            self.labelIndices = slice(self.cfg.MODEL.RECEPTIVE_FIELD - 1, self.cfg.MODEL.RECEPTIVE_FIELD,
                                      1) if self.cfg.MODEL.CAUSAL else slice(self.cfg.MODEL.RECEPTIVE_FIELD // 2,
                                                                             self.cfg.MODEL.RECEPTIVE_FIELD // 2 + 1, 1)
            self.usedIndices = slice(0, 1, 1)
        elif self.cfg.MODEL.TYPE == 2 or self.cfg.MODEL.TYPE == 1:
            self.labelIndices = slice(self.cfg.MODEL.RECEPTIVE_FIELD - 1, self.cfg.MODEL.RECEPTIVE_FIELD,
                                      1) if self.cfg.MODEL.CAUSAL else slice(self.cfg.MODEL.RECEPTIVE_FIELD // 2,
                                                                             self.cfg.MODEL.RECEPTIVE_FIELD // 2 + 1, 1)
            self.usedIndices = slice(self.cfg.MODEL.RECEPTIVE_FIELD, self.cfg.MODEL.RECEPTIVE_FIELD+1, 1)

        elif self.cfg.MODEL.TYPE == 4:
            # Use LSTM
            self.labelIndices = slice(self.cfg.MODEL.RECEPTIVE_FIELD - 1, self.cfg.MODEL.RECEPTIVE_FIELD,
                                      1) if self.cfg.MODEL.CAUSAL else slice(self.cfg.MODEL.RECEPTIVE_FIELD // 2,
                                                                             self.cfg.MODEL.RECEPTIVE_FIELD // 2 + 1, 1)
            self.usedIndices = self.labelIndices
            if self.cfg.MODEL.CAUSAL:
                self.intermediateUsedIndices = range(self.cfg.MODEL.RECEPTIVE_FIELD - 1)
            else:
                self.intermediateUsedIndices = []
                for i in range(self.cfg.MODEL.RECEPTIVE_FIELD):
                    if i != (self.cfg.MODEL.RECEPTIVE_FIELD // 2):
                        self.intermediateUsedIndices.append(i)
        else:
            # vanilla encoder
            # the output is the sequence
            self.labelIndices = range(self.cfg.MODEL.RECEPTIVE_FIELD)
            self.usedIndices = range(self.cfg.MODEL.RECEPTIVE_FIELD)

        '''
        if self.cfg.MODEL.TYPE == 0:
            # TCNs
            # set the momentum
            momentum = self.cfg.TRAINING.INITIAL_MOMENTUM
            self.model.temporalModel.temporalModel.set_bn_momentum(momentum)
            print(f'Momentum: {momentum}')'''

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def initialize_training(self, startEpoch, endEpoch, epochIdx):

        self.optimizer = torch.optim.Adam(self.params, lr=self.cfg.TRAINING.START_LR[epochIdx],
                                          amsgrad=self.cfg.TRAINING.AMSGRAD)

        self.decayRate = (self.cfg.TRAINING.END_LR[epochIdx] / self.cfg.TRAINING.START_LR[epochIdx]) ** (
                1 / (endEpoch - startEpoch + 1))
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)
        print(f'LR starts from {self.cfg.TRAINING.START_LR[epochIdx]}, LR decay rate : {self.decayRate}')

    def update(self, epoch, startEpoch, endEpoch, epochIdx):

        self.lr_scheduler.step()
        if epoch != (endEpoch - 1):
            print("Current LR: ", self.lr_scheduler.get_last_lr())

        if self.cfg.MODEL.TYPE == 0:
            # TCNs
            # set the momentum
            momentum = self.cfg.TRAINING.INITIAL_MOMENTUM * np.exp(
                - (epoch - startEpoch + 1) / (endEpoch - startEpoch + 1) * np.log(
                    self.cfg.TRAINING.INITIAL_MOMENTUM / self.cfg.TRAINING.END_MOMENTUM))
            self.model.temporalModel.temporalModel.set_bn_momentum(momentum)
            print(f'Epoch {epoch}, Momentum: {momentum}')

    def create_model(self):
        self.model = OpenSimTemporalModel(
            self.pyBaseModel, self.coordinateValueRange, self.bodyScaleValueRange, self.cfg
        ) \
            .float().to(self.COMP_DEVICE)
        # self.model.opensimTreeLayer.check_all_device(self.COMP_DEVICE)

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

    def model_forward(self, inf):

        # get input
        for k in inf:
            if isinstance(inf[k], torch.Tensor):
                inf[k] = inf[k].to(self.COMP_DEVICE)

        inf["pose3d"] = inf["pose3d"] - inf["pose3d"][:, :, 0:1, :]

        # forward
        outputs = self.model(inf)

        outputs["predMarkerPos"] = outputs["predMarkerPos"] - outputs["predJointPos"][:, :, 0:1, :]
        outputs["predJointPos"] = outputs["predJointPos"] - outputs["predJointPos"][:, :, 0:1, :]

        return outputs, inf

    def model_forward_and_calculate_loss(self, inf, evaluation=False):

        outputs, inf = self.model_forward(inf)

        predPos = outputs["predJointPos"]
        predRot = outputs["predRot"]
        predBoneScale = outputs["predBoneScale"]
        pose3d = inf["pose3d"]
        mask = inf["mask"] if not evaluation else None
        coordinateMask = inf["coordinateMask"] if not evaluation else None
        coordinateAngle = inf["coordinateAngle"]
        boneScale = inf["boneScale"]
        predMarkerPos = outputs["predMarkerPos"]

        losses = []

        # 3D marker loss
        marker3dLossValue = self.marker_positionLoss(
            predMarkerPos[:, self.usedIndices, :, :],
            pose3d[:, self.labelIndices, self.cfg.PREDICTION.NUM_JOINTS:, :],
            evaluation=evaluation,
            mask=mask[:, self.labelIndices,
                 self.cfg.PREDICTION.NUM_JOINTS:] if mask is not None else None
        ) * self.cfg.HYP.MARKER
        
        assert not torch.isnan(marker3dLossValue)
        losses.append(marker3dLossValue)

        # 3D joint position
        if self.cfg.LOSS.POS.USE:
            joint3dLossValue = self.positionLoss(
                predPos[:, self.usedIndices, :, :],
                pose3d[:, self.labelIndices, :self.cfg.PREDICTION.NUM_JOINTS, :],
                evaluation=evaluation,
                mask=mask[:, self.labelIndices,
                     :self.cfg.PREDICTION.NUM_JOINTS] if mask is not None else None
            ) * self.cfg.HYP.POS
            assert not torch.isnan(joint3dLossValue)
            losses.append(joint3dLossValue)

        # angle loss
        if self.cfg.LOSS.ANGLE.USE:
            angleLossValue = self.coordinate_angle_loss(predRot[:, self.usedIndices, :],
                                                        coordinateAngle[:, self.labelIndices, :],
                                                        mask=coordinateMask[:, self.labelIndices,
                                                             :] if not evaluation and self.cfg.LOSS.ANGLE.USEMASK else None,
                                                        evaluation=evaluation) / math.pi * 180 * self.cfg.HYP.ANGLE
            assert not torch.isnan(angleLossValue)
            losses.append(angleLossValue)

        # bone scale loss
        if self.cfg.LOSS.BODY.USE:
            bodyLossValue = self.bone_scale_loss(predBoneScale[:, self.usedIndices, :, :],
                                                 boneScale[:, self.labelIndices, :, :],
                                                 evaluation=evaluation) * self.cfg.HYP.BODY
            assert not torch.isnan(bodyLossValue)
            losses.append(bodyLossValue)

        # intermediate prediction
        if self.cfg.MODEL.TYPE == 1:

            # 3D joint position
            if self.cfg.LOSS.POS.USE:
                joint3dLossValue_intermediate = self.positionLoss(
                    predPos[:, :-1, :, :], pose3d[:, :, :self.cfg.PREDICTION.NUM_JOINTS, :],
                    evaluation=evaluation,
                    mask=mask[:, :, :self.cfg.PREDICTION.NUM_JOINTS] if mask is not None else None
                ) * self.cfg.HYP.POS
                assert not torch.isnan(joint3dLossValue_intermediate)
                losses.append(joint3dLossValue_intermediate)

            if self.cfg.LOSS.MARKER.USE:
                # 3D marker loss
                marker3dLossValue_intermediate = self.marker_positionLoss(
                    predMarkerPos[:, :-1, :, :], pose3d[:, :, self.cfg.PREDICTION.NUM_JOINTS:, :],
                    evaluation=evaluation,
                    mask=mask[:, :, self.cfg.PREDICTION.NUM_JOINTS:] if mask is not None else None
                ) * self.cfg.HYP.MARKER
                assert not torch.isnan(marker3dLossValue_intermediate)
                losses.append(marker3dLossValue_intermediate)

            # angle loss
            if self.cfg.LOSS.ANGLE.USE:
                angleLossValue_intermediate = self.coordinate_angle_loss(predRot[:, :-1, :],
                                                                         coordinateAngle,
                                                                         mask=coordinateMask if not evaluation and self.cfg.LOSS.ANGLE.USEMASK else None,
                                                                         evaluation=evaluation) / math.pi * 180 * self.cfg.HYP.ANGLE
                assert not torch.isnan(angleLossValue_intermediate)
                losses.append(angleLossValue_intermediate)

            # bone scale loss
            if self.cfg.LOSS.BODY.USE:
                bodyLossValue_intermediate = self.bone_scale_loss(predBoneScale[:, :-1, :, :],
                                                                  boneScale,
                                                                  evaluation=evaluation) * self.cfg.HYP.BODY
                assert not torch.isnan(bodyLossValue_intermediate)
                losses.append(bodyLossValue_intermediate)


        elif self.cfg.MODEL.TYPE == 4:

            # 3D joint position
            if self.cfg.LOSS.POS.USE:
                joint3dLossValue_intermediate = self.positionLoss(
                    predPos[:, self.intermediateUsedIndices, :, :],
                    pose3d[:, self.intermediateUsedIndices, :self.cfg.PREDICTION.NUM_JOINTS, :],
                    evaluation=evaluation,
                    mask=mask[:, self.intermediateUsedIndices,
                         :self.cfg.PREDICTION.NUM_JOINTS] if mask is not None else None
                ) * self.cfg.HYP.POS
                assert not torch.isnan(joint3dLossValue_intermediate)
                losses.append(joint3dLossValue_intermediate)

            if self.cfg.LOSS.MARKER.USE:
                # 3D marker loss
                marker3dLossValue_intermediate = self.marker_positionLoss(
                    predMarkerPos[:, self.intermediateUsedIndices, :, :],
                    pose3d[:, self.intermediateUsedIndices, self.cfg.PREDICTION.NUM_JOINTS:, :],
                    evaluation=evaluation,
                    mask=mask[:, self.intermediateUsedIndices,
                         self.cfg.PREDICTION.NUM_JOINTS:] if mask is not None else None
                ) * self.cfg.HYP.MARKER
                assert not torch.isnan(marker3dLossValue_intermediate)
                losses.append(marker3dLossValue_intermediate)

            # angle loss
            if self.cfg.LOSS.ANGLE.USE:
                angleLossValue_intermediate = self.coordinate_angle_loss(predRot[:, self.intermediateUsedIndices, :],
                                                                         coordinateAngle[:,
                                                                         self.intermediateUsedIndices, :],
                                                                         mask=coordinateMask[:,
                                                                              self.intermediateUsedIndices,
                                                                              :] if not evaluation and self.cfg.LOSS.ANGLE.USEMASK else None,
                                                                         evaluation=evaluation) / math.pi * 180 * self.cfg.HYP.ANGLE
                assert not torch.isnan(angleLossValue_intermediate)
                losses.append(angleLossValue_intermediate)

            # bone scale loss
            if self.cfg.LOSS.BODY.USE:
                bodyLossValue_intermediate = self.bone_scale_loss(predBoneScale[:, self.intermediateUsedIndices, :, :],
                                                                  boneScale[:, self.intermediateUsedIndices, :, :],
                                                                  evaluation=evaluation) * self.cfg.HYP.BODY
                assert not torch.isnan(bodyLossValue_intermediate)
                losses.append(bodyLossValue_intermediate)

        return losses

    def store_every_frame_prediction(self, train=False, valid=False, test=True):

        # reload the model
        print("Load ", self.bestModelPath)
        self.create_model()
        self.load_model(self.bestModelPath)
        self.model.eval()

        if valid:
            dataset = BMLOpenSimTemporalDataSet(cfg, self.h5pyFolder,
                                                self.h5pyFolder + "os_valid.npy", 1, evaluation=True)
            datasetLoader = DataLoader(dataset, batch_size=self.cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                       drop_last=False,
                                       num_workers=self.cfg.EVAL_WORKERS)

            self.save_prediction(dataset, datasetLoader, 'valid')

        if test:
            dataset = BMLOpenSimTemporalDataSet(cfg, self.h5pyFolder, self.h5pyFolder + "os_test.npy",
                                                2, evaluation=True)
            datasetLoader = DataLoader(dataset, batch_size=self.cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                       drop_last=False,
                                       num_workers=self.cfg.EVAL_WORKERS)

            self.save_prediction(dataset, datasetLoader, 'test')

        if train:
            dataset = BMLOpenSimTemporalDataSet(cfg, self.h5pyFolder,
                                                self.h5pyFolder + "os_train.npy", 0,
                                                evaluation=False)
            datasetLoader = DataLoader(dataset, batch_size=self.cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                       drop_last=False,
                                       num_workers=self.cfg.EVAL_WORKERS)

            self.save_prediction(dataset, datasetLoader, 'train')

    def save_prediction(self, dataset, datasetLoader, name):

        # prediction
        predBoneScale = np.zeros((len(dataset), len(self.cfg.PREDICTION.BODY), 3), dtype=np.float32)
        predPos = np.zeros((len(dataset), len(self.cfg.PREDICTION.JOINTS) + len(self.cfg.PREDICTION.LEAFJOINTS), 3),
                           dtype=np.float32)
        predMarkerPos = np.zeros((len(dataset), len(self.cfg.PREDICTION.MARKER), 3), dtype=np.float32)
        predRootRot = np.zeros((len(dataset), 4, 4), dtype=np.float32)
        predRot = np.zeros((len(dataset), len(self.cfg.PREDICTION.COORDINATES)), dtype=np.float32)

        labelBoneScale = np.zeros((len(dataset), len(self.cfg.PREDICTION.BODY), 3), dtype=np.float32)
        labelRot = np.zeros((len(dataset), len(self.cfg.PREDICTION.COORDINATES)), dtype=np.float32)
        pose3d = np.zeros((len(dataset), self.cfg.MODEL.NUMPRED, 3), dtype=np.float32)

        coordinateMask = np.zeros((len(dataset), len(self.cfg.PREDICTION.COORDINATES)), dtype=np.float32)
        mask = np.zeros((len(dataset), self.cfg.MODEL.NUMPRED), dtype=np.float32)

        labelIndices = -1 if self.cfg.MODEL.CAUSAL else self.cfg.MODEL.RECEPTIVE_FIELD // 2

        if (self.cfg.MODEL.TYPE == 4 or self.cfg.MODEL.TYPE == 2) and (not self.cfg.MODEL.CAUSAL):
            usedIndices = self.cfg.MODEL.RECEPTIVE_FIELD // 2
        else:
            # LSTM and no causal
            usedIndices = -1

        with torch.no_grad():

            if self.cfg.PROGRESSBAR:
                iterator = enumerate(tqdm(datasetLoader))
            else:
                iterator = enumerate(datasetLoader, 0)

            for _, inf in iterator:
                # self.optimizer.zero_grad()
                fileIndices = inf["fileIdx"].cpu().detach().numpy()

                outputs, _ = self.model_forward(inf)

                predPos[fileIndices, :, :] = outputs["predJointPos"][:, usedIndices, :, :].cpu().detach().numpy()
                predMarkerPos[fileIndices, :, :] = outputs["predMarkerPos"][:, usedIndices, :,
                                                   :].cpu().detach().numpy()
                predBoneScale[fileIndices, :, :] = outputs["predBoneScale"][:, usedIndices, :,
                                                   :].cpu().detach().numpy()
                predRootRot[fileIndices, :, :] = outputs["rootRot"][:, usedIndices, :, :].cpu().detach().numpy()
                predRot[fileIndices, :] = outputs["predRot"][:, usedIndices, :].cpu().detach().numpy()

                pose3d[fileIndices, :, :] = inf["pose3d"][:, labelIndices, :, :].cpu().detach().numpy()
                labelBoneScale[fileIndices, :, :] = inf["boneScale"][:, labelIndices, :, :].cpu().detach().numpy()
                labelRot[fileIndices, :] = inf["coordinateAngle"][:, labelIndices, :].cpu().detach().numpy()
                coordinateMask[fileIndices, :] = inf["coordinateMask"][:, labelIndices, :].cpu().detach().numpy()
                mask[fileIndices, :] = inf["mask"][:, labelIndices, :].cpu().detach().numpy()

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
    # else:
    #    trainingProgram.evaluate()
    trainingProgram.store_every_frame_prediction(train=True, valid=True, test=True)
