import argparse
import random
import numpy as np
import torch
from ms_model_estimation.models.dataset.BMLTemporalDataSet import BMLImgDataSet
from ms_model_estimation.models.loss.CustomLoss import CustomLoss
from ms_model_estimation.models.TemporalModel import TemporalPoseEstimationModel
from ms_model_estimation.models.config.config_bml_temporal import get_cfg_defaults, update_config
from ms_model_estimation.models.TorchTrainingProgram import TorchTrainingProgram
from torch.utils.data import DataLoader
from tqdm import tqdm


class Training(TorchTrainingProgram):

    def __init__(self, args, cfg):
        super(Training, self).__init__(args, cfg)

        # Dataset
        self.h5pyFolder = cfg.BML_FOLDER if cfg.BML_FOLDER.endswith("/") else cfg.BML_FOLDER + "/"
        self.trainSet = BMLImgDataSet(cfg, self.h5pyFolder,
                                      self.h5pyFolder + "prediction_train.npy", 0,
                                      evaluation=False)

        self.validationSet = BMLImgDataSet(cfg, self.h5pyFolder,
                                           self.h5pyFolder + "prediction_valid.npy", 1, evaluation=True)

        self.testSet = BMLImgDataSet(cfg, self.h5pyFolder, self.h5pyFolder + "prediction_test.npy",
                                     2, evaluation=True)

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
        if self.cfg.MODEL.TYPE == 1:
            self.lossNames.append("intermediate joint loss")
            self.lossNames.append("intermediate marker loss")
            self.numLoss += 2

        # model path
        if self.bestModelPath is not None:
            self.model.load_state_dict(torch.load(self.bestModelPath))

        # index of target frame and label
        if self.cfg.MODEL.TYPE == 2 or self.cfg.MODEL.TYPE == 1 or self.cfg.MODEL.TYPE == 0:
            self.labelIndices = self.cfg.MODEL.RECEPTIVE_FIELD - 1 if self.cfg.MODEL.CAUSAL else self.cfg.MODEL.RECEPTIVE_FIELD // 2
            self.usedIndices = 0
        elif self.cfg.MODEL.TYPE == 4:
            self.labelIndices = self.cfg.MODEL.RECEPTIVE_FIELD - 1 if self.cfg.MODEL.CAUSAL else self.cfg.MODEL.RECEPTIVE_FIELD // 2
            self.usedIndices = self.labelIndices
            if self.cfg.MODEL.CAUSAL:
                self.intermediateUsedIndices = range(self.cfg.MODEL.RECEPTIVE_FIELD)
            else:
                self.intermediateUsedIndices = []
                for i in range(self.cfg.MODEL.RECEPTIVE_FIELD):
                    if i != (self.cfg.MODEL.RECEPTIVE_FIELD // 2):
                        self.intermediateUsedIndices.append(i)
        else:
            # vanilla encoder
            self.labelIndices = range(self.cfg.MODEL.RECEPTIVE_FIELD)
            self.usedIndices = range(self.cfg.MODEL.RECEPTIVE_FIELD)

        if self.cfg.MODEL.TYPE == 0:
            # TCNs
            # set the momentum
            momentum = self.cfg.TRAINING.INITIAL_MOMENTUM
            self.model.temporalModel.set_bn_momentum(momentum)
            print(f'Momentum: {momentum}')

    def initialize_training(self, startEpoch, endEpoch, epochIdx):

        self.positionLoss = CustomLoss.pose3d_mpjpe(root=False, L=self.cfg.LOSS.POS_LOSS_TYPE)
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
            self.model.temporalModel.set_bn_momentum(momentum)
            print(f'Epoch {epoch}, Momentum: {momentum}')

    def create_model(self):
        self.model = TemporalPoseEstimationModel(
            self.cfg
        ) \
            .float().to(self.COMP_DEVICE)

    def model_forward(self, inf):

        # get input
        inputs = inf["pose3d"].to(self.COMP_DEVICE)
        pos3d = inf["label"].to(self.COMP_DEVICE)
        masks = inf["mask"].to(self.COMP_DEVICE)

        # forward
        inputs = inputs.view(inputs.shape[0], inputs.shape[1], self.cfg.MODEL.INFEATURES)
        if self.cfg.MODEL.TYPE == 1:
            pred = self.model(inputs)
            predPos = pred["predPos"]
            intermediatePos = pred["intermediatePos"]
        else:
            predPos = self.model(inputs)
            intermediatePos = None

        predPos = predPos.view(predPos.shape[0], predPos.shape[1], self.cfg.PREDICTION.NUMPRED, 3)
        if intermediatePos is not None:
            intermediatePos = intermediatePos.view(intermediatePos.shape[0], intermediatePos.shape[1],
                                                   self.cfg.PREDICTION.NUMPRED, 3)
            intermediatePos = intermediatePos - intermediatePos[:, :, :1, :]

        # move the root to original point
        predPos = predPos - predPos[:, :, 0:1, :]
        pos3d = pos3d - pos3d[:, :, 0:1, :]

        return predPos, intermediatePos, pos3d, masks

    def model_forward_and_calculate_loss(self, inf, evaluation=False):

        predPos, intermediatePos, pos3d, masks = self.model_forward(inf)

        # 3D joint position loss
        joint3dLossValue = self.positionLoss(predPos[:, self.usedIndices, :self.cfg.PREDICTION.NUM_JOINTS, :],
                                             pos3d[:, self.labelIndices, :self.cfg.PREDICTION.NUM_JOINTS, :],
                                             mask=masks[:, self.labelIndices,
                                                  :self.cfg.PREDICTION.NUM_JOINTS] if not evaluation else None,
                                             evaluation=evaluation) * \
                           self.cfg.HYP.POS

        assert not torch.isnan(joint3dLossValue)

        marker3dLossValue = self.positionLoss(predPos[:, self.usedIndices, self.cfg.PREDICTION.NUM_JOINTS:, :],
                                              pos3d[:, self.labelIndices, self.cfg.PREDICTION.NUM_JOINTS:, :],
                                              mask=masks[:, self.labelIndices,
                                                   self.cfg.PREDICTION.NUM_JOINTS:] if not evaluation else None,
                                              evaluation=evaluation) * \
                            self.cfg.HYP.MARKER

        assert not torch.isnan(marker3dLossValue)

        if evaluation and self.cfg.MODEL.TYPE == 3:
            marker3dLossValue = marker3dLossValue / self.cfg.MODEL.RECEPTIVE_FIELD
            joint3dLossValue = joint3dLossValue / self.cfg.MODEL.RECEPTIVE_FIELD
        losses = [joint3dLossValue, marker3dLossValue]

        if self.cfg.MODEL.TYPE == 1:
            intermediateJoint3dLossValue = self.positionLoss(intermediatePos[:, :, :self.cfg.PREDICTION.NUM_JOINTS, :],
                                                             pos3d[:, :, :self.cfg.PREDICTION.NUM_JOINTS, :],
                                                             mask=masks[:, :,
                                                                  :self.cfg.PREDICTION.NUM_JOINTS] if not evaluation else None,
                                                             evaluation=evaluation) * \
                                           self.cfg.HYP.POS
            intermediateMarkerLossValue = self.positionLoss(intermediatePos[:, :, self.cfg.PREDICTION.NUM_JOINTS:, :],
                                                            pos3d[:, :, self.cfg.PREDICTION.NUM_JOINTS:, :],
                                                            mask=masks[:, :,
                                                                 self.cfg.PREDICTION.NUM_JOINTS:] if not evaluation else None,
                                                            evaluation=evaluation) * \
                                          self.cfg.HYP.MARKER

            assert not torch.isnan(intermediateJoint3dLossValue)
            assert not torch.isnan(intermediateMarkerLossValue)

            if evaluation:
                intermediateJoint3dLossValue = intermediateJoint3dLossValue / self.cfg.MODEL.RECEPTIVE_FIELD
                intermediateMarkerLossValue = intermediateMarkerLossValue / self.cfg.MODEL.RECEPTIVE_FIELD
            losses.append(intermediateJoint3dLossValue)
            losses.append(intermediateMarkerLossValue)

        elif self.cfg.MODEL.TYPE == 4:

            intermediateJoint3dLossValue = self.positionLoss(predPos[:, self.intermediateUsedIndices, :self.cfg.PREDICTION.NUM_JOINTS, :],
                                                             pos3d[:, self.intermediateUsedIndices, :self.cfg.PREDICTION.NUM_JOINTS, :],
                                                             mask=masks[:, self.intermediateUsedIndices,
                                                                  :self.cfg.PREDICTION.NUM_JOINTS] if not evaluation else None,
                                                             evaluation=evaluation) * \
                                           self.cfg.HYP.POS
            intermediateMarkerLossValue = self.positionLoss(predPos[:, self.intermediateUsedIndices, self.cfg.PREDICTION.NUM_JOINTS:, :],
                                                            pos3d[:, self.intermediateUsedIndices, self.cfg.PREDICTION.NUM_JOINTS:, :],
                                                            mask=masks[:, self.intermediateUsedIndices,
                                                                 self.cfg.PREDICTION.NUM_JOINTS:] if not evaluation else None,
                                                            evaluation=evaluation) * \
                                          self.cfg.HYP.MARKER

            assert not torch.isnan(intermediateJoint3dLossValue)
            assert not torch.isnan(intermediateMarkerLossValue)

            if evaluation:
                intermediateJoint3dLossValue = intermediateJoint3dLossValue / (self.cfg.MODEL.RECEPTIVE_FIELD - 1)
                intermediateMarkerLossValue = intermediateMarkerLossValue / (self.cfg.MODEL.RECEPTIVE_FIELD - 1)
            losses.append(intermediateJoint3dLossValue)
            losses.append(intermediateMarkerLossValue)

        return losses

    def store_every_frame_prediction(self, train=False, valid=False, test=True):

        # reload the model
        print("Load ", self.bestModelPath)
        self.create_model()
        self.model.load_state_dict(torch.load(self.bestModelPath))
        self.model.eval()

        self.positionLoss = CustomLoss.pose3d_mpjpe(root=False, L=2)

        if train:
            dataset = BMLImgDataSet(self.cfg, self.h5pyFolder,
                                    self.h5pyFolder + "prediction_train.npy", 0,
                                    evaluation=True)
            datasetLoader = DataLoader(dataset, batch_size=self.cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                       drop_last=False,
                                       num_workers=self.cfg.EVAL_WORKERS)

            self.save_prediction(dataset, datasetLoader, 'train')

        if valid:
            dataset = BMLImgDataSet(self.cfg, self.h5pyFolder,
                                    self.h5pyFolder + "prediction_valid.npy", 1,
                                    evaluation=True)
            datasetLoader = DataLoader(dataset, batch_size=self.cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                       drop_last=False,
                                       num_workers=self.cfg.EVAL_WORKERS)

            self.save_prediction(dataset, datasetLoader, 'valid')

        if test:
            dataset = BMLImgDataSet(self.cfg, self.h5pyFolder, self.h5pyFolder + "prediction_test.npy",
                                    2, evaluation=True)
            datasetLoader = DataLoader(dataset, batch_size=self.cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                       drop_last=False,
                                       num_workers=self.cfg.EVAL_WORKERS)

            self.save_prediction(dataset, datasetLoader, 'test')

    def save_prediction(self, dataset, datasetLoader, name):

        # prediction
        prediction = np.zeros((len(dataset), self.cfg.PREDICTION.NUMPRED, 3), dtype=np.float32)
        labels = np.zeros((len(dataset), self.cfg.PREDICTION.NUMPRED, 3), dtype=np.float32)

        with torch.no_grad():
            if self.cfg.PROGRESSBAR:
                iterator = enumerate(tqdm(datasetLoader))
            else:
                iterator = enumerate(datasetLoader, 0)

            for _, inf in iterator:
                fileIndices = inf["fileIdx"]
                predPos, _, _, _ = self.model_forward(inf)
                prediction[fileIndices, :, :] = predPos.cpu().detach().numpy()[:, self.usedIndices, :, :]
                labels[fileIndices, :, :] = inf["label"].cpu().detach().numpy()[:, self.labelIndices, :, :]

        prediction = prediction - prediction[:, :1, :]
        labels = labels - labels[:, :1, :]

        outputs = {"prediction": prediction,
                   "label": labels}

        np.save(self.cfg.BML_FOLDER + f'prediction_temporal_{name}_{self.cfg.POSTFIX}.npy', outputs)
        jointError = np.mean(np.sum((prediction[:, :self.cfg.PREDICTION.NUM_JOINTS, :] - labels[:,
                                                                                         :self.cfg.PREDICTION.NUM_JOINTS,
                                                                                         :]) ** 2, axis=-1) ** 0.5)
        markerError = np.mean(np.sum((prediction[:, self.cfg.PREDICTION.NUM_JOINTS:, :] - labels[:,
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
