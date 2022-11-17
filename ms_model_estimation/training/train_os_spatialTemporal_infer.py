import argparse
import math
import pickle
import random
import numpy as np
import torch
from ms_model_estimation.training.dataset.BMLImgVideoDataSet import BMLImgDataSet
from ms_model_estimation.training.networks.OpenSimModel import OpenSimModel
from ms_model_estimation.training.networks.OpenSimTemporalModel import OpenSimTemporalModel
from ms_model_estimation.training.config.config_os_spatialtemporal_time import get_cfg_defaults
from torch.utils.data import DataLoader
from ms_model_estimation.training.utils.BMLUtils import CAMERA_TABLE
from ms_model_estimation.training.camera.cameralib import Camera
from tqdm import tqdm, trange
import time
from ms_model_estimation.training.TorchTrainingProgram import TorchTrainingProgram
from pathlib import Path


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
        self.h5pyFolder = self.cfg.BML_FOLDER if self.cfg.BML_FOLDER.endswith("/") else self.cfg.BML_FOLDER + "/"

        # load pyBaseModel
        self.pyBaseModel = pickle.load(open(Path(self.cfg.BML_FOLDER) / "pyBaseModel.pkl", "rb"))

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

        # body scale range
        bodyScaleValueRange = torch.zeros((self.cfg.PREDICTION.BODY_SCALE_UNIQUE_NUMS, 2))
        bodyScaleValueRange[:, 0] = torch.Tensor(self.cfg.PREDICTION.BODY_AVG_VALUE).float()
        bodyScaleValueRange[:, 1] = torch.Tensor(self.cfg.PREDICTION.BODY_AMPITUDE_VALUE).float()
        self.bodyScaleValueRange = bodyScaleValueRange.float().to(self.COMP_DEVICE).requires_grad_(False)

        # model
        self.create_model()

        # pretrained pose estimation model
        if self.cfg.STARTPOSMODELPATH is not None:
            print(f'Load Pose Estimation Model : {self.cfg.STARTPOSMODELPATH}')
            dicts = self.spatial_model.state_dict()
            weights = torch.load(self.cfg.STARTPOSMODELPATH,map_location=torch.device(self.COMP_DEVICE))
            for k, w in weights.items():
                if k in dicts:
                    dicts[k] = w
                else:
                    print(f'{k} is not in model')
            self.spatial_model.load_state_dict(dicts)
        else:
            assert False

        # pretrained temporal model
        if self.cfg.STARTTEMPORALMODELPATH is not None:
            print(f'Load Temporal Estimation Model : {self.cfg.STARTTEMPORALMODELPATH}')
            self.temporal_model.load_state_dict(torch.load(self.cfg.STARTTEMPORALMODELPATH,map_location=torch.device(self.COMP_DEVICE)))
        else:
            assert False

        self.spatial_model.eval()
        self.temporal_model.eval()

    def create_model(self):
        self.spatial_model = OpenSimModel(
            self.pyBaseModel, self.coordinateValueRange, self.bodyScaleValueRange, self.cfg
        ) \
            .float().to(self.COMP_DEVICE)
        self.spatial_model.eval()

        self.temporal_model = OpenSimTemporalModel(
            self.pyBaseModel, self.coordinateValueRange, self.bodyScaleValueRange, self.cfg
        ) \
            .float().to(self.COMP_DEVICE)
        self.temporal_model.eval()

    def spatial_model_forward(self, inf):

        with torch.no_grad():
            # get input
            image = inf["image"].float().to(self.COMP_DEVICE)

            # forward
            outputs = self.spatial_model(image)

        return outputs

    def temproal_model_forward(self, inf):

        with torch.no_grad():
            for k, w in inf.items():
                inf[k] = w.to(self.COMP_DEVICE)
            outputs = self.temporal_model(inf)

        return outputs

    def setup_prediction_variables(self):
        predBoneScale = torch.zeros((len(self.testSet), len(self.cfg.PREDICTION.BODY), 3), dtype=torch.float32)
        predPos = torch.zeros(
            (len(self.testSet), len(self.cfg.PREDICTION.JOINTS) + len(self.cfg.PREDICTION.LEAFJOINTS), 3),
            dtype=torch.float32)
        predMarkerPos = torch.zeros((len(self.testSet), len(self.cfg.PREDICTION.MARKER), 3), dtype=torch.float32)
        predRootRot = torch.zeros((len(self.testSet), 3, 3), dtype=torch.float32)
        predRot = torch.zeros((len(self.testSet), len(self.cfg.PREDICTION.COORDINATES)), dtype=torch.float32)
        return predBoneScale,predPos,predMarkerPos,predRootRot,predRot

    def run_windowed_temporal_predictions(self,predBoneScale,predPos,predMarkerPos,predRootRot,predRot):
        for videoIdx in self.testSet.videoTable:
            usedIndices = np.array(self.testSet.videoTable[videoIdx])

            windowPredPos = torch.zeros(self.cfg.MODEL.RECEPTIVE_FIELD, predPos.shape[1], 3)
            windowPredMarkerPos = torch.zeros(self.cfg.MODEL.RECEPTIVE_FIELD, predMarkerPos.shape[1], 3)
            windowPredBoneScale = torch.zeros(self.cfg.MODEL.RECEPTIVE_FIELD, predBoneScale.shape[1], 3)
            windowPredRootRot = torch.zeros(self.cfg.MODEL.RECEPTIVE_FIELD, 3, 3)
            windowPredRot = torch.zeros(self.cfg.MODEL.RECEPTIVE_FIELD, predRot.shape[1])

            windowPredPos[:self.cfg.MODEL.RECEPTIVE_FIELD // 2, :, :] = predPos[usedIndices[0], :, :]
            windowPredMarkerPos[:self.cfg.MODEL.RECEPTIVE_FIELD // 2, :, :] = predMarkerPos[usedIndices[0], :, :]
            windowPredBoneScale[:self.cfg.MODEL.RECEPTIVE_FIELD // 2, :, :] = predBoneScale[usedIndices[0], :, :]
            windowPredRootRot[:self.cfg.MODEL.RECEPTIVE_FIELD // 2, :, :] = predRootRot[usedIndices[0], :, :]
            windowPredRot[:self.cfg.MODEL.RECEPTIVE_FIELD // 2, :] = predRot[usedIndices[0], :]

            if len(usedIndices) >= self.cfg.MODEL.RECEPTIVE_FIELD // 2:
                windowPredPos[self.cfg.MODEL.RECEPTIVE_FIELD // 2:, :, :] = predPos[usedIndices[:(
                        self.cfg.MODEL.RECEPTIVE_FIELD - self.cfg.MODEL.RECEPTIVE_FIELD // 2)], :, :]
                windowPredMarkerPos[self.cfg.MODEL.RECEPTIVE_FIELD // 2:, :, :] = predMarkerPos[usedIndices[:(
                        self.cfg.MODEL.RECEPTIVE_FIELD - self.cfg.MODEL.RECEPTIVE_FIELD // 2)], :, :]
                windowPredBoneScale[self.cfg.MODEL.RECEPTIVE_FIELD // 2:, :, :] = predBoneScale[usedIndices[:(
                        self.cfg.MODEL.RECEPTIVE_FIELD - self.cfg.MODEL.RECEPTIVE_FIELD // 2)], :, :]
                windowPredRootRot[self.cfg.MODEL.RECEPTIVE_FIELD // 2:, :, :] = predRootRot[usedIndices[:(
                        self.cfg.MODEL.RECEPTIVE_FIELD - self.cfg.MODEL.RECEPTIVE_FIELD // 2)], :, :]
                windowPredRot[self.cfg.MODEL.RECEPTIVE_FIELD // 2:, :] = predRot[usedIndices[:(
                        self.cfg.MODEL.RECEPTIVE_FIELD - self.cfg.MODEL.RECEPTIVE_FIELD // 2)], :]

                endIdx = (self.cfg.MODEL.RECEPTIVE_FIELD - self.cfg.MODEL.RECEPTIVE_FIELD // 2)
            else:
                windowPredPos[
                self.cfg.MODEL.RECEPTIVE_FIELD // 2:self.cfg.MODEL.RECEPTIVE_FIELD // 2 + len(usedIndices), :,
                :] = predPos[usedIndices, :, :]
                windowPredMarkerPos[
                self.cfg.MODEL.RECEPTIVE_FIELD // 2:self.cfg.MODEL.RECEPTIVE_FIELD // 2 + len(usedIndices), :,
                :] = predMarkerPos[usedIndices, :, :]
                windowPredBoneScale[
                self.cfg.MODEL.RECEPTIVE_FIELD // 2:self.cfg.MODEL.RECEPTIVE_FIELD // 2 + len(usedIndices), :,
                :] = predBoneScale[usedIndices, :, :]
                windowPredRootRot[
                self.cfg.MODEL.RECEPTIVE_FIELD // 2:self.cfg.MODEL.RECEPTIVE_FIELD // 2 + len(usedIndices), :,
                :] = predRootRot[usedIndices, :, :]
                windowPredRot[
                self.cfg.MODEL.RECEPTIVE_FIELD // 2:self.cfg.MODEL.RECEPTIVE_FIELD // 2 + len(usedIndices),
                :] = predRot[usedIndices, :]

                endIdx = len(usedIndices) - 1

            for currentIdx in trange(0, len(usedIndices), batchSize):

                if currentIdx + batchSize >= len(usedIndices):
                    currentIdx = slice(currentIdx, len(usedIndices), 1)
                else:
                    currentIdx = slice(currentIdx, currentIdx + batchSize, 1)

                inf = {}
                inf["predPos"] = torch.empty(
                    (len(usedIndices[currentIdx]), self.cfg.MODEL.RECEPTIVE_FIELD,
                        predPos.shape[1] + predMarkerPos.shape[1], 3))
                inf["predBoneScale"] = torch.empty(
                    (len(usedIndices[currentIdx]), self.cfg.MODEL.RECEPTIVE_FIELD, predBoneScale.shape[1], 3))
                inf["predRootRot"] = torch.empty((len(usedIndices[currentIdx]), self.cfg.MODEL.RECEPTIVE_FIELD,
                                                    predRootRot.shape[1], predRootRot.shape[2]))
                inf["predRot"] = torch.empty(
                    (len(usedIndices[currentIdx]), self.cfg.MODEL.RECEPTIVE_FIELD, predRot.shape[1]))

                for i in range(batchSize):
                    inf["predPos"][i, :, :predPos.shape[1], :] = windowPredPos.clone()
                    inf["predPos"][i, :, predPos.shape[1]:, :] = windowPredMarkerPos.clone()
                    inf["predBoneScale"][i, :, :, :] = windowPredBoneScale.clone()
                    inf["predRootRot"][i, :, :, :] = windowPredRootRot.clone()
                    inf["predRot"][i, :, :] = windowPredRot.clone()

                    windowPredPos[:-1, :, :] = windowPredPos[1:, :, :].clone()
                    windowPredMarkerPos[:-1, :, :] = windowPredMarkerPos[1:, :, :].clone()
                    windowPredBoneScale[:-1, :, :] = windowPredBoneScale[1:, :, :].clone()
                    windowPredRootRot[:-1, :, :] = windowPredRootRot[1:, :, :].clone()
                    windowPredRot[:-1, :] = windowPredRot[1:, :].clone()

                    endIdx += 1
                    if endIdx < len(usedIndices):
                        windowPredPos[-1, :, :] = predPos[usedIndices[endIdx], :, :].clone()
                        windowPredMarkerPos[-1, :, :] = predMarkerPos[usedIndices[endIdx], :, :].clone()
                        windowPredBoneScale[-1, :, :] = predBoneScale[usedIndices[endIdx], :, :].clone()
                        windowPredRootRot[-1, :, :] = predRootRot[usedIndices[endIdx], :, :].clone()
                        windowPredRot[-1, :] = predRot[usedIndices[endIdx], :].clone()

                outputs = self.temproal_model_forward(inf)
                predPos[usedIndices[currentIdx], :, :] = outputs["predJointPos"][:, -1, :,
                                                            :].cpu().detach()  # .numpy()
                predMarkerPos[usedIndices[currentIdx], :, :] = outputs["predMarkerPos"][:, -1, :, :].cpu().detach()
                predBoneScale[usedIndices[currentIdx], :, :] = outputs["predBoneScale"][:, -1, :, :].cpu().detach()
                predRootRot[usedIndices[currentIdx], :, :] = outputs["rootRot"][:, -1, :3, :3].cpu().detach()
                predRot[usedIndices[currentIdx], :] = outputs["predRot"][:, -1, :].cpu().detach().detach()
        
        return predBoneScale,predPos,predMarkerPos,predRootRot,predRot

    def run_spatial_prediction(self,predBoneScale,predPos,predMarkerPos,predRootRot,predRot):
        for inf in tqdm(self.testLoader):
            outputs = self.spatial_model_forward(inf)
            fileIndices = list(inf["globalDataIdx"].numpy())
            predPos[fileIndices, :, :] = outputs["predJointPos"].cpu().detach()
            predMarkerPos[fileIndices, :, :] = outputs["predMarkerPos"].cpu().detach()
            predBoneScale[fileIndices, :, :] = outputs["predBoneScale"].cpu().detach()
            predRootRot[fileIndices, :, :] = outputs["rootRot"][:, :3, :3].cpu().detach()
            predRot[fileIndices, :] = outputs["predRot"].cpu().detach().detach()

        predPos = predPos - predPos[:, :1, :]

        return predBoneScale,predPos,predMarkerPos,predRootRot,predRot

    def run_inference(self,batchSize = 1,datasplit='test'):
        
        datasplit_dict = {"test":2,"validation":1,"training":0,"custom":3}        

        assert(datasplit in datasplit_dict)
        
        self.testSet = BMLImgDataSet(self.cfg, self.h5pyFolder, self.cameraParamter, datasplit_dict[datasplit], evaluation=True)
        self.testLoader = DataLoader(self.testSet, batch_size=batchSize, shuffle=False,
                                        drop_last=False,
                                        num_workers=self.cfg.WORKERS if batchSize >= self.cfg.WORKERS else batchSize)

        print(f'{len(self.testSet)} test data')
        print(f'{len(self.testSet.videoTable)} video')

        # prediction
        predBoneScale,predPos,predMarkerPos,predRootRot,predRot = self.setup_prediction_variables()

        predBoneScale,predPos,predMarkerPos,predRootRot,predRot = self.run_spatial_prediction(predBoneScale, predPos, predMarkerPos, predRootRot, predRot)

        predBoneScale,predPos,predMarkerPos,predRootRot,predRot = self.run_windowed_predictions(predBoneScale,predPos,predMarkerPos,predRootRot,predRot)

        final_prediction = {"predPos":predPos, "predMarkerPos":predMarkerPos,"predBoneScale":predBoneScale,"predRootRot":predRootRot,"predRot":predRot}
        np.save("predictions",final_prediction)



    def evaluate_time_cost(self):

        # batchSize = 1
        for batchSize in [1, 16, 64, 128, 256]:
            self.testSet = BMLImgDataSet(self.cfg, self.h5pyFolder, self.cameraParamter, 3, evaluation=True)
            self.testLoader = DataLoader(self.testSet, batch_size=batchSize, shuffle=False,
                                         drop_last=False,
                                         num_workers=self.cfg.WORKERS if batchSize >= self.cfg.WORKERS else batchSize)

            print(f'{len(self.testSet)} test data')
            print(f'{len(self.testSet.videoTable)} video')

            # prediction
            predBoneScale,predPos,predMarkerPos,predRootRot,predRot = self.setup_prediction_variables()

            totalStartTime = time.time()

            predBoneScale,predPos,predMarkerPos,predRootRot,predRot = self.run_spatial_prediction(predBoneScale, predPos, predMarkerPos, predRootRot, predRot)

            startImgTime = time.time()
            # spatial prediction
            
            endImgTime = time.time()

            predBoneScale,predPos,predMarkerPos,predRootRot,predRot = self.run_windowed_predictions(predBoneScale,predPos,predMarkerPos,predRootRot,predRot)

            startTemporalTime = time.time()
            # temporal prediction
            

            endTemporalTime = time.time()
            totalEndingTime = time.time()

            print(f'batchSize : {batchSize}',
                  f'Total Time : {totalEndingTime - totalStartTime} ,'
                  f' Spatial Time: {endImgTime - startImgTime} ,'
                  f' Temporal Time : {endTemporalTime - startTemporalTime}'
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
        # self.cfg = update_config(self.cfg)
    cfg.freeze()
    print(cfg)

    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    trainingProgram = Training(args, cfg)
    trainingProgram.evaluate_time_cost()
