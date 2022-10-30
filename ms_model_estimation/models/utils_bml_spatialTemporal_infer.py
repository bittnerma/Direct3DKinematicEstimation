import argparse
import random
import numpy as np
import opensim
import torch
from ms_model_estimation.models.dataset.BMLImgVideoDataSet import BMLImgDataSet
from ms_model_estimation.models.PoseEstimationModel import PoseEstimationModel
from ms_model_estimation.models.TemporalModel import TemporalPoseEstimationModel
from ms_model_estimation.models.config.config_bml_spatialtemporal_time import get_cfg_defaults
from torch.utils.data import DataLoader
import os
from pathlib import Path
from ms_model_estimation.models.BMLUtils import CAMERA_TABLE
from ms_model_estimation.models.camera.cameralib import Camera
from tqdm import tqdm, trange
import time
from ms_model_estimation.models.TorchTrainingProgram import TorchTrainingProgram
from ms_model_estimation.smplh.scalingIKInf import IKTaskSet, scalingIKSet, scaleSet
from opensim.DataReader import DataReader
#from ms_model_estimation.OpenSimModel import OpenSimModel
from ms_model_estimation.pyOpenSim.TrcGenerator import TrcGenerator
import multiprocessing as mp

virtualMarkers = {
    "MASI": ["RASI", "LASI"],
    "MPSI": ["RPSI", "LPSI"],
    "MHIPJ": ["LHIPJ", "RHIPJ"],
    "MSHO": ["RSHO", "LSHO"],
    "MNECK": ["C7", "CLAV"],
    "MRIBS": ["T10", "STRN"],
    "CENTER": ["RASI", "LASI", "RPSI", "LPSI"],
    "Virtual_LWRIST": ["LIWR", "LOWR"],
    "Virtual_RWRIST": ["RIWR", "ROWR"],
    "Virtual_LAJC": ["LANK", "LANI"],
    "Virtual_RAJC": ["RANK", "RANI"],
    "Virtual_LKJC": ["LKNE", "LKNI"],
    "Virtual_RKJC": ["RKNE", "RKNI"],
    "Virtual_LELC": ["LELB", "LELBIN"],
    "Virtual_RELC": ["RELB", "RELBIN"]
}

groundMarkers = {
    "Virtual_Ground_RAJC": "Virtual_RAJC",
    "Virtual_Ground_LAJC": "Virtual_LAJC",
    "Virtual_Ground_RFOOT": "RFOOT",
    "Virtual_Ground_LFOOT": "LFOOT",
    "Virtual_Ground_RTOE": "RTOE",
    "Virtual_Ground_LTOE": "LTOE",
    "Virtual_Ground_RMT1": "RMT1",
    "Virtual_Ground_LMT1": "LMT1",
    "Virtual_Ground_RMT5": "RMT5",
    "Virtual_Ground_LMT5": "LMT5",
    "Virtual_Ground_RHEE": "RHEE",
    "Virtual_Ground_LHEE": "LHEE",
}

dataReader = DataReader()
IKSET = dataReader.read_ik_set(IKTaskSet)
SCALEIKSET = dataReader.read_ik_set(scalingIKSet)
SCALESET = dataReader.read_scale_set(scaleSet)


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
        # model
        self.create_model()

        # pretrained pose estimation model
        if self.cfg.STARTPOSMODELPATH is not None:
            print(f'Load Pose Estimation Model : {self.cfg.STARTPOSMODELPATH}')
            dicts = self.spatial_model.state_dict()
            weights = torch.load(self.cfg.STARTPOSMODELPATH)
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
            self.temporal_model.load_state_dict(torch.load(self.cfg.STARTTEMPORALMODELPATH))
        else:
            assert False

        self.spatial_model.eval()
        self.temporal_model.eval()
        '''
        pyOpensimModel = DataReader.read_opensim_model(self.cfg.OPENSIM.MODELPATH)
        self.opensimModel = OpenSimModel(
            self.cfg.OPENSIM.MODELPATH, pyOpensimModel, SCALESET, SCALEIKSET, IKSET,
            unit=self.cfg.OPENSIM.UNIT,
            prescaling_lockedCoordinates=self.cfg.OPENSIM.prescaling_lockedCoordinates,
            prescaling_unlockedConstraints=self.cfg.OPENSIM.prescaling_unlockedConstraints,
            prescaling_defaultValues=self.cfg.OPENSIM.prescaling_defaultValues,
            postscaling_lockedCoordinates=self.cfg.OPENSIM.postscaling_lockedCoordinates,
            postscaling_unlockedConstraints=self.cfg.OPENSIM.postscaling_unlockedConstraints,
            changingParentMarkers=self.cfg.OPENSIM.changingParentMarkers
        )'''

        self.motionMarkerIndexTable = {}
        for idx, name in enumerate(self.cfg.PREDICTION.POS_NAME):
            # use joints without hands
            self.motionMarkerIndexTable[name] = idx

        # add virtual markers and ground markers
        self.staticMarkerIndexTable = self.motionMarkerIndexTable.copy()
        for marker in virtualMarkers:
            self.staticMarkerIndexTable[marker] = len(self.staticMarkerIndexTable)

    def create_model(self):
        self.spatial_model = PoseEstimationModel(
            self.cfg
        ) \
            .float().to(self.COMP_DEVICE)
        self.spatial_model.eval()

        self.temporal_model = TemporalPoseEstimationModel(
            self.cfg
        ) \
            .float().to(self.COMP_DEVICE)
        self.temporal_model.eval()

    def spatial_model_forward(self, inf):

        with torch.no_grad():
            # get input
            image = inf["image"].float().to(self.COMP_DEVICE)

            # forward
            spatialPos = self.spatial_model(image)

            # move the root to original point
            spatialPos = spatialPos - spatialPos[:, 0:1, :]

        return spatialPos

    def temproal_model_forward(self, spatialPos):

        with torch.no_grad():
            inputs = spatialPos.view(spatialPos.shape[0], spatialPos.shape[1], self.cfg.MODEL.INFEATURES).to(
                self.COMP_DEVICE)
            if self.cfg.MODEL.TYPE == 1:
                pred = self.temporal_model(inputs)
                predPos = pred["predPos"]
            else:
                predPos = self.temporal_model(inputs)
        predPos = predPos.view(predPos.shape[0], 1, self.cfg.PREDICTION.NUMPRED, 3)
        predPos = predPos - predPos[:, :, 0:1, :]
        return predPos

    def evaluate_time_cost(self):

        batchSize = 1
        self.testSet = BMLImgDataSet(self.cfg, self.h5pyFolder, self.cameraParamter, 2, evaluation=True)
        self.testLoader = DataLoader(self.testSet, batch_size=batchSize, shuffle=False,
                                     drop_last=False,
                                     num_workers=self.cfg.WORKERS if self.cfg.WORKERS >= batchSize else batchSize)

        print(f'{len(self.testSet)} test data')
        print(f'{len(self.testSet.videoTable)} video')

        spatialPosData = torch.zeros(len(self.testSet), self.cfg.PREDICTION.NUMPRED, 3)
        temporalPosData = torch.zeros(len(self.testSet), self.cfg.PREDICTION.NUMPRED, 3)

        cameraTypes = []
        print(temporalPosData.shape)
        totalStartTime = time.time()

        startImgTime = time.time()
        # spatial prediction
        for inf in tqdm(self.testLoader):
            pred = self.spatial_model_forward(inf)
            globalDataIdx = list(inf["globalDataIdx"].numpy())
            spatialPosData[globalDataIdx, :, :] = pred.cpu().detach()
            for c in inf["cameraType"]:
                cameraTypes.append(c)
        endImgTime = time.time()

        startTemporalTime = time.time()
        # temporal prediction
        for videoIdx in self.testSet.videoTable:
            usedIndices = np.array(self.testSet.videoTable[videoIdx])
            data = spatialPosData[usedIndices, :, :]

            window = torch.empty(self.cfg.MODEL.RECEPTIVE_FIELD, self.cfg.PREDICTION.NUMPRED, 3)
            window[:self.cfg.MODEL.RECEPTIVE_FIELD // 2, :, :] = data[0:1, :, :]
            if data.shape[0] >= self.cfg.MODEL.RECEPTIVE_FIELD // 2:
                window[self.cfg.MODEL.RECEPTIVE_FIELD // 2:, :, :] = data[:(
                        self.cfg.MODEL.RECEPTIVE_FIELD - self.cfg.MODEL.RECEPTIVE_FIELD // 2), :, :]
                endIdx = (self.cfg.MODEL.RECEPTIVE_FIELD - self.cfg.MODEL.RECEPTIVE_FIELD // 2)
            else:
                window[self.cfg.MODEL.RECEPTIVE_FIELD // 2:self.cfg.MODEL.RECEPTIVE_FIELD // 2 + data.shape[0], :,
                :] = data
                window[self.cfg.MODEL.RECEPTIVE_FIELD // 2 + data.shape[0]:, :, :] = data[-1, :, :]
                endIdx = data.shape[0] - 1

            for currentIdx in trange(0, len(usedIndices), batchSize):

                inf = torch.empty(batchSize, self.cfg.MODEL.RECEPTIVE_FIELD, self.cfg.PREDICTION.NUMPRED, 3)

                if currentIdx + batchSize >= data.shape[0]:
                    currentIdx = slice(currentIdx, data.shape[0], 1)
                else:
                    currentIdx = slice(currentIdx, currentIdx + batchSize, 1)

                for i in range(batchSize):
                    inf[i, :, :, :] = window.clone()
                    window[:-1, :, :] = window[1:, :, :].clone()
                    endIdx += 1
                    if endIdx < len(usedIndices):
                        window[-1, :, :] = data[endIdx, :, :]
                predTemporalPos = self.temproal_model_forward(inf)
                temporalPosData[usedIndices[currentIdx], :, :] = predTemporalPos[:, 0, :, :].cpu().detach()
        endTemporalTime = time.time()

        startTimeOpenSim = time.time()

        # call the opensim IK
        '''
        threads = []
        for idx, videoIdx in enumerate(self.testSet.videoTable):
            usedIndices = self.testSet.videoTable[videoIdx]
            thread = threading.Thread(target=self.simulateIK, args=(
                self.testSet.usedSubjects[0], temporalPosData[usedIndices, :, :], cameraTypes[usedIndices[0]], idx))
            threads.append(thread)
            # thread.start()
        if len(threads) <= mp.cpu_count():
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            cores = mp.cpu_count()
            for idx in range(0, len(threads), cores):
                for j in range(idx, idx + cores):
                    if j >= len(threads):
                        break
                    threads[j].start()
                for j in range(idx, idx + cores):
                    if j >= len(threads):
                        break
                    threads[j].join()'''

        temporalPosData = temporalPosData.cpu().detach().clone().numpy()

        pool = mp.Pool(mp.cpu_count())
        results = pool.starmap(simulateIK, [
            (self.testSet.usedSubjects[0], np.array(temporalPosData[usedIndices, :, :]), cameraTypes[usedIndices[0]],
             idx, self.cfg.OPENSIM.GTFOLDER, self.motionMarkerIndexTable, IKSET) for
            idx, (_, usedIndices) in enumerate(self.testSet.videoTable.items())
        ])
        pool.close()
        '''
        temporalPosData = temporalPosData.cpu().detach().clone().numpy()
        for idx, videoIdx in enumerate(self.testSet.videoTable):
            usedIndices = self.testSet.videoTable[videoIdx]
            #self.simulateIK(self.testSet.usedSubjects[0], temporalPosData[usedIndices, :, :], cameraTypes[usedIndices[0]], idx)
            self.simulateIK(self.testSet.usedSubjects[0], temporalPosData[usedIndices, :, :],
                            "PG1", idx)'''


        endTimeOpenSim = time.time()
        totalEndingTime = time.time()

        print(f'Total Time : {totalEndingTime - totalStartTime} ,'
              f' Spatial Time: {endImgTime - startImgTime} ,'
              f' Temporal Time : {endTemporalTime - startTemporalTime}'
              f' openSim Time : {endTimeOpenSim - startTimeOpenSim}'
              f'Batch Size : {batchSize}')


def generate_ik_setup_file(ikSet, path: str, modelFile: str, markerFile: str, timeRange: str,
                           outputMotionFilePath: str, outputCoordinateFilePath=""):
    ikXMLFilePath = path
    ikMotionFilePath = outputMotionFilePath
    scaledModelFolder = "/".join(path.split("/")[:-1]) + "/"

    with open(path, 'w') as f:
        f.writelines("<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n")
        f.writelines("<OpenSimDocument Version=\"40000\">\n")
        f.writelines("\t<InverseKinematicsTool name=\"\">\n")

        f.writelines("\t\t<model_file> " + modelFile + " </model_file>\n")
        f.writelines("\t\t<constraint_weight> 20 </constraint_weight>\n")
        f.writelines("\t\t<accuracy> 1e-005 </accuracy>\n")

        f.writelines("\t\t<IKTaskSet name=\"\">\n")
        f.writelines("\t\t\t<objects name=\"\">\n")
        for markerWeight in ikSet.markerWeight:
            weight = markerWeight.weight
            if weight is None:
                weight = 0
            f.writelines("\t\t\t\t<IKMarkerTask name=\"" + markerWeight.name + "\">\n")
            f.writelines("\t\t\t\t\t<apply> true </apply>\n")
            f.writelines("\t\t\t\t\t<weight> " + str(weight) + " </weight>\n")
            f.writelines("\t\t\t\t</IKMarkerTask>\n")
        f.writelines("\t\t\t</objects>\n")
        f.writelines("\t\t\t<groups/>\n")
        f.writelines("\t\t</IKTaskSet>\n")

        f.writelines("\t\t<marker_file> " + markerFile + " </marker_file>\n")
        f.writelines("\t\t<coordinate_file> " + outputCoordinateFilePath + "  </coordinate_file>\n")
        f.writelines("\t\t<time_range> " + timeRange + " </time_range>\n")
        f.writelines("\t\t<output_motion_file> " + outputMotionFilePath + " </output_motion_file>\n")
        f.writelines("\t\t<report_errors>true</report_errors>\n")
        f.writelines("\t\t<report_marker_locations>true</report_marker_locations>\n")
        f.writelines("\t</InverseKinematicsTool>\n")
        f.writelines("</OpenSimDocument>\n")

    return


def simulateIK(
        subject, prediction, cameraType, idx, GTFOLDER, motionMarkerIndexTable, ikSet
):
    print(cameraType)
    relativeFolder = f'Subject_{subject}_F_MoSh/'

    Path(GTFOLDER + relativeFolder + "NNIKResults").mkdir(parents=True, exist_ok=True)
    Path(GTFOLDER + relativeFolder + "PredTrcData").mkdir(parents=True, exist_ok=True)
    Path(GTFOLDER + relativeFolder + "NNResults").mkdir(parents=True, exist_ok=True)

    # convert to opensim coordinate system
    prediction[:, :, 1] *= -1
    prediction[:, :, 0] *= -1

    filename = f'Subject_{subject}_F_{idx}'

    # ik setting
    outputIKXMLPath = GTFOLDER + relativeFolder + f'IK_setUp_{idx}.xml'
    markerFile = "PredTrcData/" + filename + ".trc"
    timeRange = '0 ' + str(prediction.shape[0] * 1 / 30)
    outputMotionFilePath = "NNIKResults/" + filename + ".mot"
    outputModelFilePath = relativeFolder + f'scale_{cameraType}.osim'

    # if motion trc file exists, skip
    outputPath = GTFOLDER + relativeFolder + markerFile
    if not os.path.exists(outputPath):
        TrcGenerator.write_motion_marker_trc_file(
            outputPath, prediction, motionMarkerIndexTable, 30, 30,
            prediction.shape[0], len(motionMarkerIndexTable), 6, 0,
            prediction.shape[0]
        )

    # Generate the ik .xml file

    generate_ik_setup_file(
        ikSet, outputIKXMLPath, GTFOLDER + outputModelFilePath, markerFile, timeRange,
        outputMotionFilePath
    )

    # ik
    ik = opensim.InverseKinematicsTool(outputIKXMLPath, True)
    success = ik.run()

    # innerTime = time.time()
    # save the simulation result
    # kStoPath = self.cfg.OPENSIM.GTFOLDER + relativeFolder + "_ik_model_marker_locations.sto"

    # marker position
    # simulatedMarkerPositions = self.opensimModel.read_ik_marker_location(ikStoPath)
    # endTime = time.time()
    # print(endTime-innerTime)
    return


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
