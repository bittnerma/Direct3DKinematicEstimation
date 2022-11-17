import argparse
import pickle
import random
import h5py
import numpy as np
import torch
from ms_model_estimation.pyOpenSim.TrcGenerator import TrcGenerator
from opensim import OpenSimModel
from opensim.DataReader import DataReader
import pandas as pd
from tqdm import tqdm
from ms_model_estimation.models.dataset.BMLImgTposeDataSet import BMLImgTposeDataSet
from ms_model_estimation.models.networks.PoseEstimationModel import PoseEstimationModel
from ms_model_estimation.models.config.config_os_baseline import get_cfg_defaults
from torch.utils.data import DataLoader
import os
from pathlib import Path
from ms_model_estimation.models.utils.BMLUtils import CAMERA_TABLE
from ms_model_estimation.models.camera.cameralib import Camera
from ms_model_estimation.smplh.scalingIKInf import IKTaskSet, scalingIKSet, scaleSet

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


class OpenSimBaseline():

    def __init__(
            self, args, cfg
    ):
        self.cfg = cfg

        if args.cpu:
            self.COMP_DEVICE = torch.device("cpu")
        else:
            self.COMP_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # create camera parameters
        cameraParamter = {}
        groundInCamera = {}
        for cameraType in CAMERA_TABLE:
            cameraInf = CAMERA_TABLE[cameraType]
            R = cameraInf["extrinsic"][:3, :3]  # .T
            t = np.matmul(cameraInf["extrinsic"][:3, -1:].T, R) * -1
            distortion_coeffs = np.array(
                [cameraInf["radialDisortionCoeff"][0], cameraInf["radialDisortionCoeff"][1], 0, 0, 0], np.float32)
            intrinsic_matrix = cameraInf["intrinsic"].copy()
            camera = Camera(t, R, intrinsic_matrix, distortion_coeffs)
            cameraParamter[cameraType] = camera
            groundInCamera[cameraType] = camera.world_to_camera(np.zeros((3,)))
        self.cameraParamter = cameraParamter
        self.groundInCamera = groundInCamera

        pyOpensimModel = DataReader.read_opensim_model(cfg.OPENSIM.MODELPATH)
        self.opensimModel = OpenSimModel(
            cfg.OPENSIM.MODELPATH, pyOpensimModel, SCALESET, SCALEIKSET, IKSET,
            unit=cfg.OPENSIM.UNIT,
            prescaling_lockedCoordinates=cfg.OPENSIM.prescaling_lockedCoordinates,
            prescaling_unlockedConstraints=cfg.OPENSIM.prescaling_unlockedConstraints,
            prescaling_defaultValues=cfg.OPENSIM.prescaling_defaultValues,
            postscaling_lockedCoordinates=cfg.OPENSIM.postscaling_lockedCoordinates,
            postscaling_unlockedConstraints=cfg.OPENSIM.postscaling_unlockedConstraints,
            changingParentMarkers=cfg.OPENSIM.changingParentMarkers
        )

        self.motionMarkerIndexTable = {}
        for idx, name in enumerate(self.cfg.PREDICTION.POS_NAME):
            # use joints without hands
            self.motionMarkerIndexTable[name] = idx

        # add virtual markers and ground markers
        self.staticMarkerIndexTable = self.motionMarkerIndexTable.copy()
        for marker in virtualMarkers:
            self.staticMarkerIndexTable[marker] = len(self.staticMarkerIndexTable)

        '''   
        for marker in groundMarkers:
            self.staticMarkerIndexTable[marker] = len(self.staticMarkerIndexTable)'''

        self.create_df()

    def load_prediction(self, name=None):

        self.predictionTPose = np.load(self.cfg.BML_FOLDER + "predictionTPose.npy", allow_pickle=True)
        self.predictions = np.load(self.cfg.BML_FOLDER + "prediction_test.npy", allow_pickle=True).item()
        # self.osPredictions = np.load(self.cfg.BML_FOLDER + "os_test.npy", allow_pickle=True).item()
        if name is None:
            self.predictions_temporal = np.load(self.cfg.BML_FOLDER + "prediction_temporal_test.npy",
                                                allow_pickle=True).item()
        else:
            self.predictions_temporal = np.load(self.cfg.BML_FOLDER + name,
                                                allow_pickle=True).item()

    def create_df(self):

        self.usedSubjects = self.cfg.TEST_SUBJECTS
        usedIndices = []

        for subject in self.usedSubjects:
            if os.path.exists(self.cfg.BML_FOLDER + f'subject_{subject}.hdf5'):
                with h5py.File(self.cfg.BML_FOLDER + f'subject_{subject}.hdf5', 'r') as f:
                    cameraTypes = f["cameraType"][:]
                for localIdx, cameraType in enumerate(cameraTypes):
                    usedIndices.append([subject, cameraType.decode(), localIdx])

        subjectsList = []
        videoIdxList = []
        frameList = []
        cameraTypeList = []

        for subjectID, cameraType, localIdx in usedIndices:
            with h5py.File(self.cfg.BML_FOLDER + f'subject_{subjectID}.hdf5', 'r') as f:
                videoID = f['videoID'][localIdx]
                frame = f['frame'][localIdx]

            subjectsList.append(subjectID)
            videoIdxList.append(videoID)
            frameList.append(frame)
            cameraTypeList.append(cameraType)

        df = {
            "frame": frameList,
            "subjectID": subjectsList,
            "videoID": videoIdxList,
            "cameraType": cameraTypeList,
        }
        df = pd.DataFrame(data=df)

        self.df = df

    def model_forward(self, inf):

        # get input
        image = inf["image"].to(self.COMP_DEVICE)
        if "pose3d" in inf:
            pos3d = inf["pose3d"].to(self.COMP_DEVICE)
            rootOffset = pos3d[:, 0:1, :].clone()

        # forward
        predPos = self.model(image)

        # move the root to original point
        predPos = predPos - predPos[:, 0:1, :]

        if "pose3d" in inf:
            # aligned with the gt root
            predPos = predPos + rootOffset

        return predPos

    def create_model(self):
        # reload the model
        print("Load ", self.cfg.bestNNModelPath)

        self.model = PoseEstimationModel(
            self.cfg
        ) \
            .float().to(self.COMP_DEVICE)
        print(f'Image size : {self.cfg.MODEL.IMGSIZE}')

        dicts = self.model.state_dict()
        weights = torch.load(self.cfg.bestNNModelPath)
        for k, w in weights.items():
            if k in dicts:
                dicts[k] = w
            else:
                print(f'{k} is not in model')
        self.model.load_state_dict(dicts)
        self.model.eval()

    def predict_T_pose(self):

        self.create_model()

        predictionTPose = np.zeros((90, 2, self.cfg.PREDICTION.NUMPRED, 3), dtype=np.float32)
        self.model.eval()
        dataset = BMLImgTposeDataSet(self.cfg, self.cfg.BML_FOLDER, self.cameraParamter)
        dataLoader = DataLoader(dataset, batch_size=self.cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                drop_last=False, num_workers=self.cfg.WORKERS)
        print(f'The number of T poses : {len(dataset)}')
        with torch.no_grad():
            if self.cfg.PROGRESSBAR:
                iterator = enumerate(tqdm(dataLoader))
            else:
                iterator = enumerate(dataLoader, 0)

            for _, inf in iterator:
                predPos = self.model_forward(inf).detach().numpy()
                subjectIndices = inf["subjectID"].cpu().detach().numpy()
                cameraIndices = inf["cameraIdx"].cpu().detach().numpy()
                for i in range(subjectIndices.shape[0]):
                    subjectIdx = subjectIndices[i]
                    cameraIdx = cameraIndices[i]
                    predictionTPose[subjectIdx - 1, cameraIdx, :, :] = predPos[i, :, :]

        np.save(self.cfg.BML_FOLDER + "predictionTPose.npy", predictionTPose)
        del self.model

    def simulateBodyScaling(self, subject):

        subjectDF = self.df.loc[self.df["subjectID"] == subject, :]
        if len(subjectDF) == 0:
            return

        # subjects do not have T poses
        if subject == 1 or subject == 12:
            return

        relativeFolder = f'Subject_{subject}_F_MoSh/'
        Path(self.cfg.OPENSIM.GTFOLDER + relativeFolder).mkdir(parents=True, exist_ok=True)

        for idxCamera, cameraType in enumerate(["PG1", "PG2"]):

            # body scaling setting
            height = 170
            markerFile = relativeFolder + "static.trc"
            scaleModelFileName = "scale"
            outputModelFilePath = relativeFolder + f'scale_{cameraType}.osim'
            outputMotionFilePath = relativeFolder + "static_scale.mot"
            outputScaleFilePath = "Unassigned"
            outputMarkerFilePath = relativeFolder + "scale.trc"
            mass = 70
            scaling_timeRange = "0 0.5"

            scaledModelPath = self.cfg.OPENSIM.GTFOLDER + outputModelFilePath
            outputFolder = self.cfg.OPENSIM.GTFOLDER
            outputScaleXMLPath = "/".join(self.cfg.OPENSIM.MODELPATH.split("/")[:-1]) + "/temp_scale_setUp.xml"
            modelFile = "full_body.osim"

            if not os.path.exists(scaledModelPath) or self.cfg.OVERWRITE_BS:

                data = self.predictionTPose[subject - 1, idxCamera, :, :].copy()
                if np.sum(data) == 0:
                    continue

                # convert to opensim coordinate system
                data[:, 1] *= -1
                data[:, 0] *= -1
                # groundValue = np.min(data[:, 1]) - 0.005

                points = np.empty((len(self.staticMarkerIndexTable), 3))
                points[:data.shape[0], :] = data.copy()

                # add virtual marker
                for vMarker, markers in virtualMarkers.items():
                    markerIdx = self.staticMarkerIndexTable[vMarker]
                    tmp = points[self.staticMarkerIndexTable[markers[0]], :].copy()
                    for m in markers[1:]:
                        tmp += points[self.staticMarkerIndexTable[m], :].copy()
                    tmp /= len(markers)
                    points[markerIdx, :] = tmp
                '''
                # add ground marker
                for vMarker, marker in groundMarkers.items():
                    markerIdx = self.staticMarkerIndexTable[vMarker]
                    tmp = points[self.staticMarkerIndexTable[marker], :].copy()
                    tmp[1] = groundValue
                    points[markerIdx, :] = tmp'''

                TrcGenerator.write_static_marker_trc_file(
                    outputFolder + markerFile, points, self.staticMarkerIndexTable, 120, 30,
                    points.shape[0], len(self.staticMarkerIndexTable), 120, 0, points.shape[0],
                )

                # Generate the scaling .xml file
                self.opensimModel.generate_scale_setup_file(
                    outputScaleXMLPath, scaleModelFileName, mass, height, 99,
                    modelFile, "BMLmovi/" + markerFile, scaling_timeRange,
                               "BMLmovi/" + outputModelFilePath, outputScaleFilePath,
                               "BMLmovi/" + outputMotionFilePath, outputMarkerFilePath, measurements=True
                )

                # body scaling
                self.opensimModel.scaling(correct_parent_frame_after_scaling=False)

    def simulateIK(
            self, subject, gtScaleModel=False, temporal=False, postfix="spatial"
    ):

        subjectDF = self.df.loc[self.df["subjectID"] == subject, :]
        if len(subjectDF) == 0:
            return

        # subjects do not have T poses
        if not gtScaleModel and (subject == 1 or subject == 12):
            return

        # TODO:
        relativeFolder = f'Subject_{subject}_F_MoSh/'

        # kill folders
        '''
        try:
            shutil.rmtree(self.cfg.OPENSIM.GTFOLDER + relativeFolder + "NNIKResults")
            shutil.rmtree(self.cfg.OPENSIM.GTFOLDER + relativeFolder + "PredTrcData")
            shutil.rmtree(self.cfg.OPENSIM.GTFOLDER + relativeFolder + "NNResults")
        except:
            pass'''

        Path(self.cfg.OPENSIM.GTFOLDER + relativeFolder + "NNIKResults").mkdir(parents=True, exist_ok=True)
        Path(self.cfg.OPENSIM.GTFOLDER + relativeFolder + "PredTrcData").mkdir(parents=True, exist_ok=True)
        Path(self.cfg.OPENSIM.GTFOLDER + relativeFolder + "NNResults").mkdir(parents=True, exist_ok=True)

        videoIDList = np.array(subjectDF["videoID"].tolist())
        cameraList = np.array((subjectDF["cameraType"].tolist()))

        for videoID in tqdm(list(set(subjectDF["videoID"].tolist()))):
            for cameraType in ["PG1", "PG2"]:
                videoDF = subjectDF.loc[(videoIDList == videoID) & (cameraList == cameraType), :]
                indices = videoDF.index.tolist()
                if len(videoDF) == 0:
                    continue

                startIdx = videoDF.index[0]
                endIdx = videoDF.index[-1]
                if temporal:
                    prediction = self.predictions_temporal["prediction"][startIdx:endIdx + 1, :, :].copy()
                else:
                    prediction = self.predictions["prediction"][startIdx:endIdx + 1, :, :].copy()
                label = self.predictions["label"][startIdx:endIdx + 1, :, :].copy()
                filename = f'Subject_{subject}_F_{videoID}_pred_{cameraType}'
                if temporal:
                    filename += "_temporal"
                elif gtScaleModel:
                    filename += "_USEGT"

                if postfix is not None:
                    filename += "_"
                    filename += postfix

                # convert to opensim coordinate system
                prediction[:, :, 1] *= -1
                prediction[:, :, 0] *= -1
                label[:, :, 1] *= -1
                label[:, :, 0] *= -1

                # ik setting
                if postfix is not None:
                    outputIKXMLPath = self.cfg.OPENSIM.GTFOLDER + relativeFolder + f'IK_setUp_{postfix}.xml'
                else:
                    outputIKXMLPath = self.cfg.OPENSIM.GTFOLDER + relativeFolder + f'IK_setUp.xml'

                markerFile = "PredTrcData/" + filename + ".trc"
                timeRange = '0 ' + str(prediction.shape[0] * 1 / 30)
                outputMotionFilePath = "NNIKResults/" + filename + ".mot"
                if gtScaleModel:
                    outputModelFilePath = relativeFolder + "scale.osim"
                else:
                    outputModelFilePath = relativeFolder + f'scale_{cameraType}.osim'

                # if motion trc file exists, skip
                outputPath = self.cfg.OPENSIM.GTFOLDER + relativeFolder + markerFile
                if not os.path.exists(outputPath) or self.cfg.OVERWRITE_IK:
                    TrcGenerator.write_motion_marker_trc_file(
                        outputPath, prediction, self.motionMarkerIndexTable, 30, 30,
                        prediction.shape[0], len(self.motionMarkerIndexTable), 6, 0,
                        prediction.shape[0]
                    )

                # Generate the ik .xml file
                self.opensimModel.generate_ik_setup_file(
                    outputIKXMLPath, self.cfg.OPENSIM.GTFOLDER + outputModelFilePath, markerFile, timeRange,
                    outputMotionFilePath
                )

                # ik
                self.opensimModel.inverseKinematics()

                # save the simulation result
                ikStoPath = self.cfg.OPENSIM.GTFOLDER + relativeFolder + "_ik_model_marker_locations.sto"

                # marker position
                simulatedMarkerPositions = self.opensimModel.read_ik_marker_location(ikStoPath)
                simulation = np.zeros(prediction.shape)
                for marker, idx in self.motionMarkerIndexTable.items():
                    simulation[:, idx, :] = np.array(simulatedMarkerPositions[marker])

                # joint angle
                ikMotPath = self.cfg.OPENSIM.GTFOLDER + relativeFolder + f'NNIKResults/{filename}.mot'
                predictionAngles = self.opensimModel.read_ik_mot_file(ikMotPath)
                numFrames = len(predictionAngles["time"])
                predictionAngle = np.zeros((numFrames, len(self.cfg.PREDICTION.COORDINATES)), dtype=np.float32)
                for idx, name in enumerate(self.cfg.PREDICTION.COORDINATES):
                    predictionAngle[:, idx] = predictionAngles[name]

                outputPath = self.cfg.OPENSIM.GTFOLDER + relativeFolder + "NNResults/" + filename + ".pkl"
                data = {
                    "simulation": simulation,
                    "label": label,
                    "prediction": prediction,
                    "simulationRot": predictionAngle,
                    "index": indices,
                    # "predRot": self.osPredictions["predRot"][startIdx:endIdx + 1, :],
                    # "labelRot": self.osPredictions["labelRot"][startIdx:endIdx + 1, :],
                    # "labelBoneScale": self.osPredictions["labelBoneScale"][startIdx, :, :],
                }
                pickle.dump(data, open(outputPath, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cpu', action='store_true', default=False, help="only use cpu?")
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
