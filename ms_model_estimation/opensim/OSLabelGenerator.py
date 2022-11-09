import argparse
import collections
from pathlib import Path
import pickle as pkl
import opensim
import math
import os
from ms_model_estimation import Global_Translation_Coordinates, Right_Shoulder_Groups, Left_Shoulder_Groups
from ms_model_estimation.pyOpenSim.TrcGenerator import TrcGenerator
from ms_model_estimation.opensim.OpenSimModel import OpenSimModel
from ms_model_estimation.opensim.DataReader import DataReader
from ms_model_estimation.pyOpenSim.ScaleIKSet import IKSet, ScaleSet
from ms_model_estimation.smplh_util.SMPLHModel import SMPLHModel
import numpy as np
import scipy.io
from glob import glob
from ms_model_estimation import Postscaling_LockedCoordinates, Postscaling_UnlockedConstraints, ChangingParentMarkers
from ms_model_estimation.smplh_util.constants.scalingIKInf import IKTaskSet, scalingIKSet, scaleSet


class AmassOpenSimGTGenerator:

    def __init__(
            self, amassFolder, modelPath: str, scaleSet: ScaleSet, scaleIKSet: IKSet, ikSet: IKSet, unit="m",
            prescaling_lockedCoordinates: list = None,
            prescaling_unlockedConstraints: list = None,
            prescaling_defaultValues=None,
            postscaling_lockedCoordinates=Postscaling_LockedCoordinates,
            postscaling_unlockedConstraints=Postscaling_UnlockedConstraints,
            changingParentMarkers=ChangingParentMarkers,
            reScale=False, reIK=False, scaleOnly=False
    ):
        '''
        a class to generate ground truth given amass data
        :param amassFolder: amass dataset folder
        :param modelPath: .osim model path
        :param scaleSet: scaleSet
        :param scaleIKSet: marker weights followed the format of IKSet
        :param ikSet: marker weights followed the format of IKSet
        :param unit: unit in marker data
        :param prescaling_lockedCoordinates: coordinates locked before body scaling
        :param prescaling_unlockedConstraints: constraints unlocked before body scaling
        :param prescaling_defaultValues: set joint angles to default value. format: {coordinate_name: value}
        :param postscaling_lockedCoordinates: coordinates locked after body scaling
        :param postscaling_unlockedConstraints: coordinates unlocked after body scaling
        :param changingParentMarkers: markers changed the anchored body. format : {marker_name: new_anchored_body}
        :param reScale: conduct body scaling again and overwrite previous results
        :param reIK: conduct inverse kinematics again and overwrite previous results
        :param scaleOnly: only conduct body scaling
        '''

        # convert .osim model to pyOpenSimModel
        pyOpensimModel = DataReader.read_opensim_model(modelPath)

        self.opensimModel = OpenSimModel(
            modelPath, pyOpensimModel, scaleSet, scaleIKSet, ikSet,
            unit=unit,
            prescaling_lockedCoordinates=prescaling_lockedCoordinates,
            prescaling_unlockedConstraints=prescaling_unlockedConstraints,
            prescaling_defaultValues=prescaling_defaultValues,
            postscaling_lockedCoordinates=postscaling_lockedCoordinates,
            postscaling_unlockedConstraints=postscaling_unlockedConstraints,
            changingParentMarkers=changingParentMarkers
        )

        # rescale the body model
        self.reScale = reScale

        # re-generate motion .trc fule
        self.reIK = reIK
        self.scaleOnly = scaleOnly

        self.amassFolder = amassFolder
        self.outputFolder = self.opensimModel.modelFolder
        if self.amassFolder[-1] != "/":
            self.amassFolder += "/"
        if self.outputFolder[-1] != "/":
            self.outputFolder += "/"

        # the filename of scaled .osim model
        self.scaleModelFileName = "scale"
        # the filename of template opensim model
        self.templateModelFileName = "full_body.osim"

        self.outputScaleXMLPath = self.outputFolder + "temp_scale_setUp.xml"
        while os.path.exists(self.outputScaleXMLPath):
            self.outputScaleXMLPath = self.outputFolder + "temp_scale_setUp_" + str(np.random.randint(50000)) + ".xml"
        self.scaling_timeRange = "0 0.5"

    def generate(self, bdataPath, gender="N"):

        videoName = bdataPath.split("/")[-1].split(".")[0]
        relativeFolder = "/".join(bdataPath.split("/")[:-1]) + "/"
        relativeFolder = relativeFolder.replace(self.amassFolder, "")
        bdata = np.load(bdataPath)
        numFrames = int(bdata["poses"].shape[0])

        # construct folders
        # folder to save .mot motion files of ik results
        Path(self.outputFolder + relativeFolder + "IKResults").mkdir(parents=True, exist_ok=True)
        # folder to save raw marker data
        Path(self.outputFolder + relativeFolder + "MotionTrcData").mkdir(parents=True, exist_ok=True)
        # folder to save .pkl ik results
        Path(self.outputFolder + relativeFolder + "Results").mkdir(parents=True, exist_ok=True)

        # body scaling setting
        height = SMPLHModel.get_height(bdata, gender=gender)
        markerFile = relativeFolder + "static.trc"
        outputModelFilePath = relativeFolder + "scale.osim"
        outputMotionFilePath = relativeFolder + "static_scale.mot"
        outputScaleFilePath = "Unassigned"
        outputMarkerFilePath = relativeFolder + "scale_simulate.trc"
        mass = SMPLHModel.get_assumed_body_weight(bdata, gender=gender)

        # if scale.osim exists, rescale?
        if not os.path.exists(self.outputFolder + outputModelFilePath) or self.reScale:
            TrcGenerator.generate_static_marker_trc_file_from_amass(
                self.outputFolder + markerFile, bdata, eulerAngle=[0, math.pi / 2, 0], gender=gender
            )

            # Generate the scaling .xml file
            self.opensimModel.generate_scale_setup_file(
                self.outputScaleXMLPath, self.scaleModelFileName, mass, height, 99,
                self.templateModelFileName, markerFile, self.scaling_timeRange,
                outputModelFilePath, outputScaleFilePath,
                outputMotionFilePath, outputMarkerFilePath, measurements=True
            )

            # body scaling
            self.opensimModel.scaling()

        # only conduct body scaling
        if self.scaleOnly:
            return


        # ik setting
        outputIKXMLPath = self.outputFolder + relativeFolder + "IK_setUp.xml"
        markerFile = "MotionTrcData/" + videoName + ".trc"
        timeRange = SMPLHModel.get_time_range(bdata, 120)
        outputMotionFilePath = "IKResults/" + videoName + ".mot"

        # if motion trc file exists, re-IK?
        if not os.path.exists(self.outputFolder + relativeFolder + markerFile) or self.reIK:
            TrcGenerator.generate_motion_marker_trc_file_from_amass(
                self.outputFolder + relativeFolder + markerFile, bdata,
                gender=gender
            )
        # Generate the ik .xml file
        self.opensimModel.generate_ik_setup_file(
            outputIKXMLPath, self.outputFolder + outputModelFilePath, markerFile, timeRange,
            outputMotionFilePath
        )
        # ik
        self.opensimModel.inverseKinematics()

        # record all of simulation results in a .pkl file 
        Label = {"gender": gender}

        # evaluate
        ikStoPath = self.outputFolder + relativeFolder + "_ik_model_marker_locations.sto"
        Label.update(self.evaluate_ik_result(ikStoPath, self.outputFolder + relativeFolder + markerFile, numFrames))

        # labels for coordinate angle and
        coordinateAngleDict = self.opensimModel.read_ik_mot_file(
            self.outputFolder + relativeFolder + outputMotionFilePath)
        Label.update(self.create_jointAngle_mask(numFrames, Label["error"], coordinateAngleDict))

        # bone scale
        pyBodySet = DataReader.read_opensim_bodySet(
            opensim.Model(self.outputFolder + relativeFolder + self.scaleModelFileName + ".osim"))
        Label.update(self.get_bone_scale(pyBodySet))

        pkl.dump(Label, open(self.outputFolder + relativeFolder + "Results/" + videoName + "_gt.pkl", "wb"))

    def evaluate_ik_result(self, ikStoPath, GTPath, numFrames):
        '''
        :param ikStoPath: the .sto path of simulated marker positons
        :param GTPath: the .trc path of experiment marker data (GT marker data)
        :param numFrames: the number of frames
        :return: a dictionary to record marker error, ground truth and simulation results of IK
        '''
        simulatedMarkerPositions = self.opensimModel.read_ik_marker_location(ikStoPath)
        GTMarkerPositions = self.opensimModel.read_trc_file(GTPath)

        predPos = np.zeros((numFrames, len(TrcGenerator.StaticMarkerIndexTable), 3), dtype=np.float32)
        gtPos = np.zeros((numFrames, len(TrcGenerator.StaticMarkerIndexTable), 3), dtype=np.float32)

        # store the marker location
        for marker, idx in TrcGenerator.StaticMarkerIndexTable.items():
            predPos[:, idx, :] = np.array(simulatedMarkerPositions[marker])
            if "ground" in marker.lower():
                gtPos[:, idx, :] = np.array(simulatedMarkerPositions[marker])
            else:
                gtPos[:, idx, :] = np.array(GTMarkerPositions[marker])

        errors = np.sum((predPos - gtPos) ** 2, axis=-1) ** 0.5

        # if the marker error is bigger than 2cm, the simulation of that joint is thought as wrong simulation.
        errorMessages = collections.defaultdict(list)
        for bonyLandmark in self.opensimModel.bonyLandMarkers:
            idx = TrcGenerator.StaticMarkerIndexTable[bonyLandmark]
            error = errors[:, idx]
            errorFrames = np.where(error > 0.02)[0]
            if errorFrames.any():
                errorMessages[bonyLandmark].append(errorFrames.tolist())
                errorMessages[bonyLandmark].append(error[errorFrames])

        return {"error": errorMessages, "simulation": predPos, "label": gtPos}

    def get_bone_scale(self, pyBodySet):
        '''
        A dictionary to record the body scales
        '''
        boneScale = np.zeros((len(self.opensimModel.pyOpensimModel.bodySet.bodiesDict), 3), dtype=np.float32)
        for idx, bone in enumerate(self.opensimModel.pyOpensimModel.bodySet.bodiesDict):
            boneScale[idx, :] = pyBodySet.bodiesDict[bone].scale
        names = [k for k in self.opensimModel.pyOpensimModel.bodySet.bodiesDict]
        Label = {"boneScale": boneScale, "boneName": names}

        return Label

    def create_jointAngle_mask(self, numFrames, errorMessages, coordinateAngleDict):
        '''
        A dictionary to record joint angles, joint angle masks, joint angles name and translation.
        If a body has marker error larger than 2cm, we say the joint angles belonged to the body are wrong.
        Shoulders in muscuskleletal model are modelled by multiple bodies. Therefore, if LSHO/RSHO is wrong, we say all
        of bodies in the shoulder are wrong.
        Note: coordinate angle==joint angle
        '''

        coordinateNames = []
        for name in coordinateAngleDict:
            if name not in Global_Translation_Coordinates and "time" != name:
                coordinateNames.append(name)

        # simulated coordinate angle
        simulatedCoordinateAngle = np.zeros((numFrames, len(coordinateNames)),
                                            dtype=np.float32)

        # global translation
        globalTranslation = np.zeros((numFrames, len(Global_Translation_Coordinates)), dtype=np.float32)

        # 0 means the simulation of the coordinate is wrong
        coordinateMask = np.ones((numFrames, len(coordinateAngleDict) - len(Global_Translation_Coordinates)),
                                 dtype=np.int8)

        wrongParentFrames = collections.defaultdict(set)
        for wrongBonyLandmark in errorMessages:
            wrongFrames = errorMessages[wrongBonyLandmark][0]
            wrongParentFrame = \
                self.opensimModel.pyOpensimModel.markerSet.markerDict[wrongBonyLandmark].parentFrame.split("/")[-1]

            # If RSHO/LSHO is wrong, all of the masks for predicted coordinates on shoulders should set to 0.
            if wrongParentFrame in Right_Shoulder_Groups:
                for wrongParentFrame in Right_Shoulder_Groups:
                    wrongParentFrames[wrongParentFrame].update(set(wrongFrames))
            elif wrongParentFrame in Left_Shoulder_Groups:
                for wrongParentFrame in Left_Shoulder_Groups:
                    wrongParentFrames[wrongParentFrame].update(set(wrongFrames))
            else:
                wrongParentFrames[wrongParentFrame].update(set(wrongFrames))

        for idx, predictedCoordinate in enumerate(coordinateNames):
            simulatedCoordinateAngle[:, idx] = coordinateAngleDict[predictedCoordinate]

            # If the joint belonged to the coordinate has marker error bigger than 2cm, the mask is set to 0.
            parentFrame = \
                self.opensimModel.pyOpensimModel.jointSet.coordinatesDict[predictedCoordinate][
                    0].frames.childFrame.split("/")[-1]
            if parentFrame in wrongParentFrames:
                coordinateMask[list(wrongParentFrames[parentFrame]), idx] = 0

        for idx, coordinateName in enumerate(Global_Translation_Coordinates):
            globalTranslation[:, idx] = coordinateAngleDict[coordinateName]

        Label = {"coordinateAngle": simulatedCoordinateAngle, "coordinateMask": coordinateMask,
                 "coordinateName": coordinateNames, "globalTranslation": globalTranslation}

        return Label

    def traverse_npz_files(self, subDataSet):

        npzDataPathList = []
        files = glob(self.amassFolder + subDataSet + "/*/*.npz")
        # Search all .npz files
        for file in files:
            # two types of file names cause errors.
            if "shape" not in file and "HDM_bk_01-01_01_120_poses" not in file:
                npzDataPathList.append(file.replace('\\', "/"))

        return npzDataPathList


class BMLAmassOpenSimGTGenerator(AmassOpenSimGTGenerator):

    def __init__(
            self, v3dFolder, amassFolder: str, modelPath: str, scaleSet: ScaleSet, scaleIKSet: IKSet, ikSet: IKSet,
            unit="m",
            prescaling_lockedCoordinates=None, prescaling_unlockedConstraints=None, prescaling_defaultValues=None,
            postscaling_lockedCoordinates=Postscaling_LockedCoordinates,
            postscaling_unlockedConstraints=Postscaling_UnlockedConstraints,
            changingParentMarkers=ChangingParentMarkers,
            reScale=False, reIK=False, scaleOnly=False
    ):
        super(BMLAmassOpenSimGTGenerator, self).__init__(
            amassFolder, modelPath, scaleSet, scaleIKSet, ikSet, unit=unit,
            prescaling_lockedCoordinates=prescaling_lockedCoordinates,
            prescaling_unlockedConstraints=prescaling_unlockedConstraints,
            prescaling_defaultValues=prescaling_defaultValues,
            postscaling_lockedCoordinates=postscaling_lockedCoordinates,
            postscaling_unlockedConstraints=postscaling_unlockedConstraints,
            changingParentMarkers=changingParentMarkers,
            reScale=reScale, reIK=reIK, scaleOnly=scaleOnly
        )

        self.v3dFolder = v3dFolder if v3dFolder[-1] == "/" else v3dFolder + "/"

    def generate(self, bdataPath, neutral=False):

        subjectID = bdataPath.split("/")[-1].split(".")[0].split("_")[1]
        matFile = f'{self.v3dFolder}F_v3d_Subject_{subjectID}.mat'
        matData = scipy.io.loadmat(matFile)["Subject_" + str(subjectID) + "_F"]

        if neutral:
            gender = "N"
        else:
            gender = matData["subject"][0, 0]["sex"][0, 0][0].lower()
            if gender == "female":
                gender = "F"
            elif gender == "male":
                gender = "M"
            else:
                raise Exception("Gender is not defined!")

        super().generate(bdataPath, gender)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('opensimModelPath', action='store', default="", type=str, help=".osim model path")
    parser.add_argument('amassFolder', action='store', default="", type=str, help="Amass folder")
    parser.add_argument('v3dFolder', action='store', default="", type=str, help="v3d folder")

    args = parser.parse_args()

    dataReader = DataReader()
    ikset = dataReader.read_ik_set(IKTaskSet)
    scalingIKSet = dataReader.read_ik_set(scalingIKSet)
    scaleset = dataReader.read_scale_set(scaleSet)

    gtGenerator = BMLAmassOpenSimGTGenerator(
        args.v3dFolder, args.amassFolder, args.opensimModelPath, scaleset, scalingIKSet, ikset,
    )

    npzPathList = gtGenerator.traverse_npz_files("BMLmovi")

    for path in npzPathList[:1]:
        gtGenerator.generate(path)
