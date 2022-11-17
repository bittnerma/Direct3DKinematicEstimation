import os
from glob import glob
import h5py
import torch
import pickle
import pandas as pd
from ms_model_estimation.opensim.DataReader import DataReader
from ms_model_estimation.models.networks.model_layer.OpenSimNode import OpenSimNode
from ms_model_estimation.models.networks.model_layer.OpenSimTreeLayer import OpenSimTreeLayer
from ms_model_estimation.models.camera.cameralib import *
from tqdm import trange
from ms_model_estimation.models.utils.OSUtils import LeafJoints, PredictedBones, PredictedCoordinates, \
    PredictedOSJoints
from ms_model_estimation.models.utils.BMLUtils import PredictedMarkers, smplHJoint
from ms_model_estimation.pyOpenSim.TrcGenerator import TrcGenerator


def search_bml_data_list(hdf5Folder, opensimGTFolder, fullBodyPath, postfix="_MoSh", duration=1):
    hdf5Folder = hdf5Folder + "/" if not hdf5Folder.endswith("/") else hdf5Folder
    opensimGTFolder = opensimGTFolder + "/" if not opensimGTFolder.endswith("/") else opensimGTFolder
    pyBaseModel = DataReader.read_opensim_model(fullBodyPath)

    videos = glob(hdf5Folder + "*_*_img.hdf5")
    videos = [f.replace("\\", "/") for f in videos]

    videoIdxTable = np.load(hdf5Folder + "videoIdx.npy", allow_pickle=True).item()
    bboxInf = np.load(hdf5Folder + "all_bbox.npy", allow_pickle=True).item()

    labelList = []
    subjectIDList = []
    frameIDList = []
    videoIDList = []
    cameraTypeList = []
    imgNameList = []
    globalFrameList = []
    startFrameList = []
    endFrameList = []
    globalStartFrameList = []
    hflipUsageList = []
    seedList = []

    for video in videos:
        subjectID = int(video.split("/")[-1].split("_")[0])
        cameraType = video.split("/")[-1].split("_")[1]

        hflipUsage = np.random.uniform(0, 1) >= 0.5
        seed = np.random.randint(0, 2 ** 10)

        for videoIdx in range(1, len(videoIdxTable[subjectID][0]) + 1):

            globalStartFrame = videoIdxTable[subjectID][0][videoIdx - 1] - 1
            globalEndFrame = videoIdxTable[subjectID][1][videoIdx - 1] - 1
            folder = f'Subject_{subjectID}_F'
            labelFileName = "_".join([folder, str(videoIdx), "poses_gt.pkl"])
            labelPath = "/".join([opensimGTFolder + folder + postfix, "Results", labelFileName])
            if not os.path.exists(labelPath):
                print(f'{labelPath} is not created!')
                continue

            for globalFrame in range(globalStartFrame, globalEndFrame + 1):
                frame = globalFrame - globalStartFrame
                filename = f'BMLmovi/Subject_{subjectID}_F/F_{cameraType}_Subject_{subjectID}_L.{videoIdx}.{frame}.0.jpg'

                if bboxInf[subjectID][cameraType].shape[0] > globalFrame:
                    subjectIDList.append(subjectID)
                    imgNameList.append(filename)
                    labelList.append(labelPath)
                    frameIDList.append(frame)
                    videoIDList.append(videoIdx)
                    cameraTypeList.append(cameraType)
                    globalFrameList.append(globalFrame)
                    startFrameList.append(globalStartFrame)
                    endFrameList.append(globalEndFrame)
                    globalStartFrameList.append(videoIdxTable[subjectID][0][0] - 1)
                    hflipUsageList.append(1 if hflipUsage else 0)
                    seedList.append(seed)

    df = None
    if subjectIDList:
        df = {
            'subjectID': subjectIDList, 'videoID': videoIDList, 'frame': frameIDList,
            'globalFrame': globalFrameList, 'labelPath': labelList,
            'cameraType': cameraTypeList, "imgName": imgNameList,
            "startFrame": startFrameList, "globalStartFrame": globalStartFrameList, "globalEndFrame": endFrameList,
            "seed": seedList, "hflipUsage": hflipUsageList
        }
        df = pd.DataFrame(data=df)

    df = df.sort_values(
        by=["subjectID", "cameraType", "videoID", "frame"]).reset_index(drop=True)
    # df = checkContinous(df, duration=duration)
    df = df.sort_values(
        by=["subjectID", "cameraType", "videoID", "frame"]).reset_index(drop=True)

    return {
        "df": df,
        "pyBaseModel": pyBaseModel,
        "bboxInf": bboxInf,
    }


def store_as_h5py(subject, outputFolder, table):
    df = table["df"]
    subjectDF = df.loc[df["subjectID"] == subject, :]
    if len(subjectDF) == 0:
        return

    subjectDF = subjectDF.sort_values(
        by=["subjectID", "cameraType", "videoID", "frame"]).reset_index(drop=True)

    '''
    # create label datasets
    path = outputFolder + f'subject_{subject}.hdf5'
    create_h5py(path, subjectDF, table["bboxInf"])'''

    # create opensim dataset
    #path = outputFolder + f'subject_{subject}_opensim.hdf5'
    #create_opensim_label_dataset(path, subjectDF, table["pyBaseModel"])


def create_h5py(path, df, bboxInf):
    with h5py.File(path, 'w') as f:
        ImageListD = f.create_dataset('ImgList', (len(df),), dtype='S150', compression="gzip",
                                      compression_opts=9)
        imageNames = [name.encode() for name in df["imgName"].tolist()]
        ImageListD[:] = imageNames

        subjectIDD = f.create_dataset('subjectID', (len(df),), dtype='i8', compression="gzip",
                                      compression_opts=9)
        subjectIDD[:] = df["subjectID"].tolist()

        videoIDListD = f.create_dataset('videoID', (len(df),), dtype='i8', compression="gzip",
                                        compression_opts=9)
        videoIDListD[:] = df["videoID"].tolist()

        frameIDListD = f.create_dataset('frame', (len(df),), dtype='i8', compression="gzip",
                                        compression_opts=9)
        frameIDListD[:] = df["frame"].tolist()

        cameraTypeD = f.create_dataset('cameraType', (len(df),), dtype='S5', compression="gzip",
                                       compression_opts=9)
        cameras = [c.encode() for c in df["cameraType"].tolist()]
        cameraTypeD[:] = cameras

        globalStartFrameD = f.create_dataset('globalStartFrame', (len(df),),
                                             dtype='i8', compression="gzip",
                                             compression_opts=9)
        globalStartFrameD[:] = df["globalStartFrame"].tolist()

        globalEndFrameD = f.create_dataset('globalEndFrame', (len(df),),
                                           dtype='i8', compression="gzip",
                                           compression_opts=9)
        globalEndFrameD[:] = df["globalEndFrame"].tolist()

        startFrameD = f.create_dataset('startFrame', (len(df),),
                                       dtype='i8', compression="gzip",
                                       compression_opts=9)
        startFrameD[:] = df["startFrame"].tolist()

        globalFrameD = f.create_dataset('globalFrame', (len(df),),
                                        dtype='i8', compression="gzip",
                                        compression_opts=9)
        globalFrameD[:] = df["globalFrame"].tolist()

        hflipUsageD = f.create_dataset('hflipUsage', (len(df),),
                                       dtype='i8', compression="gzip",
                                       compression_opts=9)
        hflipUsageD[:] = df["hflipUsage"].tolist()

        seedD = f.create_dataset('seed', (len(df),),
                                 dtype='i8', compression="gzip",
                                 compression_opts=9)
        seedD[:] = df["seed"].tolist()

        create_label_dataset(f, df, bboxInf)


def create_label_dataset(f, df, bboxInf):
    imgSampleRate = 1
    MocapToVideoRatio = 4

    usedFrameD = f.create_dataset('usedFrame', (len(df),),
                                  dtype='i8', compression="gzip",
                                  compression_opts=9)
    '''
    usedFrame5D = f.create_dataset('usedFrame5', (len(df),),
                                   dtype='i8', compression="gzip",
                                   compression_opts=9)'''

    pos3dD = f.create_dataset(f'pos3d', (len(df), len(smplHJoint), 3),
                              dtype='f8', compression="gzip",
                              compression_opts=9, chunks=(1, len(smplHJoint), 3))

    pos3dD_marker = f.create_dataset(f'marker_pos3d', (len(df), len(PredictedMarkers), 3),
                                     dtype='f8', compression="gzip",
                                     compression_opts=9, chunks=(1, len(PredictedMarkers), 3))

    sim_pos3dD = f.create_dataset(f'sim_pos3d', (len(df), len(smplHJoint), 3),
                                  dtype='f8', compression="gzip",
                                  compression_opts=9, chunks=(1, len(smplHJoint), 3))

    sim_pos3dD_marker = f.create_dataset(f'sim_marker_pos3d', (len(df), len(PredictedMarkers), 3),
                                         dtype='f8', compression="gzip",
                                         compression_opts=9, chunks=(1, len(PredictedMarkers), 3))

    pos3dMaskD = f.create_dataset(f'pos3d_mask', (len(df), len(smplHJoint)),
                                  dtype='f8', compression="gzip",
                                  compression_opts=9, chunks=(1, len(smplHJoint)))

    markerMaskD = f.create_dataset(f'marker_pos3d_mask', (len(df), len(PredictedMarkers)),
                                   dtype='f8', compression="gzip",
                                   compression_opts=9, chunks=(1, len(PredictedMarkers)))

    bboxD = f.create_dataset(f'bbox', (len(df), 4),
                             dtype='i8', compression="gzip",
                             compression_opts=9, chunks=(1, 4))

    subjectID = df.loc[0, 'subjectID']
    cameraTypeList = df["cameraType"].tolist()
    bboxInf = bboxInf[subjectID]

    labelPath = None
    for i in trange(len(df)):

        currentLabelPath = df.loc[i, 'labelPath']

        if labelPath is None or currentLabelPath != labelPath:

            labelPath = currentLabelPath

            label = pickle.load(open(labelPath, "rb"))
            numMocapFrame = label["coordinateAngle"].shape[0]
            labelPos = label["label"]
            simulationPos = label["simulation"]

            # 3d marker position from amass
            markerLoc = np.zeros((numMocapFrame, len(PredictedMarkers), 3), dtype=np.float32)
            for idxM, predMarker in enumerate(PredictedMarkers):
                markerLoc[:, idxM, :] = labelPos[:, TrcGenerator.StaticMarkerIndexTable[predMarker], :]

            # 3d joint position from amass
            amass3dLoc = np.zeros((numMocapFrame, len(smplHJoint), 3), dtype=np.float32)
            for idxM, predJoint in enumerate(smplHJoint):
                amass3dLoc[:, idxM, :] = labelPos[:, TrcGenerator.StaticMarkerIndexTable[predJoint], :]

            # 3d marker position from amass
            simMarkerLoc = np.zeros((numMocapFrame, len(PredictedMarkers), 3), dtype=np.float32)
            for idxM, predMarker in enumerate(PredictedMarkers):
                simMarkerLoc[:, idxM, :] = simulationPos[:, TrcGenerator.StaticMarkerIndexTable[predMarker], :]

            # 3d joint position from amass
            simAmass3dLoc = np.zeros((numMocapFrame, len(smplHJoint), 3), dtype=np.float32)
            for idxM, predJoint in enumerate(smplHJoint):
                simAmass3dLoc[:, idxM, :] = simulationPos[:, TrcGenerator.StaticMarkerIndexTable[predJoint], :]

            # if error is above 2 cm, mask is set to 0
            amass3dMask = 1 * (np.linalg.norm(amass3dLoc - simAmass3dLoc, ord=2, axis=-1) <= 0.02)
            markerMask = 1 * (np.linalg.norm(markerLoc - simMarkerLoc, ord=2, axis=-1) <= 0.02)

            # if at least one joint moves over 20 mm , the frame will be used.
            # indices = slice(0, None, imgSampleRate * MocapToVideoRatio)
            previousLocation = None
            usedFrame = np.ones((markerLoc.shape[0],), dtype=np.int)
            for idx in range(markerLoc.shape[0]):
                if previousLocation is None:
                    previousLocation = markerLoc[idx, :, :].copy()
                    continue

                if np.sum((np.sum((markerLoc[idx, :, :] - previousLocation) ** 2, axis=-1) ** 0.5) > (20 / 1000)):
                    previousLocation = markerLoc[idx, :, :].copy()
                else:
                    usedFrame[idx] = 0

            '''
            slices = slice(0, len(df), 5)
            sampledPosition = amass3dLoc[slices, :, :].copy()
            sampledPosition[1:, :, :] -= sampledPosition[:-1, :, :].copy()
            sampledPosition[0:1, :, :] -= sampledPosition[0:1, :, :].copy()
            usedFrame5D = 1 * np.sum(1 * (np.sum(sampledPosition ** 2, axis=-1) ** 0.5) > (100 / 1000), axis=-1) >= 1'''

        frame = df.loc[i, 'frame']
        if frame * MocapToVideoRatio > numMocapFrame:
            mocapFrame = -1
        else:
            mocapFrame = int(frame * MocapToVideoRatio)

        # joint position
        pos3dD[i, :, :] = amass3dLoc[mocapFrame, :, :]
        sim_pos3dD[i, :, :] = simAmass3dLoc[mocapFrame, :, :]

        # marker position
        pos3dD_marker[i, :, :] = markerLoc[mocapFrame, :, :]
        sim_pos3dD_marker[i, :, :] = simMarkerLoc[mocapFrame, :, :]

        # mask
        pos3dMaskD[i, :] = amass3dMask[mocapFrame, :]
        markerMaskD[i, :] = markerMask[mocapFrame, :]

        # used frames
        usedFrameD[i] = usedFrame[mocapFrame]

        globalFrame = df.loc[i, 'globalFrame']
        if bboxInf[cameraTypeList[i]].shape[0] > globalFrame:
            bboxD[i, :] = bboxInf[cameraTypeList[i]][df.loc[i, 'globalFrame'], :]
        else:
            print(
                f'Bbox frame : {bboxInf[cameraTypeList[i]].shape[0]} globalFrame : {globalFrame}. Use the last frame of the video.')
            bboxD[i, :] = bboxInf[cameraTypeList[i]][-1, :]


def create_opensim_label_dataset(path, df, pyBaseModel):
    MocapToVideoRatio = 4
    COMP_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opensimTree = OpenSimTreeLayer(pyBaseModel, PredictedBones, PredictedCoordinates, PredictedOSJoints,
                                   predeict_marker=False, leafJoints=LeafJoints).float().to(COMP_DEVICE)

    with torch.no_grad():
        rot1Axis = torch.Tensor([0, 0, 1]).float()
        rot2Axis = torch.Tensor([1, 0, 0]).float()
        rot3Axis = torch.Tensor([0, 1, 0]).float()

        currentLabelPath = None
        currentSubjectID = None

        with h5py.File(path, 'w') as f:

            openSimJointPosD = f.create_dataset('openSimJointPos',
                                                (len(df), len(PredictedOSJoints) + len(LeafJoints), 3),
                                                dtype='f8', compression="gzip",
                                                compression_opts=9,
                                                chunks=(1, len(PredictedOSJoints) + len(LeafJoints), 3))

            coordinateAngleD = f.create_dataset('coordinateAngle', (len(df), len(PredictedCoordinates)),
                                                dtype='f8', compression="gzip",
                                                compression_opts=9, chunks=(1, len(PredictedCoordinates)))
            coordinateMaskD = f.create_dataset('coordinateMask', (len(df), len(PredictedCoordinates)),
                                               dtype='f8', compression="gzip",
                                               compression_opts=9, chunks=(1, len(PredictedCoordinates)))
            rootRotD = f.create_dataset('rootRot', (len(df), 4, 4),
                                        dtype='f8', compression="gzip",
                                        compression_opts=9, chunks=(1, 4, 4))

            for i in trange(len(df)):

                labelPath = df.loc[i, 'labelPath']
                subjectID = df.loc[i, 'subjectID']
                update = False

                # new label file
                if currentLabelPath is None or labelPath != currentLabelPath:
                    currentLabelPath = labelPath
                    label = pickle.load(open(labelPath, "rb"))
                    numLabelFrames = label["coordinateAngle"].shape[0]
                    coordinateNames = label["coordinateName"]
                    coordinateAngle = np.zeros((numLabelFrames, len(PredictedCoordinates)))
                    coordinateMask = np.zeros((numLabelFrames, len(PredictedCoordinates)))
                    coordinateAngles = label["coordinateAngle"]
                    coordinateMasks = label["coordinateMask"]
                    for coordIdx, name in enumerate(PredictedCoordinates):
                        nameIdx = coordinateNames.index(name)
                        coordinateAngle[:, coordIdx] = coordinateAngles[:, nameIdx]
                        coordinateMask[:, coordIdx] = coordinateMasks[:, nameIdx]
                    update = True

                # new subject
                if currentSubjectID is None or currentSubjectID != subjectID:
                    currentSubjectID = subjectID
                    boneScale = np.zeros((len(PredictedBones), 3), dtype=np.float32)
                    boneScales = label["boneScale"]
                    boneName = label["boneName"]
                    for idx, name in enumerate(PredictedBones):
                        boneScale[idx, :] = boneScales[boneName.index(name), :]

                    boneScaleD = f.create_dataset(f'{subjectID}_boneScale', (boneScale.shape),
                                                  dtype='f8', compression="gzip",
                                                  compression_opts=9)
                    boneScaleD[:, :] = boneScale
                    update = True

                if update:
                    slices = slice(0, label["globalTranslation"].shape[0], MocapToVideoRatio)
                    batchSize = label["globalTranslation"][slices, :].shape[0]

                    # get the rotation of root , R
                    parentFrame = torch.eye(4).unsqueeze(0).repeat(batchSize, 1, 1).float()
                    parentFrame[:, 1, -1] = -0.2351
                    parentFrame[:, 0, -1] += label["globalTranslation"][slices, 0]
                    parentFrame[:, 1, -1] += label["globalTranslation"][slices, 1]
                    parentFrame[:, 2, -1] += label["globalTranslation"][slices, 2]

                    # global rotation matrix
                    C1 = torch.Tensor(coordinateAngles[slices, 0:1]).float()
                    C2 = torch.Tensor(coordinateAngles[slices, 1:2]).float()
                    C3 = torch.Tensor(coordinateAngles[slices, 2:3]).float()

                    # The rotation mat of first coordinate
                    R1 = OpenSimNode.axangle2mat(rot1Axis.unsqueeze(0).repeat(batchSize, 1), C1)

                    # The rotation mat of second coordinate
                    temp = torch.einsum('bij , j -> bi', R1, rot2Axis)
                    R2 = OpenSimNode.axangle2mat(temp, C2)

                    # The rotation mat of third coordinate
                    temp = torch.einsum('bij , j -> bi', R1, rot3Axis)
                    temp = torch.einsum('bij , bj -> bi', R2, temp)
                    R3 = OpenSimNode.axangle2mat(temp, C3)

                    # The overall rotation mat
                    # x = torch.einsum('bij , jk -> bik', R1, eyeMatrix)
                    x = torch.matmul(R2, R1)
                    x = torch.matmul(R3, x)
                    R = OpenSimNode.rotMat_to_homogeneous_matrix(x, batchSize)
                    R = torch.einsum('bij , bjk -> bik', parentFrame, R)

                    '''
                    R1 = OpenSimNode.axangle2mat(rot1Axis, C1)
                    temp = torch.einsum('bij , jk -> bik', R1, rot2Axis) #torch.matmul(R1, rot2Axis)
                    R2 = OpenSimNode.axangle2mat(temp, C2)
                    temp = torch.einsum('bij , jk -> bik', R1, rot3Axis)#torch.matmul(R1, rot3Axis)
                    temp = torch.einsum('bij , bjk -> bik', R2, temp) #torch.matmul(R2, temp)
                    R3 = OpenSimNode.axangle2mat(temp, C3)
                    x = torch.matmul(R2, R1)
                    x = torch.matmul(R3, x)
                    R = OpenSimNode.rotMat_to_homogeneous_matrix(x, 0)
                    R = torch.einsum('bij , bjk -> bik', parentFrame, R)'''

                    # calculate the 3d joint location
                    x = {}
                    x["predBoneScale"] = torch.from_numpy(boneScale).float().unsqueeze(0).repeat(batchSize, 1, 1).to(
                        COMP_DEVICE)
                    x["predRot"] = torch.Tensor(coordinateAngle[slices, :]).float().to(COMP_DEVICE)
                    x["rootRot"] = R.to(COMP_DEVICE)
                    treePred = opensimTree(x)
                    openSimJointPos = treePred["predJointPos"].cpu().detach().numpy()

                # get the target frame
                frame = df.loc[i, 'frame']

                if (frame * MocapToVideoRatio) > numLabelFrames:
                    mocapFrame = -1
                    print(f'{df.loc[i, "imgName"]} uses the last frame in the label.')
                else:
                    mocapFrame = int(frame * MocapToVideoRatio)

                # openSimJointPos = treePred["predJointPos"][:, :, :].cpu().detach().numpy()
                # openSimJointPos = np.array(openSimJointPos)

                openSimJointPosD[i, :, :] = openSimJointPos[frame if frame < openSimJointPos.shape[0] else -1, :, :]
                coordinateAngleD[i, :] = coordinateAngle[mocapFrame, :]
                coordinateMaskD[i, :] = coordinateMask[mocapFrame, :]
                rootRotD[i, :, :] = R[frame if frame < openSimJointPos.shape[0] else -1, :, :]


if __name__ == "__main__":
    pass
