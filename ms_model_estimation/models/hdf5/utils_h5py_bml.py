import argparse
import os
from glob import glob
from skimage import io
import h5py
import torch
import pickle
import numpy as np
import pandas as pd
from ms_model_estimation.models.networks.model_layer.OpenSimNode import OpenSimNode
from ms_model_estimation.models.networks.model_layer.OpenSimTreeLayer import OpenSimTreeLayer
from ms_model_estimation.models.utils.OSUtils import PredictedBones, PredictedCoordinates, PredictedJointPosition, PredictedMarkers
from opensim.DataReader import DataReader
from opensim import OpenSimModel
from ms_model_estimation.smplh.smplh_vertex_index import smplHJoint


def search_bml_data_list(bboxFolder, opensimGTFolder, fullBodyPath, postfix="_MoSh"):

    boxFolder = bboxFolder + "/" if not bboxFolder.endswith("/") else bboxFolder
    opensimGTFolder = opensimGTFolder + "/" if not bboxFolder.endswith("/") else opensimGTFolder
    pyBaseModel = DataReader.read_opensim_model(fullBodyPath)

    labelList = []
    imageList = []
    subjectIDList = []
    trcList = []
    frameIDList = []
    videoIDList = []
    cameraTypeList = []

    imagePathList = glob(bboxFolder + '*/*.png')
    imagePathList = [x.replace('\\', "/") for x in imagePathList]

    for imgPath in imagePathList:

        folder = imgPath.split("/")[-2]
        filename = imgPath.split("/")[-1]
        subjectID = int(filename.split(".")[0].split("_")[3])
        videoID = int(filename.split(".")[1])
        frameID = int(filename.split(".")[2])
        cameraType = filename.split("_")[1]
        labelFileName = "_".join([folder, str(videoID), "poses_gt.pkl"])
        labelPath = "/".join([opensimGTFolder + folder + postfix, "Results", labelFileName])
        trcFileName = "_".join([folder, str(videoID), "poses.trc"])
        trcPath = "/".join([opensimGTFolder + folder + postfix, "MotionTrcData", trcFileName])

        if os.path.exists(labelPath):
            subjectIDList.append(subjectID)
            imageList.append("/".join([bboxFolder + folder, filename]))
            labelList.append(labelPath)
            trcList.append(trcPath)
            frameIDList.append(frameID)
            videoIDList.append(videoID)
            cameraTypeList.append(cameraType)

    df = None
    if subjectIDList:
        df = {
            'subjectID': subjectIDList, 'videoID': videoIDList, 'frameID': frameIDList,
            'patchPath': imageList, 'labelPath': labelList, 'trcPath': trcList,
            'cameraType': cameraTypeList
        }
        df = pd.DataFrame(data=df)

    return {
        "df": df,
        "pyBaseModel": pyBaseModel
    }


def split_dataset(
        x, training=0.75,
        valid=0.05, test=0.2, seed=1
):
    np.random.seed(seed)

    df = x["df"]
    subjectIDList = x["df"]["subjectID"].tolist()

    total = training + valid + test
    training, valid, test = training / total, valid / total, test / total

    subjects = np.array([s for s in range(1, 91)])
    numSubjects = len(subjects)

    training = int(training * numSubjects)
    valid = int(valid * numSubjects)

    indices = np.random.permutation(numSubjects)
    trainingSubjectID = subjects[indices[:training]].tolist()
    valSubjectID = subjects[indices[training:(training + valid)]].tolist()
    testSubjectID = subjects[indices[(training + valid):]].tolist()

    trainingSubjectIndices = []
    valSubjectIndices = []
    testSubjectIndices = []
    for idx, subjectIdx in enumerate(subjectIDList):
        if subjectIdx in trainingSubjectID:
            trainingSubjectIndices.append(idx)
        elif subjectIdx in valSubjectID:
            valSubjectIndices.append(idx)
        elif subjectIdx in testSubjectID:
            testSubjectIndices.append(idx)
        else:
            assert False

    trainingDF = df.iloc[trainingSubjectIndices, :].sort_values(
        by=["subjectID", "cameraType", "videoID", "frameID"]).reset_index(drop=True)
    validDF = df.iloc[valSubjectIndices, :].sort_values(
        by=["subjectID", "cameraType", "videoID", "frameID"]).reset_index(drop=True)
    testDF = df.iloc[testSubjectIndices, :].sort_values(
        by=["subjectID", "cameraType", "videoID", "frameID"]).reset_index(drop=True)

    table = {
        "trainingDF": trainingDF,
        "validDF": validDF,
        "testDF": testDF,
        "pyBaseModel": x["pyBaseModel"]
    }

    return table


def store_as_h5py(outputFolder, table, bboxInfPath, fpsRatio=4):
    '''
    # create image dataset
    path =  outputFolder + "train_img.hdf5"
    create_image_dataset(path, table["trainingDF"])

    path = outputFolder + "test_img.hdf5"
    create_image_dataset(path, table["testDF"])'''

    path = outputFolder + "valid_img.hdf5"
    create_image_dataset(path, table["validDF"])

    # create opensim dataset

    pyBaseModel = table["pyBaseModel"]
    '''
    path = outputFolder + "train.hdf5"  
    create_h5py(path, table["trainingDF"], pyBaseModel, fpsRatio=fpsRatio)

    path = outputFolder + "test.hdf5" 
    create_h5py(path, table["testDF"], pyBaseModel, fpsRatio=fpsRatio)'''

    path = outputFolder + "valid.hdf5"
    create_h5py(path, table["validDF"], pyBaseModel, fpsRatio=fpsRatio)

    path = outputFolder + "camera_setting.hdf5"
    create_camera_setting(path, bboxInfPath)


def create_image_dataset(path, df):
    patchList = df["patchPath"].tolist()
    numData = 2000
    numDataset = len(patchList) // numData
    numDataset = 1 if numDataset == 0 else numDataset

    for i in range(numDataset):

        tmpPath = path.replace(".hdf5", f'_{i + 1}.hdf5')
        if i != (numDataset - 1):
            tmpList = patchList[numData * i:numData * (i + 1)]
        else:
            tmpList = patchList[numData * i:]

        with h5py.File(tmpPath, 'w') as f:
            for imgPath in tmpList:
                img = np.array(io.imread(imgPath), dtype=np.float64) / 255
                imgPath = "/".join(imgPath.split("/")[-3:])
                imgDataset = f.create_dataset(f'{imgPath}', (img.shape), dtype='f8', compression="gzip",
                                              compression_opts=9)
                imgDataset[:, :, :] = img


def create_h5py(path, df, pyBaseModel, fpsRatio=4):
    patchList = df["patchPath"].tolist()
    labelList = df["labelPath"].tolist()
    trcList = df["trcPath"].tolist()
    videoIDList = df["videoID"].tolist()
    frameIDList = df["frameID"].tolist()
    cameraTypeList = df["cameraType"].tolist()

    imgList = ["/".join(imgPath.split("/")[-3:]) for imgPath in patchList]
    with h5py.File(path, 'w') as f:
        ImageListD = f.create_dataset('ImgList', (len(imgList),), dtype='S150', compression="gzip",
                                      compression_opts=9)
        ImageListD[:] = imgList

        LabelListD = f.create_dataset('LabelList', (len(labelList),), dtype='S150', compression="gzip",
                                      compression_opts=9)
        LabelListD[:] = labelList

        videoIDListD = f.create_dataset('videoID', (len(videoIDList),), dtype='i8', compression="gzip",
                                        compression_opts=9)
        videoIDListD[:] = videoIDList

        frameIDListD = f.create_dataset('frameID', (len(frameIDList),), dtype='i8', compression="gzip",
                                        compression_opts=9)
        frameIDListD[:] = frameIDList

        cameraTypeD = f.create_dataset('cameraType', (len(cameraTypeList),), dtype='S5', compression="gzip",
                                       compression_opts=9)
        cameraTypeD[:] = cameraTypeList

        create_mapping(f, patchList, labelList)
        create_label_dataset(f, patchList, labelList, trcList, pyBaseModel, fpsRatio=fpsRatio)


def create_mapping(f, patchList, labelList):
    seenID = {}
    index = 0

    # mapping the data to the local index in the coordinateAngle dataset and coordinateMask dataset
    labelIDList = []
    subjectID_videoTypeList = []

    for idx, (imgPath, labelPath) in enumerate(zip(patchList, labelList)):

        subjectID = (imgPath.split("/")[-1].split(".")[0].split("_")[-2])
        videoType = imgPath.split("/")[-1].split("_")[1]
        subjectID_videoType = f'{subjectID}_{videoType}'

        if subjectID_videoType not in seenID:
            seenID[subjectID_videoType] = 1
            index = 0
        labelIDList.append(index)
        index += 1
        subjectID_videoTypeList.append(subjectID_videoType)

    # save the mapping
    localIndexD = f.create_dataset('localIndex', (len(labelIDList),),
                                   dtype='i8', compression="gzip",
                                   compression_opts=9)
    localIndexD[:] = labelIDList

    subjectIDD = f.create_dataset('subjectID', (len(subjectID_videoTypeList),),
                                  dtype='S64', compression="gzip",
                                  compression_opts=9)
    subjectIDD[:] = subjectID_videoTypeList


def create_label_dataset(f, patchList, labelList, trcList, pyBaseModel, fpsRatio=4):
    opensimTree = OpenSimTreeLayer(pyBaseModel, PredictedBones, PredictedCoordinates[3:], PredictedJointPosition,
                                   predict_projection=False, predeict_marker=False)

    seenID = {}
    coordinateAngleList = []
    coordinateMaskList = []
    rootRot = []
    amass3dJointList = []
    marker3dList = []
    joint3dPosList = []
    amass2dJointList = []
    marker2dList = []
    joint2dPosList = []

    subjectID_videoType = ''
    previousIdxVideo = -1
    rot1Axis = torch.Tensor([0, 0, 1]).float()
    rot2Axis = torch.Tensor([1, 0, 0]).float()
    rot3Axis = torch.Tensor([0, 1, 0]).float()

    for idx, (imgPath, labelPath, trcPath) in enumerate(zip(patchList, labelList, trcList)):

        label = pickle.load(open(labelPath, "rb"))

        previousSubject = subjectID_videoType
        subjectID = (imgPath.split("/")[-1].split(".")[0].split("_")[-2])
        videoType = imgPath.split("/")[-1].split("_")[1]
        subjectID_videoType = f'{subjectID}_{videoType}'

        if subjectID_videoType not in seenID:
            seenID[subjectID_videoType] = 1

            # subject changes, create coordinate angle/mask dataset for the previous subject
            if coordinateAngleList:
                coordinateAngleList = np.array(coordinateAngleList, dtype=np.float64)
                coordinateMaskList = np.array(coordinateMaskList, dtype=np.float64)
                rootRot = np.array(rootRot, dtype=np.float64)
                marker3dList = np.array(marker3dList, dtype=np.float64)
                joint3dPosList = np.array(joint3dPosList, dtype=np.float64)
                amass3dJointList = np.array(amass3dJointList, dtype=np.float64)
                amass2dJointList = np.array(amass2dJointList, dtype=np.float64)
                marker2dList = np.array(marker2dList, dtype=np.float64)
                joint2dPosList = np.array(joint2dPosList, dtype=np.float64)

                # hdf5 dataset
                # coordinate angle
                coordinateAngleD = f.create_dataset(f'{previousSubject}_coordinateAngle',
                                                    (coordinateAngleList.shape),
                                                    dtype='f8', compression="gzip",
                                                    compression_opts=9)
                coordinateAngleD[:, :] = coordinateAngleList

                # coordinate mask
                coordinateMaskD = f.create_dataset(f'{previousSubject}_coordinateMask',
                                                   (coordinateMaskList.shape),
                                                   dtype='f8',
                                                   compression="gzip",
                                                   compression_opts=9)
                coordinateMaskD[:, :] = coordinateMaskList

                # root rotation matrix
                rootRotD = f.create_dataset(f'{previousSubject}_rootRot',
                                            (rootRot.shape),
                                            dtype='f8',
                                            compression="gzip",
                                            compression_opts=9)
                rootRotD[:, :, :] = rootRot

                # 3d marker location
                markerLocD = f.create_dataset(f'{previousSubject}_markerLoc',
                                              (marker3dList.shape),
                                              dtype='f8',
                                              compression="gzip",
                                              compression_opts=9)
                markerLocD[:, :, :] = marker3dList

                # 3d opensim joint position
                jointPosD = f.create_dataset(f'{previousSubject}_pos',
                                             (joint3dPosList.shape),
                                             dtype='f8',
                                             compression="gzip",
                                             compression_opts=9)
                jointPosD[:, :, :] = joint3dPosList

                # 3d amass joint position
                amassJointPosD = f.create_dataset(f'{previousSubject}_amass_pos',
                                                  (amass3dJointList.shape),
                                                  dtype='f8',
                                                  compression="gzip",
                                                  compression_opts=9)
                amassJointPosD[:, :, :] = amass3dJointList

                # 2d opensim joint position
                joint2dPosD = f.create_dataset(f'{previousSubject}_2d_pos',
                                               (joint2dPosList.shape),
                                               dtype='f8',
                                               compression="gzip",
                                               compression_opts=9)
                joint2dPosD[:, :, :] = joint2dPosList

                # 2d marker position
                marker2dPosD = f.create_dataset(f'{previousSubject}_2d_markerLoc',
                                                (marker2dList.shape),
                                                dtype='f8',
                                                compression="gzip",
                                                compression_opts=9)
                marker2dPosD[:, :, :] = marker2dList

                # 2d amass joint position
                amassJoint2dPosD = f.create_dataset(f'{previousSubject}_2d_amass_pos',
                                                    (amass2dJointList.shape),
                                                    dtype='f8',
                                                    compression="gzip",
                                                    compression_opts=9)
                amassJoint2dPosD[:, :, :] = amass2dJointList

            # renew the list
            coordinateAngleList = []
            coordinateMaskList = []
            rootRot = []
            amass3dJointList = []
            marker3dList = []
            joint3dPosList = []
            amass2dJointList = []
            marker2dList = []
            joint2dPosList = []

            # new subject, create bone scale dataset
            boneScale = np.array(label["boneScale"], dtype=np.float64)
            boneScaleD = f.create_dataset(f'{subjectID_videoType}_scale', (boneScale.shape), dtype='f8',
                                          compression="gzip",
                                          compression_opts=9)
            boneScaleD[:, :] = boneScale

        # get the target frame
        inf = imgPath.split("/")[-1].split(".")
        indxVideo, frames = inf[1], int(inf[2])
        numLabelFrames = label["coordinateAngle"].shape[0]
        # videoType = imgPath.split("/")[-1].split("_")[1]
        if frames * fpsRatio > numLabelFrames:
            frames = -1
        else:
            frames = int(frames * fpsRatio)

        # get coordinate Angle and Mask
        coordinateAngle = label["coordinateAngle"][frames, :]
        coordinateMask = label["coordinateMask"][frames, :]
        coordinateAngleList.append(coordinateAngle)
        coordinateMaskList.append(coordinateMask)

        # extrinstic camera parameters
        extrinsicCamera , intrinsicCamera = CAMERA_PARAMETER[videoType]

        # get the rotation of root , R
        parentFrame = torch.eye(4).float()
        parentFrame[1, -1] = -0.2351
        parentFrame[0, -1] += label["globalTranslation"][frames, 0]
        parentFrame[1, -1] += label["globalTranslation"][frames, 1]
        parentFrame[2, -1] += label["globalTranslation"][frames, 2]
        C1 = torch.Tensor(coordinateAngle[0:1]).float()
        C2 = torch.Tensor(coordinateAngle[1:2]).float()
        C3 = torch.Tensor(coordinateAngle[2:3]).float()
        R1 = OpenSimNode.axangle2mat(rot1Axis, C1)
        temp = torch.matmul(R1, rot2Axis)
        R2 = OpenSimNode.axangle2mat(temp, C2)
        temp = torch.matmul(R1, rot3Axis)
        temp = torch.matmul(R2, temp)
        R3 = OpenSimNode.axangle2mat(temp, C3)
        x = torch.matmul(R2, R1)
        x = torch.matmul(R3, x)
        R = OpenSimNode.rotMat_to_homogeneous_matrix(x, 0)
        R = torch.einsum('ij , jk -> ik', parentFrame, R)
        R = torch.einsum('ij , jk -> ik', extrinsicCamera, R)

        # 3d marker position from amass
        extrinsicCamera = np.array(extrinsicCamera)

        # if new subject and new video , read load trc file
        if previousIdxVideo != indxVideo or previousSubject != subjectID_videoType:
            markerLocTable = OpenSimModel.read_trc_file(trcPath)
            previousIdxVideo = indxVideo

        markerLoc = np.zeros((len(PredictedMarkers), 3))
        for idxM, predMarker in enumerate(PredictedMarkers):
            markerLoc[idxM, :] = markerLocTable[predMarker][frames]
        tmp = np.ones((markerLoc.shape[0], 4))
        tmp[:, :3] = markerLoc
        markerLoc = tmp
        markerLoc = extrinsicCamera.dot(markerLoc.T).T[:, :3]
        marker3dList.append(markerLoc)

        # 3d joint position from amass
        amass3dLoc = np.zeros((len(smplHJoint), 3))
        for idxM, predMarker in enumerate(smplHJoint):
            amass3dLoc[idxM, :] = markerLocTable[predMarker][frames]
        tmp = np.ones((amass3dLoc.shape[0], 4))
        tmp[:, :3] = amass3dLoc
        amass3dLoc = tmp
        amass3dLoc = extrinsicCamera.dot(amass3dLoc.T).T[:, :3]
        amass3dJointList.append(amass3dLoc)

        # calculate the 3d joint location
        x = {}
        x["predBoneScale"] = torch.from_numpy(boneScale).float().unsqueeze(0)
        x["predRot"] = torch.Tensor(coordinateAngle[3:]).float().unsqueeze(0)
        x["rootRot"] = R.unsqueeze(0)
        pred = opensimTree(x)
        predJointPos = pred["predJointPos"][0, :, :]
        predJointPos = predJointPos[:, :]
        predJointPos = np.array(predJointPos)
        joint3dPosList.append(predJointPos)
        rootRot.append(np.array(R))

        # calculate 2d marker / opensim / amass joint position
        amass2dJointList.append(get_2d_projection(amass3dLoc.copy(), intrinsicCamera))
        marker2dList.append(get_2d_projection(markerLoc.copy(), intrinsicCamera))
        joint2dPosList.append(get_2d_projection(predJointPos.copy(), intrinsicCamera))

    # the last one subject
    if coordinateAngleList:
        coordinateAngleList = np.array(coordinateAngleList, dtype=np.float64)
        coordinateMaskList = np.array(coordinateMaskList, dtype=np.float64)
        rootRot = np.array(rootRot, dtype=np.float64)
        marker3dList = np.array(marker3dList, dtype=np.float64)
        joint3dPosList = np.array(joint3dPosList, dtype=np.float64)
        amass3dJointList = np.array(amass3dJointList, dtype=np.float64)
        amass2dJointList = np.array(amass2dJointList, dtype=np.float64)
        marker2dList = np.array(marker2dList, dtype=np.float64)
        joint2dPosList = np.array(joint2dPosList, dtype=np.float64)

        # hdf5 dataset
        # coordinate angle
        coordinateAngleD = f.create_dataset(f'{previousSubject}_coordinateAngle',
                                            (coordinateAngleList.shape),
                                            dtype='f8', compression="gzip",
                                            compression_opts=9)
        coordinateAngleD[:, :] = coordinateAngleList

        # coordinate mask
        coordinateMaskD = f.create_dataset(f'{previousSubject}_coordinateMask',
                                           (coordinateMaskList.shape),
                                           dtype='f8',
                                           compression="gzip",
                                           compression_opts=9)
        coordinateMaskD[:, :] = coordinateMaskList

        # root rotation matrix
        rootRotD = f.create_dataset(f'{previousSubject}_rootRot',
                                    (rootRot.shape),
                                    dtype='f8',
                                    compression="gzip",
                                    compression_opts=9)
        rootRotD[:, :, :] = rootRot

        # 3d marker location
        markerLocD = f.create_dataset(f'{previousSubject}_markerLoc',
                                      (marker3dList.shape),
                                      dtype='f8',
                                      compression="gzip",
                                      compression_opts=9)
        markerLocD[:, :, :] = marker3dList

        # 3d opensim joint position
        jointPosD = f.create_dataset(f'{previousSubject}_pos',
                                     (joint3dPosList.shape),
                                     dtype='f8',
                                     compression="gzip",
                                     compression_opts=9)
        jointPosD[:, :, :] = joint3dPosList

        # 3d amass joint position
        amassJointPosD = f.create_dataset(f'{previousSubject}_amass_pos',
                                          (amass3dJointList.shape),
                                          dtype='f8',
                                          compression="gzip",
                                          compression_opts=9)
        amassJointPosD[:, :, :] = amass3dJointList

        # 2d opensim joint position
        joint2dPosD = f.create_dataset(f'{previousSubject}_2d_pos',
                                       (joint2dPosList.shape),
                                       dtype='f8',
                                       compression="gzip",
                                       compression_opts=9)
        joint2dPosD[:, :, :] = joint2dPosList

        # 2d marker position
        marker2dPosD = f.create_dataset(f'{previousSubject}_2d_markerLoc',
                                        (marker2dList.shape),
                                        dtype='f8',
                                        compression="gzip",
                                        compression_opts=9)
        marker2dPosD[:, :, :] = marker2dList

        # 2d amass joint position
        amassJoint2dPosD = f.create_dataset(f'{previousSubject}_2d_amass_pos',
                                            (amass2dJointList.shape),
                                            dtype='f8',
                                            compression="gzip",
                                            compression_opts=9)
        amassJoint2dPosD[:, :, :] = amass2dJointList

'''
def get_2d_projection(position3d, intrisicParameters):
    position3d = position3d / position3d[:, -1:]
    return position3d.dot(intrisicParameters)


def create_camera_setting(path, bboxInfPath):
    bboxInf = pickle.load(open(bboxInfPath, "rb"))

    with h5py.File(path, 'w') as f:

        # PG1
        pg1_extrinsicCamera = torch.Tensor([
            [-0.99942183, 0.0305191, -0.01498661, 0.17723154],
            [-0.01303452, 0.06318672, 0.9979166, -1.03055751],
            [-0.03140247, -0.99753497, 0.06275239, -4.99931781],
            [0.0000, 0.0000, 0.0000, 1.0000]
        ]).float()
        pg1_intrinsicCamera = np.array(
            [[979.17889011, 0., 0.],
             [0., 978.10179305, 0.],
             [408.0273103, 291.16967878, 1.]]
        )

        extrinsicCameraD = f.create_dataset('PG1_extrinsic',
                                            (pg1_extrinsicCamera.shape),
                                            dtype='f8',
                                            compression="gzip",
                                            compression_opts=9)
        extrinsicCameraD[:, :] = pg1_extrinsicCamera

        intrinsicCameraD = f.create_dataset('PG1_intrinsic',
                                            (pg1_intrinsicCamera.shape),
                                            dtype='f8',
                                            compression="gzip",
                                            compression_opts=9)
        intrinsicCameraD[:, :] = pg1_intrinsicCamera

        # PG2
        pg2_extrinsicCamera = torch.Tensor([
            [-0.04569302, 0.9988672, 0.01328404, 0.24368025],
            [0.17839144, -0.00492513, 0.98394727, -0.67257332],
            [-0.98289808, -0.04732928, 0.17796432, -4.05087267],
            [0.0000, 0.0000, 0.0000, 1.0000]
        ]).float()
        pg2_intrinsicCamera = np.array(
            [[980.04337094, 0., 0.],
             [0., 980.69881345, 0.],
             [392.27529092, 309.91125524, 1.]]
        )

        extrinsicCameraD = f.create_dataset('PG2_extrinsic',
                                            (pg2_extrinsicCamera.shape),
                                            dtype='f8',
                                            compression="gzip",
                                            compression_opts=9)
        extrinsicCameraD[:, :] = pg2_extrinsicCamera

        intrinsicCameraD = f.create_dataset('PG2_intrinsic',
                                            (pg2_intrinsicCamera.shape),
                                            dtype='f8',
                                            compression="gzip",
                                            compression_opts=9)
        intrinsicCameraD[:, :] = pg2_intrinsicCamera

        # calculate the perspective correction
        for imgPath, value in bboxInf.items():

            (row, col), width = value
            imgPath = imgPath.split("/")[-1]
            videoType = imgPath.split("_")[1]
            if videoType == "PG1":
                focal = pg1_intrinsicCamera[0, 0] + pg1_intrinsicCamera[1, 1]
                focal /= 2
            elif videoType == "PG2":
                focal = pg2_intrinsicCamera[0, 0] + pg2_intrinsicCamera[1, 1]
                focal /= 2

            x = 400 - col
            y = 300 - row
            thetaX = math.atan(x / focal)
            thetaY = math.atan(y / (focal ** 2 + x ** 2 + y ** 2) ** 0.5)

            R = euler2mat(thetaX, thetaY, 0)

            bbD = f.create_dataset(f'{imgPath}_bb',
                                   (3,),
                                   dtype='f8',
                                   compression="gzip",
                                   compression_opts=9)
            bbD[:] = [row, col, width]

            correctionD = f.create_dataset(f'{imgPath}_R',
                                           (R.shape),
                                           dtype='f8',
                                           compression="gzip",
                                           compression_opts=9)
            correctionD[:, :] = R


def convert_image_to_pos(poseModelPath, datasetPath, h5pyImgPath, numPredJoints, numPredMarkers, numImage=2000,
                         multipleImgDataset=True, postfix="pos"):
    outputPath = datasetPath.replace(".hdf5", f'_{postfix}.hdf5')

    transform = transforms.Compose([
        ToTensor(),
        Rescale((256, 256)),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    model = PoseEstimationModel(numPredJoints, numPredMarkers).float()
    model.load_state_dict(torch.load(poseModelPath))
    model.eval()

    with h5py.File(datasetPath, 'r') as f:
        numData = len(f["subjectID"][:])
        patchList = f[f'ImgList'][:]

    with torch.no_grad():
        with h5py.File(outputPath, 'w') as fw:

            posD = fw.create_dataset('intermediate_pos',
                                     (numData, numPredJoints + numPredMarkers, 3),
                                     dtype='f8',
                                     compression="gzip",
                                     compression_opts=9)

            if not multipleImgDataset:

                for idx in range(numData):
                    imagePath = patchList[idx]
                    with h5py.File(h5pyImgPath, 'r') as f:
                        imgPath = imagePath.decode()
                        image = f[f'{imgPath}'][:, :, :]
                    image = transform({"image": image})["image"]
                    predPos = model(image.unsqueeze(0)).squeeze(0).detach().numpy()
                    posD[idx, :, :] = predPos.detach().numpy()

            else:

                numDataset = numData // numImage
                numDataset = 1 if numDataset == 0 else numDataset

                idx = 0
                for i in range(numDataset):

                    if i != (numDataset - 1):
                        tmpList = patchList[numImage * i:numImage * (i + 1)]
                    else:
                        tmpList = patchList[numImage * i:]

                    for imgPath in tmpList:
                        with h5py.File(h5pyImgPath.replace(".hdf5", f'_{i + 1}.hdf5'), 'r') as f:
                            imgPath = imgPath.decode()
                            image = f[f'{imgPath}'][:, :, :]

                        image = transform({"image": image})["image"]
                        predPos = model(image.unsqueeze(0)).squeeze(0).detach().numpy()
                        posD[idx, :, :] = predPos
                        idx += 1

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('bboxFolder', action='store', type=str,
                        help="The bouning box folder")
    parser.add_argument('opensimGTFolder', action='store', type=str,
                        help="The opensim gt folder")
    parser.add_argument('fullBodyPath', action='store', type=str,
                        help="The full body")
    parser.add_argument('bboxInfPath', action='store', type=str,
                        help="bboxInfPath")
    parser.add_argument('outputPath', action='store', type=str,
                        help="The output folder")
    args = parser.parse_args()

    table = search_bml_data_list(args.bboxFolder, args.opensimGTFolder, args.fullBodyPath)
    table = split_dataset(table)
    store_as_h5py(args.outputPath, table, args.bboxInfPath)
