import os
import h5py
import torch
import numpy as np
from ms_model_estimation.training.dataset.TorchDataset import TorchDataset
from ms_model_estimation.training.dataset.data_loading import load_and_transform3d


class BMLImgOpenSimDataSet_Conv3d(TorchDataset):

    def __init__(
            self, cfg, h5pyBMLFolder, cameraParameter, datasetType, evaluation=True, useEveryFrame=False,
            usePredefinedSeed=False
    ):
        super(BMLImgOpenSimDataSet_Conv3d, self).__init__(cfg, evaluation=evaluation)
        np.random.seed(self.cfg.SEED)
        self.evaluation = evaluation

        self.h5pyBMLFolder = h5pyBMLFolder
        self.useEveryFrame = useEveryFrame
        self.cameraParameter = cameraParameter
        self.usePredefinedSeed = usePredefinedSeed

        self.usedSubjects = []
        if datasetType == 0:
            self.usedSubjects = self.cfg.TRAINING_SUBJECTS
        elif datasetType == 1:
            self.usedSubjects = self.cfg.VALID_SUBJECTS
        elif datasetType == 2:
            self.usedSubjects = self.cfg.TEST_SUBJECTS
        else:
            raise Exception("The type of the dataset are not defined")
        self.datasetType = datasetType

        self.usedEachFrame = self.cfg.DATASET.TRAINING_USEDEACHFRAME if not self.evaluation else self.cfg.DATASET.USEDEACHFRAME
        self.create_mapping()

        # self.awaredOcclusion = AwaredOcclusion(cfg.PREDICTION.POS_NAME, cfg.PREDICTION.Joints_Inside_Cylinder_Body_LIST)

    def create_mapping(self):

        # create mapping to idx : [subjectID, cameraType, index of local hdf5]

        self.usedIndices = []
        for subject in self.usedSubjects:
            if os.path.exists(self.h5pyBMLFolder + f'subject_{subject}.hdf5'):
                with h5py.File(self.h5pyBMLFolder + f'subject_{subject}.hdf5', 'r') as f:
                    globalFrames = f["globalFrame"][:]
                    Frames = f["frame"][:]
                    cameraTypes = f["cameraType"][:]
                    usedFrames = f['usedFrame'][:]
                with h5py.File(self.h5pyBMLFolder + f'subject_{subject}_opensim.hdf5', 'r') as f:
                    coordinateMask = f['coordinateMask'][:, :]

                if globalFrames.shape[0] != coordinateMask.shape[0]:
                    print(
                        f'Subject {subject} , label shape : {globalFrames.shape[0]} , opensim shape : {coordinateMask.shape[0]}')

                for localIdx, globalFrame in enumerate(globalFrames):
                    if self.useEveryFrame or (Frames[localIdx] % self.usedEachFrame) == 0 and (
                            self.evaluation or usedFrames[localIdx]):
                        self.usedIndices.append([subject, cameraTypes[localIdx].decode(), localIdx])

        self.usedIndices = self.usedIndices

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        subjectID, cameraType, localIdx = self.usedIndices[idx]

        with h5py.File(self.h5pyBMLFolder + f'subject_{subjectID}.hdf5', 'r') as f:
            videoStartFrame = int(f["startFrame"][localIdx])
            frame = int(f["globalFrame"][localIdx])
            videoEndFrame = int(f["globalEndFrame"][localIdx])

        backwardRepeatSize = 0
        forwardRepeatSize = 0
        if not self.cfg.MODEL.CAUSAL:

            # the frame in the middle is the target for prediction
            endFrame = frame + self.cfg.MODEL.RECEPTIVE_FIELD // 2 - 1 * (self.cfg.MODEL.RECEPTIVE_FIELD % 2 == 0)
            if endFrame > videoEndFrame:
                backwardRepeatSize = endFrame - videoEndFrame
                endFrame = videoEndFrame

            startFrame = frame - self.cfg.MODEL.RECEPTIVE_FIELD // 2
            if startFrame < videoStartFrame:
                forwardRepeatSize = videoStartFrame - startFrame
                startFrame = videoStartFrame

        else:
            # casual
            # the last frame is the target for prediction
            endFrame = frame
            startFrame = frame - self.cfg.MODEL.RECEPTIVE_FIELD + 1

            if startFrame < videoStartFrame:
                forwardRepeatSize = videoStartFrame - startFrame
                startFrame = videoStartFrame

        with h5py.File(self.h5pyBMLFolder + f'subject_{subjectID}.hdf5', 'r') as f:
            pos3d = f['pos3d'][localIdx, :self.cfg.PREDICTION.NUM_SMPL_JOINTS, :]
            marker_pose3d = f['marker_pos3d'][localIdx,
                            :(self.cfg.PREDICTION.NUM_MARKERS - self.cfg.PREDICTION.NUM_SMPL_JOINTS), :]

            posMask = f['pos3d_mask'][localIdx, :self.cfg.PREDICTION.NUM_JOINTS]
            markerMask = f['marker_pos3d_mask'][localIdx,
                         :(self.cfg.PREDICTION.NUM_MARKERS - self.cfg.PREDICTION.NUM_SMPL_JOINTS)]

            bboxes = f['bbox'][startFrame - frame + localIdx: endFrame + 1 - frame + localIdx, :]
            globalFrame = f['globalFrame'][localIdx]
            #hflipUsage = f['hflipUsage'][localIdx]
            videoID = int(f['videoID'][localIdx])
            name = f['ImgList'][localIdx].decode()

        with h5py.File(self.h5pyBMLFolder + f'subject_{subjectID}_opensim.hdf5', 'r') as f:
            openSimJointPos = f['openSimJointPos'][localIdx, :, :]

            coordinateAngle = f['coordinateAngle'][localIdx, :]

            coordinateMask = f['coordinateMask'][localIdx, :]

            boneScale = f[f'{subjectID}_boneScale'][:, :]

            rootRot = f['rootRot'][localIdx, :3, :4]

        smplhmarkers = np.concatenate((pos3d, marker_pose3d), axis=0)
        # concatenate the joint and the marker
        pos3d = np.concatenate((openSimJointPos, smplhmarkers), axis=0)

        # get camera
        camera = self.cameraParameter[cameraType].copy()

        with h5py.File(self.h5pyBMLFolder + f'{subjectID}_{cameraType}_img.hdf5', 'r') as f:
            images = f["images"][startFrame - frame + globalFrame: endFrame + 1 - frame + globalFrame, :, :, :][:, :, :,
                     [2, 1, 0]]

        seed = np.random.randint(25000)
        # data augmentation
        image = images[frame - startFrame, :, :, :]
        bbox = bboxes[frame - startFrame, :]
        image, pos3d, pos2d, joint3d_validity_mask, _, hFlip, rootRot = load_and_transform3d(
            self.cfg, image, bbox, pos3d, camera, None, evaluation=self.evaluation, hflipUsage=False,
            seed=seed, rotMat=rootRot
        )
        assert not hFlip
        resizedImages = np.zeros((self.cfg.MODEL.RECEPTIVE_FIELD, self.cfg.MODEL.IMGSIZE[0], self.cfg.MODEL.IMGSIZE[1], 3),
                                 dtype=np.float32)
        resizedImages[self.cfg.MODEL.RECEPTIVE_FIELD // 2, :, :, :] = image
        for i in range(startFrame, endFrame + 1):
            if i == frame:
                continue
            resizedImages[forwardRepeatSize + i - startFrame, :, :, :] = self.load_image(camera,
                                                                                         bboxes[i - startFrame, :],
                                                                                         images[i - startFrame, :, :,
                                                                                         :], seed)

        if forwardRepeatSize > 0:
            resizedImages[:forwardRepeatSize, :, :, :] = resizedImages[forwardRepeatSize:forwardRepeatSize + 1, :, :, :]

        if backwardRepeatSize > 0:
            resizedImages[-backwardRepeatSize:, :, :, :] = resizedImages[-backwardRepeatSize - 1:-backwardRepeatSize, :,
                                                           :, :]


        mask = np.concatenate((posMask, markerMask), axis=0)
        joint3d_validity_mask[openSimJointPos.shape[0]:] = np.minimum(mask,
                                                                      joint3d_validity_mask[openSimJointPos.shape[0]:])

        coordinateMask = (coordinateMask == 0) * self.cfg.LOSS.ANGLE.MASKMINVALUE + (coordinateMask == 1) * 1.0
        # prevent from dividing by 0
        if np.sum(coordinateMask) == 0:
            coordinateMask[0] = 10 ** -5

        # prevent from dividing by 0
        if np.sum(joint3d_validity_mask) == 0:
            joint3d_validity_mask[0] = 1

        sample = {
            'images': resizedImages / 255, 'name': name, 'pose3d': pos3d,
            'pose2d': pos2d, "bbox": bbox, 'joint3d_validity_mask': joint3d_validity_mask,
            'boneScale': boneScale, 'coordinateAngle': coordinateAngle, 'coordinateMask': coordinateMask,
            'localIdx': localIdx, 'fileIdx': idx, 'subject': subjectID, "cameraType": cameraType,
            "videoID": videoID, "frame": frame, "rootRotMat": rootRot,
        }

        if self.transform:
            sample = self.transform(sample)

        sample["images"].requires_grad = False
        sample["pose3d"].requires_grad = False
        sample["pose2d"].requires_grad = False
        sample["joint3d_validity_mask"].requires_grad = False
        sample["boneScale"].requires_grad = False
        sample["coordinateAngle"].requires_grad = False
        sample["coordinateMask"].requires_grad = False

        return sample

    def __len__(self):

        return len(self.usedIndices)

    def load_image(self, camera, bbox, image, seed):

        # data augmentation
        image = load_and_transform3d(
            self.cfg, image, bbox, None, camera, None, evaluation=self.evaluation, hflipUsage=False,
            seed=seed
        )

        return image
