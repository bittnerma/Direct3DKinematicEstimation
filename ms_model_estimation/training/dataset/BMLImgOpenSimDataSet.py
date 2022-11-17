import os
import h5py
import torch
import numpy as np
from ms_model_estimation.training.dataset.TorchDataset import TorchDataset
from ms_model_estimation.training.dataset.data_loading import load_and_transform3d


class BMLImgOpenSimDataSet(TorchDataset):

    def __init__(
            self, cfg, h5pyBMLFolder, cameraParameter, datasetType, evaluation=True, useEveryFrame=False,
            usePredefinedSeed=False
    ):
        super(BMLImgOpenSimDataSet, self).__init__(cfg, evaluation=evaluation)
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
          #HACK: added by Marian 
        elif datasetType == 3:
            self.usedSubjects = [11]      
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
            if not self.cfg.TRAINING.USE_OS_SIMULATION or self.datasetType != 0:
                pos3d = f['pos3d'][localIdx, :self.cfg.PREDICTION.NUM_SMPL_JOINTS, :]
                marker_pose3d = f['marker_pos3d'][localIdx,
                                :(self.cfg.PREDICTION.NUM_MARKERS - self.cfg.PREDICTION.NUM_SMPL_JOINTS), :]
            else:
                pos3d = f['sim_pos3d'][localIdx, :self.cfg.PREDICTION.NUM_SMPL_JOINTS, :]
                marker_pose3d = f['sim_marker_pos3d'][localIdx,
                                :(self.cfg.PREDICTION.NUM_MARKERS - self.cfg.PREDICTION.NUM_SMPL_JOINTS), :]

            posMask = f['pos3d_mask'][localIdx, :self.cfg.PREDICTION.NUM_JOINTS]
            markerMask = f['marker_pos3d_mask'][localIdx,
                         :(self.cfg.PREDICTION.NUM_MARKERS - self.cfg.PREDICTION.NUM_SMPL_JOINTS)]

            bbox = f['bbox'][localIdx, :]
            globalFrame = f['globalFrame'][localIdx]
            hflipUsage = f['hflipUsage'][localIdx]
            videoID = int(f['videoID'][localIdx])
            frame = int(f['frame'][localIdx])
            seed = f['seed'][localIdx]
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
            image = f["images"][globalFrame, :, :, :][:, :, [2, 1, 0]]

        if self.usePredefinedSeed:
            # data augmentation
            image, pos3d, pos2d, joint3d_validity_mask, _, hFlip, rootRot = load_and_transform3d(
                self.cfg, image, bbox, pos3d, camera, None, evaluation=self.evaluation, hflipUsage=hflipUsage,
                seed=seed, rotMat=rootRot
            )
        else:
            image, pos3d, pos2d, joint3d_validity_mask, _, hFlip, rootRot = load_and_transform3d(
                self.cfg, image, bbox, pos3d, camera, None, evaluation=self.evaluation, rotMat=rootRot
            )
        assert not hFlip

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

        ''' 
        if self.cfg.TRAINING.AWARED_OCCLUSION.USE:
            occulsionMask = self.awaredOcclusion.measure_occlusion(pos3d)
            joint3d_validity_mask = np.minimum(joint3d_validity_mask, occulsionMask)'''

        sample = {
            'image': image / 255, 'name': name, 'pose3d': pos3d,
            'pose2d': pos2d, "bbox": bbox, 'joint3d_validity_mask': joint3d_validity_mask,
            'boneScale': boneScale, 'coordinateAngle': coordinateAngle, 'coordinateMask': coordinateMask,
            'localIdx': localIdx, 'fileIdx': idx, 'subject': subjectID, "cameraType": cameraType,
            "videoID": videoID, "frame": frame, "rootRotMat": rootRot,
        }

        if self.transform:
            sample = self.transform(sample)

        sample["pose3d"].requires_grad = False
        sample["pose2d"].requires_grad = False
        sample["joint3d_validity_mask"].requires_grad = False
        sample["boneScale"].requires_grad = False
        sample["coordinateAngle"].requires_grad = False
        sample["coordinateMask"].requires_grad = False

        return sample

    def __len__(self):

        return len(self.usedIndices)
