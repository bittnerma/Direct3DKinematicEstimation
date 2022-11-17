import math
import os
import h5py
import torch
import numpy as np
from ms_model_estimation.training.dataset.TorchDataset import TorchDataset
from transforms3d.euler import euler2mat
from ms_model_estimation.training.utils.BMLUtils import MIRROR_JOINTS


class BMLImgDataSet(TorchDataset):

    def __init__(
            self, cfg, h5pyBMLFolder, dataFilePath, datasetType, evaluation=True, useEveryFrame=False,
    ):
        super(BMLImgDataSet, self).__init__(cfg, evaluation=evaluation)
        np.random.seed(self.cfg.SEED)
        self.evaluation = evaluation

        self.h5pyBMLFolder = h5pyBMLFolder
        self.useEveryFrame = useEveryFrame

        self.usedSubjects = []
        if datasetType == 0:
            self.usedSubjects = self.cfg.TRAINING_SUBJECTS
        elif datasetType == 1:
            self.usedSubjects = self.cfg.VALID_SUBJECTS
        elif datasetType == 2:
            self.usedSubjects = self.cfg.TEST_SUBJECTS
        else:
            raise Exception("The type of the dataset are not defined")

        self.usedEachFrame = self.cfg.DATASET.TRAINING_USEDEACHFRAME if not self.evaluation else self.cfg.DATASET.USEDEACHFRAME
        self.data = np.load(dataFilePath, allow_pickle=True).item()
        self.create_mapping()
        '''
        if self.cfg.MODEL.AWARED_OCCLUSION.USE:
            self.awaredOcclusion = AwaredOcclusion(cfg)
        '''

    def create_mapping(self):

        # create mapping to idx : [subjectID, cameraType, index of local hdf5 , globalDatasetIdx]
        self.usedIndices = []
        globalDatasetIdx = 0
        for subject in self.usedSubjects:
            if os.path.exists(self.h5pyBMLFolder + f'subject_{subject}.hdf5'):
                with h5py.File(self.h5pyBMLFolder + f'subject_{subject}.hdf5', 'r') as f:
                    globalFrames = f["globalFrame"][:]
                    cameraTypes = f["cameraType"][:]
                    usedFrames = f['usedFrame'][:]
                    videoStartFrame = f["globalStartFrame"][:]
                    videoEndFrame = f["globalEndFrame"][:]

                for localIdx, globalFrame in enumerate(globalFrames):
                    if self.useEveryFrame or (globalFrame % self.usedEachFrame) == 0 and (
                            self.evaluation or usedFrames[localIdx]):
                        '''
                        if not self.evaluation and self.cfg.MODEL.CAUSAL and (
                                globalFrame - videoStartFrame[localIdx]) < self.cfg.MODEL.RECEPTIVE_FIELD:
                            continue

                        if not self.evaluation and not self.cfg.MODEL.CAUSAL and (
                                (globalFrame - videoStartFrame[localIdx]) < self.cfg.MODEL.RECEPTIVE_FIELD // 2 or (
                                videoEndFrame[localIdx] - globalFrame) < self.cfg.MODEL.RECEPTIVE_FIELD // 2
                        ):
                            continue'''

                        self.usedIndices.append(
                            [subject, cameraTypes[localIdx].decode(), int(localIdx), int(globalDatasetIdx)])
                    globalDatasetIdx += 1

        self.usedIndices = self.usedIndices

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        subjectID, cameraType, localIdx, globalDatasetIdx = self.usedIndices[idx]

        with h5py.File(self.h5pyBMLFolder + f'subject_{subjectID}.hdf5', 'r') as f:
            videoStartFrame = int(f["startFrame"][localIdx])
            frame = int(f["globalFrame"][localIdx])
            videoEndFrame = int(f["globalEndFrame"][localIdx])

        backwardRepeatSize = 0
        forwardRepeatSize = 0
        if not self.cfg.MODEL.CAUSAL:

            # the frame in the middle is the target for prediction
            endFrame = frame + self.cfg.MODEL.RECEPTIVE_FIELD // 2
            if endFrame > videoEndFrame:
                backwardRepeatSize = endFrame - videoEndFrame
                endFrame = videoEndFrame

            startFrame = frame - self.cfg.MODEL.RECEPTIVE_FIELD // 2 + 1 * (self.cfg.MODEL.RECEPTIVE_FIELD % 2 == 0)
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

        if not self.cfg.MODEL.USE_GT:
            pos3ds = np.empty((self.cfg.MODEL.RECEPTIVE_FIELD, self.cfg.PREDICTION.NUMPRED, 3))

        '''
        if self.cfg.MODEL.AWARED_OCCLUSION.USE and self.cfg.MODEL.AWARED_OCCLUSION.USE_PREDICTION:
            occulsionMask = np.empty((self.cfg.MODEL.RECEPTIVE_FIELD, self.cfg.PREDICTION.NUMPRED))
        else:
            occulsionMask = np.ones((self.cfg.MODEL.RECEPTIVE_FIELD, self.cfg.PREDICTION.NUMPRED))'''

        labels = np.empty((self.cfg.MODEL.RECEPTIVE_FIELD, self.cfg.PREDICTION.NUMPRED, 3))
        masks = np.empty((self.cfg.MODEL.RECEPTIVE_FIELD, self.cfg.PREDICTION.NUMPRED))

        for idxFrame, currentFrame in enumerate(range(startFrame, endFrame + 1)):

            if not self.cfg.MODEL.USE_GT:
                pos3ds[forwardRepeatSize + idxFrame, :, :] = self.data["prediction"][
                                                             currentFrame - frame + globalDatasetIdx,
                                                             :, :]

            '''
            if self.cfg.MODEL.AWARED_OCCLUSION.USE and self.cfg.MODEL.AWARED_OCCLUSION.USE_PREDICTION:
                occulsionMask[forwardRepeatSize + idxFrame, :] = self.data["predictionOcclusion"][
                                                                 startFrame + idxFrame - frame + globalDatasetIdx,
                                                                 :]'''

            labels[forwardRepeatSize + idxFrame, :, :] = self.data["label"][
                                                         currentFrame - frame + globalDatasetIdx,
                                                         :, :]
            masks[forwardRepeatSize + idxFrame, :] = self.data["masks"][
                                                     currentFrame - frame + globalDatasetIdx,
                                                     :]
        if forwardRepeatSize > 0:
            if not self.cfg.MODEL.USE_GT:
                pos3ds[:forwardRepeatSize, :, :] = pos3ds[forwardRepeatSize:forwardRepeatSize + 1, :, :]
            '''
            if self.cfg.MODEL.AWARED_OCCLUSION.USE and self.cfg.MODEL.AWARED_OCCLUSION.USE_PREDICTION:
                occulsionMask[:forwardRepeatSize, :] = occulsionMask[forwardRepeatSize:forwardRepeatSize + 1, :]'''

            labels[:forwardRepeatSize, :, :] = labels[forwardRepeatSize:forwardRepeatSize + 1, :, :]
            masks[:forwardRepeatSize, :] = masks[forwardRepeatSize:forwardRepeatSize + 1, :]

        if backwardRepeatSize > 0:
            if not self.cfg.MODEL.USE_GT:
                pos3ds[-backwardRepeatSize:, :, :] = pos3ds[-backwardRepeatSize - 1:-backwardRepeatSize, :, :]
            '''    
            if self.cfg.MODEL.AWARED_OCCLUSION.USE and self.cfg.MODEL.AWARED_OCCLUSION.USE_PREDICTION:
                occulsionMask[-backwardRepeatSize:, :] = occulsionMask[-backwardRepeatSize - 1:-backwardRepeatSize, :]'''

            labels[-backwardRepeatSize:, :, :] = labels[-backwardRepeatSize - 1:-backwardRepeatSize, :, :]
            masks[-backwardRepeatSize:, :] = masks[-backwardRepeatSize - 1:-backwardRepeatSize, :]

        if self.cfg.MODEL.USE_GT:
            pos3ds = labels.copy()

        pos3ds = pos3ds - pos3ds[:, :1, :]
        labels = labels - labels[:, :1, :]

        # masks for occlusion
        ''' 
        usedSequenceIndex = self.cfg.MODEL.RECEPTIVE_FIELD // 2 if not self.cfg.MODEL.CAUSAL else -1
        if self.cfg.MODEL.AWARED_OCCLUSION.USE and self.cfg.MODEL.AWARED_OCCLUSION.USE_PREDICTION:
            occulsionMask = np.empty((self.cfg.MODEL.RECEPTIVE_FIELD, self.cfg.PREDICTION.NUMPRED))
            for i in range(self.cfg.MODEL.RECEPTIVE_FIELD):
                occulsionMask[i, :] = self.awaredOcclusion.measure_occlusion(pos3ds[i, :, :])

        elif self.cfg.MODEL.AWARED_OCCLUSION.USE and self.cfg.MODEL.AWARED_OCCLUSION.GT:
            occulsionMask = self.awaredOcclusion.measure_occlusion(labels[usedSequenceIndex, :, :])
            masks = np.minimum(masks, occulsionMask)
        elif self.cfg.MODEL.AWARED_OCCLUSION.USE and not self.cfg.MODEL.AWARED_OCCLUSION.GT:
            occulsionMask = self.awaredOcclusion.measure_occlusion(pos3ds[usedSequenceIndex, :, :])
            masks = np.minimum(masks, occulsionMask)
        
            
        if not self.cfg.MODEL.AWARED_OCCLUSION.USE_PREDICTION and self.cfg.MODEL.AWARED_OCCLUSION.USE and self.cfg.MODEL.AWARED_OCCLUSION.GT:

            occulsionMask = self.data["labelOcclusion"][globalDatasetIdx, :]
            masks = np.minimum(masks, occulsionMask)

        elif not self.cfg.MODEL.AWARED_OCCLUSION.USE_PREDICTION and self.cfg.MODEL.AWARED_OCCLUSION.USE and not self.cfg.MODEL.AWARED_OCCLUSION.GT:

            occulsionMask = self.data["predictionOcclusion"][globalDatasetIdx, :]
            masks = np.minimum(masks, occulsionMask)
        '''

        # data augmentation
        if not self.evaluation and self.cfg.DATASET.GEOM.AUG:

            pos3ds = pos3ds - pos3ds[:, :1, :]
            labels = labels - labels[:, :1, :]

            # horizontal flip
            if self.cfg.DATASET.GEOM.HFLIP and np.random.uniform(0, 1, 1)[0] > 0.5:
                pos3ds[:, :, 0] *= -1
                labels[:, :, 0] *= -1
                pos3ds = pos3ds[:, MIRROR_JOINTS, :]
                labels = labels[:, MIRROR_JOINTS, :]
                pos3ds = pos3ds - pos3ds[:, :1, :]
                labels = labels - labels[:, :1, :]

            # random rotation
            if np.random.uniform(0, 1, 1)[0] > (1 - self.cfg.DATASET.ROATION_PROB):
                rotMat = euler2mat(
                    np.random.uniform(-self.cfg.DATASET.GEOM.ROT, self.cfg.DATASET.GEOM.ROT) / 180 * math.pi,
                    np.random.randn(),
                    np.random.uniform(-self.cfg.DATASET.GEOM.ROT, self.cfg.DATASET.GEOM.ROT) / 180 * math.pi,
                )
                pos3ds = np.einsum('ik,bjk -> bji', rotMat, pos3ds)
                labels = np.einsum('ik,bjk -> bji', rotMat, labels)

            # random scaling
            pos3ds = pos3ds - pos3ds[:, :1, :]
            labels = labels - labels[:, :1, :]
            if np.random.uniform(0, 1, 1)[0] > (1 - self.cfg.DATASET.SCALING_PROB):
                scale = np.random.uniform(0.5, 1.5, 1)[0]
                pos3ds = pos3ds * scale
                labels = labels * scale

            '''
            # random occlusion
            if np.random.uniform(0, 1, 1)[0] > (1 - self.cfg.DATASET.OCCLUSION.PROB):
                index = np.random.randint(1, pos3ds.shape[1], (pos3ds.shape[0],))
                for i in range(pos3ds.shape[0]):
                    if not self.cfg.DATASET.OCCLUSION.ZERO:
                        pos3ds[i, index[i], :] += np.random.uniform(-0.1, 0.1, (3,))
                    else:
                        pos3ds[i, index[i], :] = 0'''

            pos3ds = pos3ds - pos3ds[:, :1, :]
            labels = labels - labels[:, :1, :]

            if np.random.uniform(0, 1, 1)[0] > (1 - self.cfg.DATASET.MIX_PROB):
                pos3ds = labels.copy()
                masks[:, :] = 1
                # occulsionMask[:, :] = 1
        '''
        if self.cfg.MODEL.AWARED_OCCLUSION.USE and self.cfg.MODEL.AWARED_OCCLUSION.USE_PREDICTION:
            # reshape to 1 d
            pos3ds = np.reshape(pos3ds, (pos3ds.shape[0], -1))
            pos3ds = np.concatenate((pos3ds, occulsionMask), axis=1)

        el'''
        if self.cfg.MODEL.USE_2D:
            usedAxis = 2 if self.cfg.MODEL.USE_2D else 3
            pos3ds = pos3ds[:, :, :, :usedAxis]

        samples = {
            "pose3d": pos3ds,
            "mask": masks,
            "label": labels,
            "fileIdx": globalDatasetIdx,
            # "occulsionMask": occulsionMask,
        }

        if self.transform:
            samples = self.transform(samples)

        return samples

    def __len__(self):
        return len(self.usedIndices)
