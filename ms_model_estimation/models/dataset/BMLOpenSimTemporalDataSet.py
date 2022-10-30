import os
import h5py
import torch
import numpy as np
from ms_model_estimation.models.dataset.TorchDataset import TorchDataset


class BMLOpenSimTemporalDataSet(TorchDataset):

    def __init__(
            self, cfg, h5pyBMLFolder, dataFilePath, datasetType, evaluation=True, useEveryFrame=False,
    ):
        super(BMLOpenSimTemporalDataSet, self).__init__(cfg, evaluation=evaluation)
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

    def create_mapping(self):

        self.usedIndices = []
        globalDatasetIdx = 0
        for subject in self.usedSubjects:
            if os.path.exists(self.h5pyBMLFolder + f'subject_{subject}.hdf5'):
                with h5py.File(self.h5pyBMLFolder + f'subject_{subject}.hdf5', 'r') as f:
                    globalFrames = f["globalFrame"][:]
                    cameraTypes = f["cameraType"][:]
                    usedFrames = f['usedFrame'][:]

                for localIdx, globalFrame in enumerate(globalFrames):
                    if self.useEveryFrame or (globalFrame % self.usedEachFrame) == 0 and (
                            self.evaluation or usedFrames[localIdx]):
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

        predBoneScale = np.zeros((self.cfg.MODEL.RECEPTIVE_FIELD, len(self.cfg.PREDICTION.BODY), 3), dtype=np.float32)
        predPos = np.zeros((self.cfg.MODEL.RECEPTIVE_FIELD, self.cfg.PREDICTION.NUM_JOINTS, 3),
                           dtype=np.float32)
        predMarkerPos = np.zeros((self.cfg.MODEL.RECEPTIVE_FIELD, self.cfg.PREDICTION.NUM_MARKERS, 3), dtype=np.float32)
        predRootRot = np.zeros((self.cfg.MODEL.RECEPTIVE_FIELD, 4, 4), dtype=np.float32)
        predRot = np.zeros((self.cfg.MODEL.RECEPTIVE_FIELD, len(self.cfg.PREDICTION.COORDINATES)), dtype=np.float32)
        pose3d = np.zeros((self.cfg.MODEL.RECEPTIVE_FIELD, self.cfg.MODEL.NUMPRED, 3), dtype=np.float32)
        coordinateAngle = np.zeros((self.cfg.MODEL.RECEPTIVE_FIELD, len(self.cfg.PREDICTION.COORDINATES)),
                                   dtype=np.float32)
        coordinateMask = np.ones((self.cfg.MODEL.RECEPTIVE_FIELD, len(self.cfg.PREDICTION.COORDINATES)),
                                 dtype=np.float32)
        mask = np.ones((self.cfg.MODEL.RECEPTIVE_FIELD, self.cfg.MODEL.NUMPRED), dtype=np.float32)

        # for idxFrame, currentFrame in enumerate(range(startFrame, endFrame + 1)):
        # predPos[forwardRepeatSize + idxFrame, :, :] = self.data["predPos"][currentFrame - frame + globalDatasetIdx: endFrame + 1 - frame + globalDatasetIdx,
        #                                              :, :]
        collectedData = endFrame + 1 - startFrame
        assert endFrame>=startFrame
        predPos[forwardRepeatSize:forwardRepeatSize + collectedData, :, :] = self.data["predPos"][
                                                      startFrame - frame + globalDatasetIdx: endFrame + 1 - frame + globalDatasetIdx,
                                                      :, :]

        pose3d[forwardRepeatSize:forwardRepeatSize + collectedData, :, :] = self.data["pose3d"][
                                                     startFrame - frame + globalDatasetIdx: endFrame + 1 - frame + globalDatasetIdx,
                                                     :, :]
        predMarkerPos[forwardRepeatSize:forwardRepeatSize + collectedData, :, :] = self.data["predMarkerPos"][
                                                            startFrame - frame + globalDatasetIdx: endFrame + 1 - frame + globalDatasetIdx,
                                                            :, :]
        predBoneScale[forwardRepeatSize:forwardRepeatSize + collectedData, :, :] = self.data["predBoneScale"][
                                                            startFrame - frame + globalDatasetIdx: endFrame + 1 - frame + globalDatasetIdx,
                                                            :, :]
        predRootRot[forwardRepeatSize:forwardRepeatSize + collectedData, :3, :] = self.data["predRootRot"][
                                                           startFrame - frame + globalDatasetIdx: endFrame + 1 - frame + globalDatasetIdx,
                                                           :, :]
        predRot[forwardRepeatSize:forwardRepeatSize + collectedData, :] = self.data["predRot"][
                                                   startFrame - frame + globalDatasetIdx: endFrame + 1 - frame + globalDatasetIdx,
                                                   :]
        coordinateAngle[forwardRepeatSize:forwardRepeatSize + collectedData, :] = self.data["labelRot"][
                                                           startFrame - frame + globalDatasetIdx: endFrame + 1 - frame + globalDatasetIdx,
                                                           :]
        coordinateMask[forwardRepeatSize:forwardRepeatSize + collectedData, :] = self.data["coordinateMask"][
                                                          startFrame - frame + globalDatasetIdx: endFrame + 1 - frame + globalDatasetIdx,
                                                          :]
        mask[forwardRepeatSize:forwardRepeatSize + collectedData, :] = self.data["mask"][
                                                startFrame - frame + globalDatasetIdx: endFrame + 1 - frame + globalDatasetIdx,
                                                :]

        if forwardRepeatSize > 0:
            predPos[:forwardRepeatSize, :, :] = predPos[forwardRepeatSize:forwardRepeatSize + 1, :, :]
            pose3d[:forwardRepeatSize, :, :] = pose3d[forwardRepeatSize:forwardRepeatSize + 1, :, :]
            predMarkerPos[:forwardRepeatSize, :, :] = predMarkerPos[forwardRepeatSize:forwardRepeatSize + 1, :, :]
            predBoneScale[:forwardRepeatSize, :, :] = predBoneScale[forwardRepeatSize:forwardRepeatSize + 1, :, :]
            predRootRot[:forwardRepeatSize, :, :] = predRootRot[forwardRepeatSize:forwardRepeatSize + 1, :, :]
            predRot[:forwardRepeatSize, :] = predRot[forwardRepeatSize:forwardRepeatSize + 1, :]
            coordinateAngle[:forwardRepeatSize, :] = coordinateAngle[forwardRepeatSize:forwardRepeatSize + 1, :]
            coordinateMask[:forwardRepeatSize, :] = coordinateMask[forwardRepeatSize:forwardRepeatSize + 1, :]
            mask[:forwardRepeatSize, :] = mask[forwardRepeatSize:forwardRepeatSize + 1, :]

        if backwardRepeatSize > 0:
            predPos[-backwardRepeatSize:, :, :] = predPos[-backwardRepeatSize - 1:-backwardRepeatSize, :, :]
            pose3d[-backwardRepeatSize:, :, :] = pose3d[-backwardRepeatSize - 1:-backwardRepeatSize, :, :]
            predMarkerPos[-backwardRepeatSize:, :, :] = predMarkerPos[-backwardRepeatSize - 1:-backwardRepeatSize, :, :]
            predBoneScale[-backwardRepeatSize:, :, :] = predBoneScale[-backwardRepeatSize - 1:-backwardRepeatSize, :, :]
            predRootRot[-backwardRepeatSize:, :, :] = predRootRot[-backwardRepeatSize - 1:-backwardRepeatSize, :, :]
            predRot[-backwardRepeatSize:, :] = predRot[-backwardRepeatSize - 1:-backwardRepeatSize, :]
            coordinateAngle[-backwardRepeatSize:, :] = coordinateAngle[-backwardRepeatSize - 1:-backwardRepeatSize, :]
            coordinateMask[-backwardRepeatSize:, :] = coordinateMask[-backwardRepeatSize - 1:-backwardRepeatSize, :]
            mask[-backwardRepeatSize:, :] = mask[-backwardRepeatSize - 1:-backwardRepeatSize, :]

        boneScales = np.zeros((self.cfg.MODEL.RECEPTIVE_FIELD, len(self.cfg.PREDICTION.BODY), 3))
        boneScales[:, :, :] = self.data["labelBoneScale"][globalDatasetIdx:globalDatasetIdx + 1, :, :]
        # boneScale = self.data["labelBoneScale"][globalDatasetIdx, :, :]

        predMarkerPos = predMarkerPos - predPos[:, :1, :]
        predPos = predPos - predPos[:, :1, :]
        pose3d = pose3d - pose3d[:, :1, :]
        samples = {

            # input
            "predPos": predPos,
            "predMarkerPos": predMarkerPos,
            "predBoneScale": predBoneScale,
            "predRootRot": predRootRot[:, :3, :3],
            "predRot": predRot,

            # label
            "pose3d": pose3d,
            "boneScale": boneScales,  # np.expand_dims(boneScale, axis=0),
            "fileIdx": idx,
            "globalDatasetIdx": globalDatasetIdx,
            "coordinateAngle": coordinateAngle,

            # mask
            "coordinateMask": coordinateMask,
            "mask": mask,
        }

        if self.transform:
            samples = self.transform(samples)

        return samples

    def __len__(self):

        return len(self.usedIndices)
