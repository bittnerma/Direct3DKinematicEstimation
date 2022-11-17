import os
import multiprocessing as mp
import h5py
import torch
import numpy as np
from ms_model_estimation.training.dataset.TorchDataset import TorchDataset
from ms_model_estimation.training.dataset.data_loading import load_and_transform3d
from ms_model_estimation.training.utils.BMLUtils import MIRROR_JOINTS
import multiprocessing


class BMLImgDataSet(TorchDataset):

    def __init__(
            self, cfg, h5pyBMLFolder, cameraParamter, dataseType, numWorkers, evaluation=True, useEveryFrame=False,
    ):

        '''
        
        :param cfg: 
        :param h5pyBMLFolder: 
        :param cameraParamter: 
        :param dataseType: 0 : train , 1 : valid , 2 : test 
        :param evaluation: 
        :param useEveryFrame: 
        '''
        super(BMLImgDataSet, self).__init__(cfg, evaluation=evaluation)

        np.random.seed(self.cfg.SEED)
        self.evaluation = evaluation

        self.h5pyBMLFolder = h5pyBMLFolder
        self.useEveryFrame = useEveryFrame
        self.cameraParamter = cameraParamter

        self.usedSubjects = []
        if dataseType == 0:
            self.usedSubjects = self.cfg.TRAINING_SUBJECTS
        elif dataseType == 1:
            self.usedSubjects = self.cfg.VALID_SUBJECTS
        elif dataseType == 2:
            self.usedSubjects = self.cfg.TEST_SUBJECTS
        else:
            raise Exception("The type of the dataset are not defined")

        self.usedEachFrame = self.cfg.DATASET.TRAINING_USEDEACHFRAME if not self.evaluation or dataseType == 1 else self.cfg.DATASET.USEDEACHFRAME

        self.create_mapping()
        self.numWorkers = numWorkers

    def create_mapping(self):

        # create mapping to idx : [subjectID, cameraType, index of local hdf5]
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
                            [subject, cameraTypes[localIdx].decode(), int(localIdx), int(globalDatasetIdx)]
                        )
                    globalDatasetIdx += 1

        self.usedIndices = self.usedIndices

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        subjectID, cameraType, localIdx, globalDatasetIdx = self.usedIndices[idx]

        with h5py.File(self.h5pyBMLFolder + f'subject_{subjectID}.hdf5', 'r') as f:
            videoStartFrame = int(f["globalStartFrame"][localIdx])
            frame = int(f["globalFrame"][localIdx])
            videoEndFrame = int(f["globalEndFrame"][localIdx])

        if self.evaluation:
            hflipUsage = False
        else:
            hflipUsage = np.random.uniform(0, 1) >= 0.5
        seed = np.random.randint(0, 2 ** 16)

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

        images = np.empty((
            self.cfg.MODEL.RECEPTIVE_FIELD, self.cfg.MODEL.IMGSIZE[0], self.cfg.MODEL.IMGSIZE[0], 3),
            dtype=np.float32)
        pose3ds = np.empty(
            (self.cfg.MODEL.RECEPTIVE_FIELD, self.cfg.PREDICTION.NUMPRED, 3),
            dtype=np.float32)
        pose2ds = np.empty(
            (self.cfg.MODEL.RECEPTIVE_FIELD, self.cfg.PREDICTION.NUMPRED, 2),
            dtype=np.float32)
        masks = np.empty(
            (self.cfg.MODEL.RECEPTIVE_FIELD, self.cfg.PREDICTION.NUMPRED),
            dtype=np.int)

        # parallel processing
        '''
        localIndices = [localIdx + frameIdx - frame for frameIdx in range(startFrame, endFrame + 1)]
        pool = multiprocessing.Pool(self.numWorkers)
        results = pool.starmap(self.load_data, [(subjectID, cameraType, tmpIdx, hflipUsage, seed) for tmpIdx in localIndices])
        pool.close()'''

        for idx, frameIdx in enumerate(range(startFrame, endFrame + 1)):
            image, pos3d, pos2d, joint3d_validity_mask = self.load_data(subjectID, cameraType,
                                                                        localIdx + frameIdx - frame, hflipUsage,
                                                                        seed=seed)
            '''
            images[forwardRepeatSize + idx, :, :, :] = results[idx][0]
            pose3ds[forwardRepeatSize + idx, :, :] = results[idx][1]
            pose2ds[forwardRepeatSize + idx, :, :] = results[idx][2]
            masks[forwardRepeatSize + idx, :] = results[idx][3]'''

            images[forwardRepeatSize + idx, :, :, :] = image
            pose3ds[forwardRepeatSize + idx, :, :] = pos3d
            pose2ds[forwardRepeatSize + idx, :, :] = pos2d
            masks[forwardRepeatSize + idx, :] = joint3d_validity_mask

        if forwardRepeatSize > 0:
            images[:forwardRepeatSize, :, :, :] = images[forwardRepeatSize:forwardRepeatSize + 1, :, :, :]
            pose3ds[:forwardRepeatSize, :, :] = pose3ds[forwardRepeatSize:forwardRepeatSize + 1, :, :]
            pose2ds[:forwardRepeatSize, :, :] = pose2ds[forwardRepeatSize:forwardRepeatSize + 1, :, :]
            masks[:forwardRepeatSize, :] = masks[forwardRepeatSize:forwardRepeatSize + 1, :]

        if backwardRepeatSize > 0:
            images[-backwardRepeatSize:, :, :, :] = images[-backwardRepeatSize - 1:-backwardRepeatSize, :, :, :]
            pose3ds[-backwardRepeatSize:, :, :] = pose3ds[-backwardRepeatSize - 1:-backwardRepeatSize, :, :]
            pose2ds[-backwardRepeatSize:, :, :] = pose2ds[-backwardRepeatSize - 1:-backwardRepeatSize, :, :]
            masks[-backwardRepeatSize:, :] = masks[-backwardRepeatSize - 1:-backwardRepeatSize, :]

        pose3ds = pose3ds - pose3ds[:, :1, :]

        samples = {
            "images": images / 255,
            "pose3ds": pose3ds,
            "pose2ds": pose2ds,
            "masks": masks,
        }

        samples = self.transform(samples)

        samples["images"].requires_grad_(False)
        samples["pose3ds"].requires_grad_(False)
        samples["pose2ds"].requires_grad_(False)
        samples["masks"].requires_grad_(False)

        return samples

    def load_data(self, subjectID, cameraType, localIdx, hflipUsage, seed):

        with h5py.File(self.h5pyBMLFolder + f'subject_{subjectID}.hdf5', 'r') as f:
            pos3d = f['pos3d'][localIdx, :self.cfg.PREDICTION.NUM_JOINTS, :]
            marker_pose3d = f['marker_pos3d'][localIdx, :, :]
            bbox = f['bbox'][localIdx, :]
            globalFrame = f['globalFrame'][localIdx]

        # concatenate the joint and the marker
        pos3d = np.concatenate((pos3d, marker_pose3d), axis=0)

        # get camera
        camera = self.cameraParamter[cameraType].copy()

        with h5py.File(self.h5pyBMLFolder + f'{subjectID}_{cameraType}_img.hdf5', 'r') as f:
            image = f["images"][globalFrame, :, :, :]

        # data augmentation
        image, pos3d, pos2d, joint3d_validity_mask, _, _ = load_and_transform3d(
            self.cfg, image, bbox, pos3d, camera, MIRROR_JOINTS, evaluation=self.evaluation, hflipUsage=hflipUsage,
            seed=seed
        )

        return image, pos3d, pos2d, joint3d_validity_mask

    def __len__(self):

        return len(self.usedIndices)
