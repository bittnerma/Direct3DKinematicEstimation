import os
import h5py
import torch
import numpy as np
from ms_model_estimation.models.dataset.TorchDataset import TorchDataset
from ms_model_estimation.models.dataset.data_loading import load_and_transform3d
from ms_model_estimation.models.utils.BMLUtils import MIRROR_JOINTS
import collections


class BMLImgDataSet(TorchDataset):

    def __init__(
            self, cfg, h5pyBMLFolder, cameraParamter, dataseType, evaluation=True, useEveryFrame=False,
            usePredefinedSeed=False
    ):
        super(BMLImgDataSet, self).__init__(cfg, evaluation=evaluation)
        np.random.seed(self.cfg.SEED)

        self.evaluation = evaluation

        self.h5pyBMLFolder = h5pyBMLFolder
        self.useEveryFrame = useEveryFrame
        self.cameraParamter = cameraParamter
        self.usePredefinedSeed = usePredefinedSeed

        self.usedSubjects = []
        if dataseType == 0:
            self.usedSubjects = self.cfg.TRAINING_SUBJECTS
            # self.data = np.load(h5pyBMLFolder + "prediction_train.npy", allow_pickle=True).item()
        elif dataseType == 1:
            self.usedSubjects = self.cfg.VALID_SUBJECTS
            # self.data = np.load(h5pyBMLFolder + "prediction_valid.npy", allow_pickle=True).item()
        elif dataseType == 2:
            self.usedSubjects = self.cfg.TEST_SUBJECTS
        #HACK: added by Marian 
        elif dataseType == 3:
            self.usedSubjects = [11]
            # self.data = np.load(h5pyBMLFolder + "prediction_test.npy", allow_pickle=True).item()
        else:
            raise Exception("The type of the dataset are not defined")

        self.usedEachFrame = self.cfg.DATASET.TRAINING_USEDEACHFRAME if not self.evaluation else self.cfg.DATASET.USEDEACHFRAME
        self.create_mapping()
        # self.awaredOcclusion = AwaredOcclusion(cfg)

    def create_mapping(self):

        # create mapping to idx : [subjectID, cameraType, index of local hdf5]

        globalDataIdx = 0
        self.usedIndices = []
        self.videoTable = collections.defaultdict(list)
        for subject in self.usedSubjects[:1]:
            if os.path.exists(self.h5pyBMLFolder + f'subject_{subject}.hdf5'):
                with h5py.File(self.h5pyBMLFolder + f'subject_{subject}.hdf5', 'r') as f:
                    globalFrames = f["globalFrame"][:]
                    cameraTypes = f["cameraType"][:]
                    usedFrames = f['usedFrame'][:]
                    videoIndices = f['videoID'][:]
                for localIdx, globalFrame in enumerate(globalFrames):
                    self.usedIndices.append([(subject), cameraTypes[localIdx].decode(), localIdx, globalDataIdx])
                    self.videoTable[videoIndices[localIdx]].append(globalDataIdx)
                    globalDataIdx += 1

        self.usedIndices = self.usedIndices

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        subjectID, cameraType, localIdx, globalDataIdx = self.usedIndices[idx]

        with h5py.File(self.h5pyBMLFolder + f'subject_{subjectID}.hdf5', 'r') as f:
            bbox = f['bbox'][localIdx, :]
            globalFrame = f['globalFrame'][localIdx]

        # get camera
        camera = self.cameraParamter[cameraType].copy()

        with h5py.File(self.h5pyBMLFolder + f'{subjectID}_{cameraType}_img.hdf5', 'r') as f:
            image = f["images"][globalFrame, :, :, :][:, :, [2, 1, 0]]

        image = load_and_transform3d(
            self.cfg, image, bbox, None, camera, MIRROR_JOINTS, evaluation=True
        )

        sample = {
            'image': image / 255,
            "cameraType": cameraType,
            "globalDataIdx": globalDataIdx,
        }

        if self.transform:
            sample = self.transform(sample)

        sample["image"].requires_grad = False

        return sample

    def __len__(self):

        return len(self.usedIndices)
