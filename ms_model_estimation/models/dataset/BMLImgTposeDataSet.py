import os
import h5py
import torch
import numpy as np
from ms_model_estimation.models.dataset.TorchDataset import TorchDataset
from ms_model_estimation.models.dataset.data_loading import load_and_transform3d


class BMLImgTposeDataSet(TorchDataset):

    def __init__(
            self, cfg, h5pyBMLFolder, cameraParamter
    ):
        super(BMLImgTposeDataSet, self).__init__(cfg, evaluation=True)
        np.random.seed(self.cfg.SEED)

        self.evaluation = True
        self.cfg = cfg
        self.h5pyBMLFolder = h5pyBMLFolder
        self.cameraParamter = cameraParamter
        bboxInf = self.cfg.BML_FOLDER + "all_bbox.npy"
        self.bboxInf = np.load(bboxInf, allow_pickle=True).item()

        self.__create_mapping()

    def __create_mapping(self):

        # create mapping to idx : [subjectID, cameraType, index of local hdf5]

        self.usedIndices = []
        for i in range(1, 91):
            # No T pose
            if i == 1 and i == 12:
                continue
            for idxCamera, cameraType in enumerate(["PG1", "PG2"]):
                path = self.h5pyBMLFolder + f'{i}_{cameraType}_img.hdf5'
                if os.path.exists(path) and (i in self.bboxInf) and (cameraType in self.bboxInf[i]):
                    self.usedIndices.append([i, cameraType, path])

        self.usedIndices = self.usedIndices

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        subjectID, cameraType, path = self.usedIndices[idx]

        with h5py.File(path, 'r') as f:
            image = f["images"][0, :, :, :][:, :, [2, 1, 0]]

        if cameraType == "PG1":
            cameraIdx = 0
        elif cameraType == "PG2":
            cameraIdx = 1
        else:
            assert False

        bbox = self.bboxInf[subjectID][cameraType][0, :]
        image = load_and_transform3d(
            self.cfg, image, bbox, None, self.cameraParamter[cameraType], None, evaluation=True
        )

        sample = {
            'image': image / 255, 'subjectID': subjectID, 'cameraIdx': cameraIdx
        }

        if self.transform:
            sample = self.transform(sample)

        sample["image"].requires_grad = False

        return sample

    def __len__(self):

        return len(self.usedIndices)
