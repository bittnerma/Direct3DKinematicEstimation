import argparse
import random
from ms_model_estimation.models.config.config_bml_temporal import get_cfg_defaults
import numpy as np
from tqdm import tqdm
import torch
from ms_model_estimation.models.dataset.BMLImgDataSet import BMLImgDataSet
from ms_model_estimation.models.PoseEstimationModel import PoseEstimationModel
from torch.utils.data import DataLoader
from ms_model_estimation.models.BMLUtils import CAMERA_TABLE
from ms_model_estimation.models.camera.cameralib import Camera


class Prediction:

    def __init__(self, args, cfg):

        self.cfg = cfg

        if args.cpu:
            self.COMP_DEVICE = torch.device("cpu")
        else:
            self.COMP_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # create camera parameter
        cameraParamter = {}
        for cameraType in CAMERA_TABLE:
            cameraInf = CAMERA_TABLE[cameraType]
            R = cameraInf["extrinsic"][:3, :3]  # .T
            t = np.matmul(cameraInf["extrinsic"][:3, -1:].T, R) * -1
            distortion_coeffs = np.array(
                [cameraInf["radialDisortionCoeff"][0], cameraInf["radialDisortionCoeff"][1], 0, 0, 0], np.float32)
            intrinsic_matrix = cameraInf["intrinsic"].copy()
            camera = Camera(t, R, intrinsic_matrix, distortion_coeffs)
            cameraParamter[cameraType] = camera
        self.cameraParamter = cameraParamter

        # Dataset
        self.h5pyFolder = cfg.BML_FOLDER if cfg.BML_FOLDER.endswith("/") else cfg.BML_FOLDER + "/"

    def pose_estiomation_model_forward(self, dataset, datasetLoader, name):

        print(f'{name} : {len(dataset)} data.')

        if self.cfg.PROGRESSBAR:
            iterator = enumerate(tqdm(datasetLoader))
        else:
            iterator = enumerate(datasetLoader, 0)

        predictions = np.empty((len(dataset), self.cfg.TRAINING.NUMPRED, 3))
        labels = np.empty((len(dataset), self.cfg.TRAINING.NUMPRED, 3))
        masks = np.empty((len(dataset), self.cfg.TRAINING.NUMPRED))

        with torch.no_grad():
            for _, inf in iterator:
                image = inf["image"].to(self.COMP_DEVICE)
                pose3d = inf["pose3d"]
                fileIdx = inf["fileIdx"].cpu().detach().numpy()
                mask = inf["joint3d_validity_mask"].cpu().detach().numpy()
                # B, S, J, _ = pose3d.shape
                # image = image.view(B * S, 3, image.shape[-2], image.shape[-1])
                predPos = self.model(image)
                # predPos = predPos.view(B, S, J, 3)
                pose3d = pose3d - pose3d[:, :1, :]
                predPos = predPos - predPos[:, :1, :]
                predictions[fileIdx, :, :] = predPos.cpu().detach().numpy()
                labels[fileIdx, :, :] = pose3d.cpu().detach().numpy()
                masks[fileIdx, :] = mask

        outputs = {
            "prediction": predictions,
            "label": labels,
            "masks": masks,
        }

        erros = np.mean(np.sum((predictions - labels) ** 2, axis=-1) ** 0.5)
        print(f'{name} errors : {erros} mm')
        np.save(self.cfg.BML_FOLDER + f'pose_estimation_prediction_{name}.npy', outputs)

    def save_sequence_data(self):
        self.model = PoseEstimationModel(self.cfg).to(self.COMP_DEVICE)
        self.model.load_state_dict(torch.load(self.cfg.PRETRAINED_POSE_MODEL))
        self.model.eval()

        dataset = BMLImgDataSet(self.cfg, self.h5pyFolder, self.cameraParamter, 0, evaluation=False, useEveryFrame=True,
                                usePredefinedSeed=True)
        datasetLoader = DataLoader(dataset, batch_size=self.cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                   drop_last=False,
                                   num_workers=self.cfg.EVAL_WORKERS)
        self.pose_estiomation_model_forward(dataset, datasetLoader, "augmented_train")

        dataset = BMLImgDataSet(self.cfg, self.h5pyFolder, self.cameraParamter, 1, evaluation=True, useEveryFrame=True)
        datasetLoader = DataLoader(dataset, batch_size=self.cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                   drop_last=False,
                                   num_workers=self.cfg.EVAL_WORKERS)
        self.pose_estiomation_model_forward(dataset, datasetLoader, "valid")

        dataset = BMLImgDataSet(self.cfg, self.h5pyFolder, self.cameraParamter, 2, evaluation=True, useEveryFrame=True)
        datasetLoader = DataLoader(dataset, batch_size=self.cfg.TRAINING.EVALUATION_BATCHSIZE, shuffle=False,
                                   drop_last=False,
                                   num_workers=self.cfg.EVAL_WORKERS)
        self.pose_estiomation_model_forward(dataset, datasetLoader, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cpu', action='store_true', default=False, help="only use cpu?")
    parser.add_argument('--ymlFile', action='store',
                        default="", type=str,
                        help="The hdf5 folder")

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    if args.ymlFile:
        cfg.merge_from_file(args.ymlFile)
    cfg.freeze()
    print(cfg)

    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    prediction = Prediction(args, cfg)
    prediction.save_sequence_data()
