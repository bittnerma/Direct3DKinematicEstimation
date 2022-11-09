import argparse
import math
import sys
sys.path.append("/home/WTYANG/thesis/ms_model_estimation")
import torch
from ms_model_estimation.models.dataset import TorchDataset
from ms_model_estimation.models.PoseEstimationModel import PoseEstimationModel
from ms_model_estimation.models.loss.CustomLoss import CustomLoss
from ms_model_estimation import PredictedMarkers
from ms_model_estimation.smplh.smplh_vertex_index import smplHJoint
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path


class Training:

    def __init__(self, args):

        if args.cpu:
            self.COMP_DEVICE = torch.device("cpu")
        else:
            self.COMP_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.BATCH_SIZE = args.batchSize
        self.VALID_BATCH_SIZE = args.validBatchSize
        self.TEST_BATCH_SIZE = args.testBatchSize

        self.trainPath = args.training_h5py_path
        self.validPath = args.valid_h5py_path
        self.testPath = args.test_h5py_path
        self.trainImgPath = args.training_img_h5py_path
        self.validImgPath = args.valid_img_h5py_path
        self.testImgPath = args.test_img_h5py_path
        self.cameraParameterPath = args.cameraParameterPath

        self.predictedJoint = smplHJoint  # [:22]
        self.predictedMarker = PredictedMarkers

        self.epoch = args.epoch
        self.modelFolder = args.modelFolder
        if not os.path.exists(self.modelFolder):
            Path(self.modelFolder).mkdir(parents=True, exist_ok=True)

        # Dataset
        self.perspective_correction = args.perspective_correction
        self.projection2d = args.projection2d
        self.inputSize = args.inputSize
        self.trainSet = TorchDataset(
            args.inputSize, self.trainPath, self.trainImgPath, self.cameraParameterPath,
            fpsRatio=args.fpsRatio, evaluation=False, p=args.occlusionProb,
            zoomProb=args.zoomProb, cropRatio=args.cropRatio, rectangleSize=args.rectangleSize,
            perspective_correction=self.perspective_correction, projection2d=self.projection2d,
            numImage=2000, multipleImgDataset=True, maxIdx=6
        )

        self.validationSet = TorchDataset(
            args.inputSize, self.validPath, self.validImgPath, self.cameraParameterPath,
            fpsRatio=args.fpsRatio, evaluation=True,
            perspective_correction=self.perspective_correction, projection2d=self.projection2d,
            numImage=2000, multipleImgDataset=True, maxIdx=0
        )

        self.testSet = TorchDataset(
            args.inputSize, self.testPath, self.testImgPath, self.cameraParameterPath,
            fpsRatio=args.fpsRatio, evaluation=True,
            perspective_correction=self.perspective_correction, projection2d=self.projection2d,
            numImage=2000, multipleImgDataset=True, maxIdx=0
        )

        print("%d training data, %d valid data, %d test data" % (
            len(self.trainSet), len(self.validationSet), len(self.testSet)))

        self.trainLoader = DataLoader(
            self.trainSet, batch_size=self.BATCH_SIZE, shuffle=True
        )
        self.valLoader = DataLoader(self.validationSet, batch_size=self.VALID_BATCH_SIZE, shuffle=True, drop_last=False)
        self.testLoader = DataLoader(self.testSet, batch_size=self.TEST_BATCH_SIZE, shuffle=True, drop_last=False)

        # model
        self.predeict_marker = args.predeict_marker
        self.create_model()

        # optimizer
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.params, lr=args.lr)

        # learning rate schedule
        self.decayRate = args.decayRate
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        # loss
        self.numLoss = 2
        self.pose3d_mpjpe = CustomLoss.pose3d_mpjpe(root=False)
        self.lossNames = [
            "total loss",
            "3d pose loss",
            "3d marker loss",
        ]
        if self.projection2d:
            self.numLoss += 2
            self.lossNames.append("2d pose loss")
            self.lossNames.append("2d marker loss")

        # model path
        self.bestValidLoss = math.inf
        self.bestModelPath = None
        self.postfix = args.postfix

    def run(self):
        self.train()
        self.evaluate()

    def train(self):
        for epoch in range(self.epoch):

            self.model.train()
            with tqdm(self.trainLoader, unit="batch") as tepoch:

                for inf in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    self.optimizer.zero_grad()
                    losses = self.model_forward(inf, evaluation=False)
                    loss = losses[0]
                    for a in range(1, len(losses)):
                        loss = loss + losses[a]

                    loss.backward()
                    self.optimizer.step()

                    if self.projection2d:
                        tepoch.set_postfix(loss=loss.item(),
                                           pose3d_positionLoss=losses[0].item(),
                                           marker_positionLoss=losses[1].item(),
                                           pose2d_loss=losses[2].item(),
                                           marker2d_loss=losses[3].item(),
                                           )
                    else:
                        tepoch.set_postfix(loss=loss.item(),
                                           pose3d_positionLoss=losses[0].item(),
                                           marker_positionLoss=losses[1].item()
                                           )

            # validation set

            runningLoss = [0] * (self.numLoss + 1)

            self.model.eval()
            with torch.no_grad():
                for _, inf in enumerate(self.valLoader, 0):
                    losses = self.model_forward(inf, evaluation=True)
                    loss = losses[0]
                    for a in range(1, len(losses)):
                        loss = loss + losses[a]

                    runningLoss[0] += loss.item()
                    for i in range(self.numLoss):
                        runningLoss[i + 1] += losses[i].item()

            self.report("valid set", runningLoss, len(self.validationSet))

            # print the lr if lr changes
            self.lr_scheduler.step()
            if epoch % 1 == 0 and epoch != 0:
                print("Current LR: ", self.lr_scheduler.get_last_lr())

            # save the best model
            if self.bestValidLoss > (runningLoss[0] / len(self.validationSet)):
                self.bestModelPath = self.save_model(epoch)
                self.bestValidLoss = (runningLoss[0] / len(self.validationSet))

        # save the model in the last epoch
        _ = self.save_model(epoch)

        print('Finished Training')

    def save_model(self, epoch):
        path = f'{self.modelFolder}model_{epoch}_{self.postfix}.pt'
        torch.save(self.model.state_dict(), path)
        return path

    def evaluate(self, test=True, valid=False, train=False):

        # kill the model
        try:
            del self.model
            torch.cuda.empty_cache()
        except:
            pass

        # reload the model
        print("Load ", self.bestModelPath)
        self.create_model()
        self.model.load_state_dict(torch.load(self.bestModelPath))

        runningLoss = [0] * (self.numLoss + 1)

        self.model.eval()
        with torch.no_grad():
            if test:
                for _, inf in enumerate(self.testLoader, 0):

                    losses = self.model_forward(inf, evaluation=True)
                    loss = losses[0]
                    for a in range(1, len(losses)):
                        loss = loss + losses[a]

                    runningLoss[0] += loss.item()
                    for i in range(self.numLoss):
                        runningLoss[i + 1] += losses[i].item()

                self.report("test set", runningLoss, len(self.testSet))
            if valid:
                for _, inf in enumerate(self.valLoader, 0):

                    losses = self.model_forward(inf, evaluation=True)
                    loss = losses[0]
                    for a in range(1, len(losses)):
                        loss = loss + losses[a]

                    runningLoss[0] += loss.item()
                    for i in range(self.numLoss):
                        runningLoss[i + 1] += losses[i].item()

                self.report("valid set", runningLoss, len(self.validationSet))

            if train:
                for _, inf in enumerate(self.trainLoader, 0):

                    losses = self.model_forward(inf, evaluation=True)
                    loss = losses[0]
                    for a in range(1, len(losses)):
                        loss = loss + losses[a]

                    runningLoss[0] += loss.item()
                    for i in range(self.numLoss):
                        runningLoss[i + 1] += losses[i].item()

                self.report("train set", runningLoss, len(self.trainSet))

    def create_model(self):
        self.model = PoseEstimationModel(
            22,
            len(self.predictedMarker),
        ) \
            .float().to(self.COMP_DEVICE)
        print(f'Image size : {self.inputSize}')

    def report(self, dataset, runnigLoss, numData):
        s = ""
        s += f'{dataset} : '
        for i in range(len(self.lossNames)):
            s += f'{self.lossNames[i]} : {runnigLoss[i] / numData}   '

        print(s)

    def model_forward(self, inf, evaluation=False, prediction=False):

        # get input
        image = inf["image"].to(self.COMP_DEVICE)

        markerLoc = inf["markerLoc"].to(self.COMP_DEVICE)
        jointPos = inf["jointPos"].to(self.COMP_DEVICE)
        rootOffset = jointPos[:, 0:1, :].clone()
        # jointPos = jointPos - rootOffset
        # markerLoc = markerLoc - rootOffset

        # forward
        self.optimizer.zero_grad()
        predPos = self.model(image)

        # move the root to original point
        predPos = predPos - predPos[:, 0:1, :]
        if self.perspective_correction:
            perspective_correction_R = inf["perspective_correction"].to(self.COMP_DEVICE)
            predPos = torch.einsum('bik , bjk -> bji', perspective_correction_R, predPos)

        # aligned with the gt root
        predPos = predPos + rootOffset
        predMarkerPos = predPos[:, jointPos.shape[1]:, :]
        predPos = predPos[:, :jointPos.shape[1], :]

        # 3D joint position
        joint3dLossValue = self.pose3d_mpjpe(predPos, jointPos, evaluation=evaluation) * 10.0

        # 3D marker loss
        marker3dLossValue = self.pose3d_mpjpe(predMarkerPos, markerLoc, evaluation=evaluation) * 10.0

        losses = [joint3dLossValue, marker3dLossValue]

        if self.projection2d:
            intrinsic_camera_R = inf["intrinsic"].to(self.COMP_DEVICE)
            marker2d = inf["marker2d"].to(self.COMP_DEVICE)
            pose2d = inf["pose2d"].to(self.COMP_DEVICE)
            predPos2d = torch.einsum('bji , bik -> bjk', predPos / predPos[..., -1:], intrinsic_camera_R)
            predMarkerPos2d = torch.einsum('bji , bik -> bjk', predMarkerPos / predMarkerPos[..., -1:],
                                           intrinsic_camera_R)

            # 2D joint position
            joint2dLossValue = self.pose3d_mpjpe(predPos2d, pose2d, evaluation=evaluation) / 100

            # 2D marker loss
            marker2dLossValue = self.pose3d_mpjpe(predMarkerPos2d, marker2d, evaluation=evaluation) / 100

            losses.append(joint2dLossValue)
            losses.append(marker2dLossValue)

        if prediction:
            return predPos, losses
        else:
            return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--representation6D', action='store_true', default=False,
                        help="Represent 3D rotation with 6D vectors")
    parser.add_argument('--decayRate', action='store', type=float, default=0.96, help="The decay rate")
    parser.add_argument('--lr', action='store', type=float, default=0.0005, help="the learning rate")
    parser.add_argument('--epoch', action='store', type=int, default=25, help="The number of epoch")
    parser.add_argument('--inputSize', action='store', type=tuple, default=(256, 256),
                        help="The input size of the NN model")
    parser.add_argument('--numberBone', action='store', type=int, default=48,
                        help="The number of bone scale. The default value is 144(48*3)")
    parser.add_argument('--batchSize', action='store', type=int, default=64,
                        help="BatchSize for train set.")
    parser.add_argument('--validBatchSize', action='store', type=int, default=64,
                        help="BatchSize for validation set.")
    parser.add_argument('--testBatchSize', action='store', type=int, default=64,
                        help="BatchSize for test set.")
    parser.add_argument('--cpu', action='store_true', default=False, help="only use cpu?")
    parser.add_argument('--fpsRatio', action='store', type=int, default=4,
                        help="The ratio of the video frame rate to the "
                             "mocap frame rate. The default of BML dataset "
                             "is 4 since the vidoe frame rate is 30 fps "
                             "and the mocap frame rate is 120 fps.")
    # model
    parser.add_argument('--predeict_marker', action='store_false', default=True, help="predict marker location?")
    parser.add_argument('--projection2d', action='store_false', default=True, help="use 2d projection loss?")
    parser.add_argument('--perspective_correction', action='store_true', default=False, help="perspective correction?")

    # occlusion
    parser.add_argument('--occlusionProb', action='store', default=0.8, type=float,
                        help="occlusion probability")
    parser.add_argument('--zoomProb', action='store', default=0.8, type=float,
                        help="random crop probability")
    parser.add_argument('--cropRatio', action='store', default=0.3, type=float,
                        help="the ratio to the image width/height for random crop")
    parser.add_argument('--rectangleSize', action='store', default=60, type=int,
                        help="rectangular occlusion width/height for 224x224 image")

    # hdf5 path
    parser.add_argument('--postfix', action='store',
                        default="", type=str,
                        help="The postfix of the model path")
    parser.add_argument('--training_h5py_path', action='store',
                        default="/home/WTYANG/thesis/dataset/train.hdf5", type=str,
                        help="The pkl path for the datalist.")
    parser.add_argument('--valid_h5py_path', action='store', default="/home/WTYANG/thesis/dataset/valid.hdf5",
                        type=str,
                        help="The pkl path for the datalist.")
    parser.add_argument('--test_h5py_path', action='store', default="/home/WTYANG/thesis/dataset/test.hdf5",
                        type=str,
                        help="The pkl path for the datalist.")
    parser.add_argument('--training_img_h5py_path', action='store',
                        default="/home/WTYANG/thesis/dataset/train_img.hdf5", type=str,
                        help="The pkl path for the datalist.")
    parser.add_argument('--valid_img_h5py_path', action='store',
                        default="/home/WTYANG/thesis/dataset/valid_img.hdf5",
                        type=str,
                        help="The pkl path for the datalist.")
    parser.add_argument('--test_img_h5py_path', action='store',
                        default="/home/WTYANG/thesis/dataset/test_img.hdf5",
                        type=str,
                        help="The pkl path for the datalist.")
    parser.add_argument('--coordinateInfPath', action='store',
                        default="/home/WTYANG/thesis/dataset/coordinateAxis2.pkl",
                        type=str, help="The pkl path for the coordinate axis.")
    parser.add_argument('--cameraParameterPath', action='store',
                        default="/home/WTYANG/thesis/dataset/camera_setting.hdf5",
                        type=str, help="The hdf5 path for the camera parameter?")
    parser.add_argument('--modelFolder', action='store', type=str, default="/home/WTYANG/thesis/models/",
                        help="The folder path for saving the model.")

    args = parser.parse_args()

    trainingProgram = Training(args)
    trainingProgram.run()
