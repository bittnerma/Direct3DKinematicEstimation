import torch
import os
from pathlib import Path
import math
from tqdm import tqdm
import abc

class TorchTrainingProgram:
    '''
    Do not instantiate this class.
    '''

    def __init__(self, args, cfg):

        self.cfg = cfg

        if args.cpu:
            self.COMP_DEVICE = torch.device("cpu")
        else:
            self.COMP_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.modelFolder = cfg.MODEL_FOLDER if cfg.MODEL_FOLDER.endswith("/") else cfg.MODEL_FOLDER + "/"
        if not os.path.exists(self.modelFolder):
            Path(self.modelFolder).mkdir(parents=True, exist_ok=True)

        # dataset
        self.trainSet = None
        self.validationSet = None
        self.testSet = None
        self.trainLoader = None
        self.valLoader = None
        self.testLoader = None

        # loss
        self.numLoss = 0
        self.lossNames = []

        # model path
        self.bestValidLoss = math.inf
        self.bestModelPath = self.cfg.STARTMODELPATH
        self.optimizer = None
        self.metricLossIdx = [0]

        self.model = None

    def run(self):

        assert self.cfg.TRAINING.TRAINING_STEPS >= 1
        assert isinstance(self.cfg.TRAINING.EPOCH, list)
        assert len(self.cfg.TRAINING.EPOCH) == self.cfg.TRAINING.TRAINING_STEPS

        startEpoch = 0
        endEpoch = 0
        for i in range(self.cfg.TRAINING.TRAINING_STEPS):
            startEpoch += endEpoch
            endEpoch += self.cfg.TRAINING.EPOCH[i]
            self.train(startEpoch, endEpoch, i)

        print('Finished Training')

    def train(self, startEpoch, endEpoch, epochIdx):

        try:
            del self.optimizer
        except:
            pass

        # optimizer
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.initialize_training(startEpoch, endEpoch, epochIdx)
        assert self.optimizer is not None


        epoch = 0
        NUM_ITERATION = len(self.trainSet) // self.cfg.TRAINING.BATCHSIZE

        for epoch in range(startEpoch, endEpoch):

            self.model.train()
            self.swapLoss(evaluation=False)
            runningLoss = [0] * (self.numLoss + 1)

            try:
                del iterator
            except:
                pass

            if self.cfg.PROGRESSBAR:
                iterator = enumerate(tqdm(self.trainLoader))
            else:
                iterator = enumerate(self.trainLoader, 0)

            for _, inf in iterator:
                self.optimizer.zero_grad()
                losses = self.model_forward_and_calculate_loss(inf, evaluation=False)
                loss = losses[0]
                for a in range(1, len(losses)):
                    loss = loss + losses[a]
                loss.backward()
                self.optimizer.step()

                runningLoss[0] += loss.item()
                for i in range(self.numLoss):
                    value = losses[i]
                    if torch.is_tensor(value):
                        value = value.item()
                    runningLoss[i + 1] += value

            self.report(f'Train , Epoch {epoch} :', runningLoss, NUM_ITERATION)

            # validation set
            try:
                del iterator
            except:
                pass

            runningLoss = [0] * (self.numLoss + 1)
            self.model.eval()
            self.swapLoss(evaluation=True)
            numValidData = len(self.validationSet)
            with torch.no_grad():

                if self.cfg.PROGRESSBAR:
                    iterator = enumerate(tqdm(self.valLoader))
                else:
                    iterator = enumerate(self.valLoader, 0)

                for _, inf in iterator:
                    losses = self.model_forward_and_calculate_loss(inf, evaluation=True)
                    loss = losses[0]
                    for a in range(1, len(losses)):
                        loss = loss + losses[a]

                    runningLoss[0] += loss.item()
                    for i in range(self.numLoss):
                        value = losses[i]
                        if torch.is_tensor(value):
                            value = value.item()
                        runningLoss[i + 1] += value

            self.report(f'Valid, Epoch {epoch} :', runningLoss, len(self.validationSet))

            # update the learning rate
            self.update(epoch, startEpoch, endEpoch, epochIdx)

            currentValLoss = 0.0
            for idx in self.metricLossIdx:
                currentValLoss += runningLoss[idx]
            currentValLoss = currentValLoss / numValidData / len(self.metricLossIdx)

            # save the best model
            if self.bestValidLoss > currentValLoss:
                self.bestModelPath = self.save_model("best")
                self.bestValidLoss = currentValLoss
                print(f'Best Loss : {self.bestValidLoss}')

        # save the model in the last epoch
        _ = self.save_model(epoch)

    def swapLoss(self, evaluation=False):
        pass

    def report(self, dataset, runnigLoss, numData):
        s = ""
        s += f'{dataset} : '
        for i in range(len(self.lossNames)):
            s += f'{self.lossNames[i]} : {runnigLoss[i] / numData}   '

        print(s)

    def save_model(self, name):
        path = f'{self.modelFolder}model_{name}_{self.cfg.POSTFIX}.pt'
        torch.save(self.model.state_dict(), path)
        return path

    @abc.abstractmethod
    def model_forward_and_calculate_loss(self, inf, evaluation=False):
        pass

    @abc.abstractmethod
    def initialize_training(self, startEpoch, endEpoch, epochIdx):
        pass

    @abc.abstractmethod
    def update(self, epoch, startEpoch, endEpoch, epochIdx):
        # update after each iteration
        pass

    @abc.abstractmethod
    def create_model(self):
        pass

    @abc.abstractmethod
    def model_forward(self, inf, evaluation=False):
        pass

    @abc.abstractmethod
    def model_forward_and_calculate_loss(self, inf, evaluation=False):
        pass

    @abc.abstractmethod
    def store_every_frame_prediction(self, train=False, valid=False, test=True):
        pass

    @abc.abstractmethod
    def save_prediction(self):
        pass
