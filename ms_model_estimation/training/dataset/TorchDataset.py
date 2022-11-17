import torch
from torchvision import transforms
from ms_model_estimation.training.dataset.DataAugmentation import ToTensor


class TorchDataset(torch.utils.data.Dataset):

    def __init__(
            self, cfg, evaluation=True
    ):
        super(TorchDataset, self).__init__()

        self.cfg = cfg
        self.evaluation = evaluation

        self.transform = transforms.Compose([
            ToTensor(),
        ])

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
