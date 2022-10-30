import torch
import torch.nn as nn
import torchvision
from ms_model_estimation.models.spatial.ResNet_CenterStriding import resnet50 as resnet50_center_striding
from ms_model_estimation.models.spatial.ResNet_CenterStriding import resnext50_32x4d as resnext50_32x4d_center_striding

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

        means = torch.Tensor([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        stds = torch.Tensor([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
        self.register_buffer('means', means)
        self.register_buffer('stds', stds)

    def forward(self, x):
        assert x.shape[1] == 3
        x = x - self.means
        x = x / self.stds
        return x

class ResNet50(nn.Module):

    def __init__(
            self, next=False, pretrained=True, fullyConv=True, centerStriding=True
    ):
        super(ResNet50, self).__init__()

        shownString = "Petrained" if pretrained else "NOT Pretrained"

        if not next and centerStriding:
            model = resnet50_center_striding(pretrained=pretrained, progress=True)
            print(f'Use ResNet50 with center striding and {shownString} model')
        elif not next and not centerStriding:
            model = torchvision.models.resnet50(pretrained=pretrained, progress=True)
            print(f'Use ResNet50 without center striding and {shownString} model')
        elif next and centerStriding:
            model = resnext50_32x4d_center_striding(pretrained=pretrained, progress=True)
            print(f'Use ResNet50 Next with center striding and {shownString} model')
        elif next and not centerStriding:
            model = torchvision.models.resnext50_32x4d(pretrained=pretrained, progress=True)
            print(f'Use ResNet50 Next without center striding and {shownString} model')
        else:
            raise Exception("Model is not defined.")

        if fullyConv:
            self.resnet50 = torch.nn.Sequential(*(list(model.children())[:-2]))
        else:
            # the model has the adaptive pooling
            self.resnet50 = torch.nn.Sequential(*(list(model.children())[:-1]))

        self.normalizeLayer = Normalize()

    def forward(self, x):

        x = self.normalizeLayer(x)
        return self.resnet50(x)
