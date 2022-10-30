import math

import torch
import torchvision
from torchvision import transforms
import numpy as np


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.f = torchvision.transforms.Resize(self.output_size)

    def __call__(self, sample):
        _, H, W = sample["image"].shape
        assert H == W
        scale = self.output_size[0] / H
        sample["image"] = self.f(sample["image"])
        if "pose2d" in sample:
            # means = torch.mean(sample["pose2d"], dim=0, keepdim=True)
            # sample["pose2d"] -= means
            # sample["pose2d"] *= scale
            # sample["pose2d"] += means
            # sample["pose2d"] = sample["pose2d"]*scale
            pass
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for key, val in sample.items():

            if key == "image":
                val = val.transpose((2, 0, 1))
            elif key == "images":
                assert len(val.shape) == 4
                val = val.transpose((0, 3, 1, 2))

            if isinstance(val, list):
                sample[key] = torch.Tensor(val).float()
            elif isinstance(val, np.ndarray):
                sample[key] = torch.from_numpy(val).float()
            elif isinstance(val, str):
                continue
            elif isinstance(val, int):
                continue
            else:
                raise Exception(f'{type(val)}  is not defined in ToTensor.')
        return sample


class Zoom(object):
    def __init__(self, zoomProb=0.5, zoomRatio=0.4):

        # zoom (random cropping) probability
        self.zoomProb = 1 - zoomProb

        # zoom ratio
        self.zoomRatio = 1 - zoomRatio

    def __call__(self, sample):
        if np.random.uniform(0, 1, 1)[0] >= self.zoomProb:
            image = sample["image"]
            H, W = image.shape[1], image.shape[2]

            if np.random.uniform(0, 1, 1)[0] >= self.zoomProb:
                zoomRatio = np.random.uniform(self.zoomRatio, 1, 1)[0]
                # zoom the image
                newH = int(H * zoomRatio)
                newW = int(W * zoomRatio)

                sample["image"] = torchvision.transforms.functional.center_crop(image, (newH, newW))

        return sample


class Occlusion(object):
    def __init__(self, occlusionProb=0.7, rectangleSizeRatio=0.4):
        self.occlusionProb = 1 - occlusionProb
        self.rectangleSizeRatio = rectangleSizeRatio

    def __call__(self, sample):
        if np.random.uniform(0, 1, 1)[0] >= self.occlusionProb:
            image = sample["image"]
            pose2d = sample["pose2d"].detach().numpy()
            # distance = np.mean(np.sqrt(np.sum((pose2d - pose2d[:1, :]) ** 2 , axis=-1)))
            usedJoints = np.where(np.sum(pose2d >= 0, axis=-1))
            usedIdx = usedJoints[0][np.random.randint(usedJoints[0].shape[0])]

            H, W = image.shape[1], image.shape[2]

            center = pose2d[usedIdx, :]
            center[0] = max(center[0], 10)
            center[1] = max(center[1], 10)

            cropSizeH = int(self.rectangleSizeRatio * H)
            cropSizeW = int(self.rectangleSizeRatio * W)

            topH = int(max(0, center[1] - cropSizeH // 2))
            leftW = int(max(0, center[0] - cropSizeW // 2))
            buttomH = int(min(H - 1, center[1] + cropSizeH // 2))
            rightW = int(min(W - 1, center[0] + cropSizeW // 2))

            if topH < buttomH and leftW < rightW:
                noise = torch.rand((1, (buttomH - topH), (rightW - leftW)))
                image[0:1, topH: buttomH, leftW: rightW] = noise
                image[1:2, topH: buttomH, leftW: rightW] = noise
                image[2:, topH: buttomH, leftW: rightW] = noise

            '''
            # add white nose between 0 and 1
            cropSizeH = int(self.rectangleSizeRatio * H)
            cropSizeW = int(self.rectangleSizeRatio * W)

            startH, endH = cropSizeH, H - cropSizeH - 1
            startW, endW = max(cropSizeW, W // 5), min(W // (5 / 4), W - cropSizeW - 1)

            centerH = np.random.randint(startH, endH, 1)[0]
            centerW = np.random.randint(startW, endW, 1)[0]

            image[:, centerH - cropSizeH: centerH + cropSizeH,
            centerW - cropSizeW: centerW + cropSizeW] = torch.rand(
                image[:, centerH - cropSizeH: centerH + cropSizeH,
                centerW - cropSizeW: centerW + cropSizeW].shape)'''

            sample["image"] = image

        return sample


class ColorJitter(object):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2):
        self.colorjitter = torchvision.transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                                              saturation=saturation, hue=hue)

    def __call__(self, sample):
        sample["image"] = self.colorjitter(sample["image"])

        return sample


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        if 'image' in sample:
            sample['image'] = transforms.functional.normalize(sample['image'], self.mean, self.std)
        return sample


class HorizontalFlip(object):

    def __init__(self, horizontalProb):

        self.horizontalProb = 1 - horizontalProb

    def __call__(self, sample):

        if np.random.uniform(0, 1, 1)[0] >= self.horizontalProb:
            sample['image'] = torchvision.transforms.functional.hflip(sample['image'])

            if "pose2d" in sample:
                pose2d = sample['pose2d']
                pose2d[:, 0] *= -1
                sample['pose2d'] = pose2d

            if "pose3d" in sample:
                pose3d = sample['pose3d']
                pose3d[:, 0] *= -1
                sample['pose3d'] = pose3d

        return sample


class Rotation(object):

    def __init__(self, rotationProb, rotationDegree):
        self.rotationProb = 1 - rotationProb
        self.rotationDegree = rotationDegree

    def __call__(self, sample):
        if np.random.uniform(0, 1, 1)[0] >= self.rotationProb:

            degree = np.random.uniform(-self.rotationDegree, self.rotationDegree, 1)[0]
            sample['image'] = torchvision.transforms.functional.affine(sample["image"], degree, [0, 0], 1, 0)

            R = torch.eye(3)
            R[0, 0] = np.cos(degree / 180 * math.pi)
            R[1, 1] = np.cos(degree / 180 * math.pi)
            R[0, 1] = - np.sin(degree / 180 * math.pi)
            R[1, 0] = np.sin(degree / 180 * math.pi)

            # TODO:
            if "pose2d" in sample:
                sample['pose2d'] = torch.einsum('ij,kj ->ki', R[:2, :2], sample['pose2d'])

            if "pose3d" in sample:
                sample['pose3d'] = torch.einsum('ij,kj ->ki', R, sample['pose3d'])

        return sample


class Translation(object):

    def __init__(self, translationProb, translationRatio):
        self.translationProb = 1 - translationProb
        self.translationRatio = translationRatio

    def __call__(self, sample):
        if np.random.uniform(0, 1, 1)[0] >= self.translationProb:
            image = sample["image"]
            W = image.shape[2]

            tW = np.random.uniform(0, self.translationRatio * (W // 2), 1)[0]
            sample['image'] = torchvision.transforms.functional.affine(image, 0, [tW, 0], 1, 0)

        return sample
