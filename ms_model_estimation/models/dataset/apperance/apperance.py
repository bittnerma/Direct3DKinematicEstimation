'''
The script is the source code for the paper, "MeTRAbs: Metric-Scale Truncation-Robust Heatmaps for Absolute 3D Human Pose
Estimation".
script url: https://github.com/isarandi/metrabs
paper url : https://arxiv.org/abs/2007.07227
'''
import h5py
import numpy as np
import ms_model_estimation.models.dataset.apperance.color as augmentation_color
import ms_model_estimation.models.dataset.apperance.voc_loader as voc_loader
from ms_model_estimation.models.dataset.apperance.utils import resize_by_factor, paste_over


def augment_appearance(cfg, im, rng, evaluation=False):
    occlusion_rng = new_rng(rng)
    color_rng = new_rng(rng)

    if not evaluation:
        if cfg.DATASET.OCCLUSION.PROB > 0:
            occlude_type = str(occlusion_rng.choice(['objects', 'random-erase']))
        else:
            occlude_type = None

        if occlude_type == 'objects':
            # For object occlusion augmentation, do the occlusion first, then the filtering,
            # so that the occluder blends into the image better.
            if occlusion_rng.uniform(0.0, 1.0) < cfg.DATASET.OCCLUSION.PROB:
                im = object_occlude(cfg, im, occlusion_rng, inplace=True)
            if cfg.DATASET.COLOR.AUG:
                im = augmentation_color.augment_color(im, color_rng)
        elif occlude_type == 'random-erase':
            # For random erasing, do color aug first, to keep the random block distributed
            # uniformly in 0-255, as in the Random Erasing paper
            if cfg.DATASET.COLOR.AUG:
                im = augmentation_color.augment_color(im, color_rng)
            if occlude_type and occlusion_rng.uniform(0.0, 1.0) < cfg.DATASET.OCCLUSION.PROB:
                im = random_erase(cfg, im, 0, 1 / 3, 0.3, 1.0 / 0.3, occlusion_rng, inplace=True)

    return im


def object_occlude(cfg, im, rng, inplace=True):
    # Following [Sárándi et al., arxiv:1808.09316, arxiv:1809.04987]
    factor = im.shape[0] / 256
    count = rng.randint(1, 8)
    #occluders = voc_loader.load_occluders(cfg.PASCAL_ROOT)

    for i in range(count):
        with h5py.File(cfg.PASCAL_PATH, 'r') as f:
            numData = f['numData'][0]
            currentIdx = rng.randint(0, numData)
            occluder = f[f'{currentIdx}_image'][:, :, :]
            occ_mask = f[f'{currentIdx}_mask'][:, :]

        # occluder, occ_mask = choice(occluders, rng)
        rescale_factor = rng.uniform(0.2, 1.0) * factor * cfg.DATASET.OCCLUSION.SCALE

        occ_mask = resize_by_factor(occ_mask, rescale_factor)
        occluder = resize_by_factor(occluder, rescale_factor)

        center = rng.uniform(0, im.shape[0], size=2)
        im = paste_over(occluder, im, alpha=occ_mask, center=center, inplace=inplace)

    return im


def random_erase(cfg, im, area_factor_low, area_factor_high, aspect_low, aspect_high, rng, inplace=True):
    # Following the random erasing paper [Zhong et al., arxiv:1708.04896]
    image_area = cfg.MODEL.IMGSIZE[0] ** 2
    while True:
        occluder_area = (
                rng.uniform(area_factor_low, area_factor_high) *
                image_area * cfg.DATASET.OCCLUSION.SCALE)
        aspect_ratio = rng.uniform(aspect_low, aspect_high)
        height = (occluder_area * aspect_ratio) ** 0.5
        width = (occluder_area / aspect_ratio) ** 0.5
        pt1 = rng.uniform(0, cfg.MODEL.IMGSIZE[0], size=2)
        pt2 = pt1 + np.array([width, height])
        if np.all(pt2 < cfg.MODEL.IMGSIZE[0]):
            pt1 = pt1.astype(int)
            pt2 = pt2.astype(int)
            if not inplace:
                im = im.copy()
            im[pt1[1]:pt2[1], pt1[0]:pt2[0]] = rng.randint(
                0, 255, size=(pt2[1] - pt1[1], pt2[0] - pt1[0], 3), dtype=im.dtype)
            return im


def new_rng(rng):
    if rng is not None:
        return np.random.RandomState(rng.randint(2 ** 16))
    else:
        return np.random.RandomState()


def choice(items, rng):
    return items[rng.randint(len(items))]
