import random
import cv2
import numpy as np
from ms_model_estimation.models.camera.cameralib import Camera, reproject_image, reproject_image_points
from ms_model_estimation.models.dataset.apperance.apperance import augment_appearance


def load_and_transform3d(cfg, origsize_im, box, world_coords, camera, mirrorJointIdx, evaluation=True, valid=False,
                         hflipUsage=False, seed=None , rotMat=None):
    geom_rng = new_rng(seed)
    partial_visi_rng = new_rng(seed)
    appearance_rng = new_rng(None)

    output_side = cfg.MODEL.IMGSIZE[0]
    output_imshape = cfg.MODEL.IMGSIZE

    '''
    if not evaluation and cfg.DATASET.BBOX.PARTIAL_VISUALBILITY:
        box = expand_to_square(box)
        box = random_partial_subbox(box, partial_visi_rng)'''


    crop_side = np.max(box[2:])
    center_point = center(box)
    if not evaluation and cfg.DATASET.GEOM.AUG:
        center_point += random_uniform_disc(geom_rng) * cfg.DATASET.GEOM.SHIFT_AUG / 100 * crop_side

    if box[2] < box[3]:
        delta_y = np.array([0, box[3] / 2])
        sidepoints = center_point + np.stack([-delta_y, delta_y])
    else:
        delta_x = np.array([box[2] / 2, 0])
        sidepoints = center_point + np.stack([-delta_x, delta_x])

    cam = camera.copy()
    cam.turn_towards(target_image_point=center_point)
    cam.undistort()
    cam.square_pixels()
    world_sidepoints = camera.image_to_world(sidepoints)
    cam_sidepoints = cam.world_to_image(world_sidepoints)
    crop_side = np.linalg.norm(cam_sidepoints[0] - cam_sidepoints[1])
    cam.zoom(output_side / crop_side)
    cam.center_principal_point(output_imshape)

    if cfg.DATASET.GEOM.AUG and not evaluation:
        s1 = cfg.DATASET.GEOM.SCALE_DOWN / 100
        s2 = cfg.DATASET.GEOM.SCALE_UP / 100
        r = cfg.DATASET.GEOM.ROT * np.pi / 180
        zoom = geom_rng.uniform(1 - s1, 1 + s2)
        cam.zoom(zoom)
        cam.rotate(roll=geom_rng.uniform(-r, r))

    if world_coords is not None:
        world_coords = world_coords
        metric_world_coords = world_coords

        if cfg.DATASET.GEOM.HFLIP and not evaluation and (geom_rng.rand() < 0.5 or hflipUsage):
            hFlip = True
            cam.horizontal_flip()
            camcoords = cam.world_to_camera(world_coords)[mirrorJointIdx, :]
            metric_world_coords = metric_world_coords[mirrorJointIdx, :]
        else:
            hFlip = False
            camcoords = cam.world_to_camera(world_coords)

        imcoords = cam.world_to_image(metric_world_coords)

        if rotMat is not None:
            rotMat = cam.rotMat_to_camera(rotMat)

    else:
        camcoords = None
        imcoords = None

    interp_str = cfg.DATASET.INTERPOLATION.TRAIN if (not evaluation or valid) else cfg.DATASET.INTERPOLATION.TEST
    antialias = cfg.DATASET.ANTIALIAS.TRAIN if (not evaluation or valid) else cfg.DATASET.ANTIALIAS.TEST

    interp = getattr(cv2, 'INTER_' + interp_str.upper())

    if origsize_im is not None:

        image_aug = reproject_image(
            origsize_im, camera, cam, output_imshape, antialias_factor=antialias, interp=interp)
        if not evaluation:
            image_aug = augment_appearance(cfg, image_aug, appearance_rng, evaluation=evaluation)
        image_aug = np.float32(image_aug)
    else:
        image_aug = None

    if world_coords is not None:
        joint3d_validity_mask = ~np.logical_or(np.any(imcoords < 0, axis=-1),
                                               np.any(imcoords >= output_side, axis=-1))
        joint2d_validity_mask = ~np.any(np.isnan(camcoords), axis=-1)

        camcoords = np.nan_to_num(camcoords)
        imcoords = np.nan_to_num(imcoords)

        if rotMat is not None:
            return image_aug, np.float32(camcoords), np.float32(
                imcoords), joint3d_validity_mask, joint2d_validity_mask, hFlip, rotMat
        else:
            return image_aug, np.float32(camcoords), np.float32(
                imcoords), joint3d_validity_mask, joint2d_validity_mask, hFlip
    else:
        return image_aug


def load_and_transform2d(cfg, im_from_file, bbox, coords, mirrorJointIdx, evaluation=True):
    # Get the random number generators for the different augmentations to make it reproducibile
    geom_rng = new_rng(None)
    partial_visi_rng = new_rng(None)
    appearance_rng = new_rng(None)

    # Determine bounding box
    if not evaluation and cfg.DATASET.BBOX.PARTIAL_VISUALBILITY:
        bbox = expand_to_square(bbox)
        bbox = random_partial_subbox(bbox, partial_visi_rng)

    crop_side = np.max(bbox)
    center_point = center(bbox)
    orig_cam = Camera.create2D(im_from_file.shape)
    cam = orig_cam.copy()
    cam.zoom(cfg.MODEL.IMGSIZE[0] / crop_side)

    if cfg.DATASET.GEOM.AUG and not evaluation:
        center_point += random_uniform_disc(geom_rng) * cfg.DATASET.GEOM.SHIFT_AUG / 100 * crop_side
        s1 = cfg.DATASET.GEOM.SCALE_DOWN / 100
        s2 = cfg.DATASET.GEOM.SCALE_UP / 100
        r = cfg.DATASET.GEOM.ROT * np.pi / 180
        zoom = geom_rng.uniform(1 - s1, 1 + s2)
        cam.zoom(zoom)
        cam.rotate(roll=geom_rng.uniform(-r, r))

    if cfg.DATASET.GEOM.HFLIP and not evaluation and geom_rng.rand() < 0.5:
        hFlip = True
        # Horizontal flipping
        cam.horizontal_flip()
        # Must also permute the joints to exchange e.g. left wrist and right wrist!
        imcoords = coords[mirrorJointIdx, :]
    else:
        imcoords = coords
        hFlip = False

    new_center_point = reproject_image_points(center_point, orig_cam, cam)
    cam.shift_to_center(new_center_point, cfg.MODEL.IMGSIZE)

    # is_annotation_invalid = (np.nan_to_num(imcoords[:, 1]) > im_from_file.shape[0] * 0.95)
    # imcoords[is_annotation_invalid] = np.nan
    imcoords = reproject_image_points(imcoords, orig_cam, cam)

    interp_str = cfg.DATASET.INTERPOLATION.TRAIN if not evaluation else cfg.DATASET.INTERPOLATION.TEST
    antialias = cfg.DATASET.ANTIALIAS.TRAIN if not evaluation else cfg.DATASET.ANTIALIAS.TEST
    interp = getattr(cv2, 'INTER_' + interp_str.upper())
    image_aug = reproject_image(
        im_from_file, orig_cam, cam, cfg.MODEL.IMGSIZE, antialias_factor=antialias, interp=interp)

    if not evaluation:
        image_aug = augment_appearance(cfg, image_aug, appearance_rng, evaluation=evaluation)

    is_joint_in_fov = ~np.logical_or(np.any(imcoords < 0, axis=-1),
                                     np.any(imcoords >= cfg.MODEL.IMGSIZE[0], axis=-1))
    joint2d_validity_mask = ~np.any(np.isnan(imcoords), axis=-1)

    joint2d_validity_mask = joint2d_validity_mask & is_joint_in_fov

    imcoords = np.nan_to_num(imcoords)



    return np.float32(image_aug), np.float32(imcoords), joint2d_validity_mask, hFlip


def random_partial_box(random_state):
    def generate():
        x1 = random_state.uniform(0, 0.5)
        x2, y2 = random_state.uniform(0.5, 1, size=2)
        side = x2 - x1
        if not 0.5 < side < y2:
            return None
        return np.array([x1, y2 - side, side, side])

    while True:
        box = generate()
        if box is not None:
            return box


def random_partial_subbox(box, random_state):
    subbox = random_partial_box(random_state)
    topleft = box[:2] + subbox[:2] * box[2:]
    size = subbox[2:] * box[2:]
    return np.concatenate([topleft, size])


def random_uniform_disc(rng):
    """Samples a random 2D point from the unit disc with a uniform distribution."""
    angle = rng.uniform(-np.pi, np.pi)
    radius = np.sqrt(rng.uniform(0, 1))
    return radius * np.array([np.cos(angle), np.sin(angle)])


def random_partial_box(random_state):
    def generate():
        x1 = random_state.uniform(0, 0.5)
        x2, y2 = random_state.uniform(0.5, 1, size=2)
        side = x2 - x1
        if not 0.5 < side < y2:
            return None
        return np.array([x1, y2 - side, side, side])

    while True:
        box = generate()
        if box is not None:
            return box


def expand_to_square(box):
    center_point = center(box)
    side = np.max(box[2:])
    return np.array([center_point[0] - side / 2, center_point[1] - side / 2, side, side])


def center(box):
    return box[:2] + box[2:] / 2


def new_rng(rng):
    if rng is not None:
        return np.random.RandomState(rng)
    else:
        return np.random.RandomState(random.randint(0, 2 ** 16))
