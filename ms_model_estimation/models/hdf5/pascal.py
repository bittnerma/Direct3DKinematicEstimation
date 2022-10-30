from tqdm import tqdm
import glob
import xml
import xml.etree.ElementTree
import cv2
import numpy as np
import h5py
import functools
from PIL import Image
from pathlib import Path


def load_occluders(outputFolder, pascal_root):
    outputFolder = outputFolder if outputFolder.endswith("/") else outputFolder + "/"
    Path(outputFolder).mkdir(parents=True, exist_ok=True)

    currentIdx = 0
    with h5py.File(outputFolder + "pascal.hdf5", 'w') as f:

        for annotation_path in tqdm(glob.glob(f'{pascal_root}/Annotations/*.xml')):
            xml_root = xml.etree.ElementTree.parse(annotation_path).getroot()
            is_segmented = (xml_root.find('segmented').text != '0')

            if not is_segmented:
                continue

            boxes = []
            for i_obj, obj in enumerate(xml_root.findall('object')):
                is_person = (obj.find('name').text == 'person')
                is_difficult = (obj.find('difficult').text != '0')
                is_truncated = (obj.find('truncated').text != '0')
                if not is_person and not is_difficult and not is_truncated:
                    bndbox = obj.find('bndbox')
                    box = [int(bndbox.find(s).text) for s in ['xmin', 'ymin', 'xmax', 'ymax']]
                    boxes.append((i_obj, box))

            if not boxes:
                continue

            image_filename = xml_root.find('filename').text
            segmentation_filename = image_filename.replace('jpg', 'png')

            path = f'{pascal_root}/JPEGImages/{image_filename}'
            seg_path = f'{pascal_root}/SegmentationObject/{segmentation_filename}'

            im = cv2.imread(path)
            labels = np.asarray(Image.open(seg_path))

            for i_obj, (xmin, ymin, xmax, ymax) in boxes:
                object_mask = (labels[ymin:ymax, xmin:xmax] == i_obj + 1).astype(np.uint8)
                object_image = im[ymin:ymax, xmin:xmax]
                # Ignore small objects
                if cv2.countNonZero(object_mask) < 500:
                    continue

                object_mask = soften_mask(object_mask)
                downscale_factor = 0.5
                object_image = resize_by_factor(object_image, downscale_factor)
                object_mask = resize_by_factor(object_mask, downscale_factor)

                object_imageD = f.create_dataset(f'{currentIdx}_image', (object_image.shape), dtype=np.uint8,
                                                 compression="gzip",
                                                 compression_opts=9)
                object_imageD[:, :, :] = object_image

                object_maskD = f.create_dataset(f'{currentIdx}_mask', (object_mask.shape), dtype=np.float32,
                                                compression="gzip",
                                                compression_opts=9)
                object_maskD[:, :] = object_mask
                currentIdx += 1

        numDataD = f.create_dataset("numData", (1,), dtype=np.int)
        numDataD[0] = currentIdx


def soften_mask(mask):
    morph_elem = get_structuring_element(cv2.MORPH_ELLIPSE, (8, 8))
    eroded = cv2.erode(mask, morph_elem)
    result = mask.astype(np.float32)
    result[eroded < result] = 0.75
    return result


def resize_by_factor(im, factor, interp=None):
    """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
    for downscaling.
    """
    new_size = rounded_int_tuple([im.shape[1] * factor, im.shape[0] * factor])
    if interp is None:
        interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)


def rounded_int_tuple(p):
    return tuple(np.round(p).astype(int))


@functools.lru_cache()
def get_structuring_element(shape, ksize, anchor=None):
    if not isinstance(ksize, tuple):
        ksize = (ksize, ksize)
    return cv2.getStructuringElement(shape, ksize, anchor)
