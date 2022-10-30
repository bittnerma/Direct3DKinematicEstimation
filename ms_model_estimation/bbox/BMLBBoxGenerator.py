import argparse
import h5py
from ms_model_estimation.bbox.BBoxGenerator import BBoxGenerator
import numpy as np
from glob import glob
from tqdm import tqdm


class BMLBBoxGenerator(BBoxGenerator):

    def __init__(self, hdf5Folder):
        super(BMLBBoxGenerator, self).__init__()
        self.hdf5Folder = hdf5Folder if hdf5Folder.endswith("/") else hdf5Folder + "/"

    def generate_bbox_from_video(
            self, path):

        with h5py.File(path, 'r') as f:

            numFrames = f["numFrames"][0]
            name = f["name"][0]
            bboxes = np.zeros((numFrames, 4))
            for i in range(numFrames):
                frame = f["images"][i, :, :, :]

                frame, offset = BBoxGenerator.check_ImgSize(frame)

                results = self.generate_bbox(frame, offset, single=True,
                                             returnImg=False)

                if len(results) == 0:
                    print(f'The {i}-th frame of {name} does not find person.')
                    if i > 0:
                        bboxes[i, :] = bboxes[i - 1, :].copy()
                else:
                    bboxes[i, :] = results[0]

        path = path.replace("_img.hdf5", "_bbox.npy")
        np.save(path, bboxes)

    def generate(self):

        files = glob(self.hdf5Folder + "*_*_img.hdf5")
        files = [f.replace("\\", "/") for f in files]
        return files

    def coolect_all_bbox(self):

        files = glob(self.hdf5Folder + "*_bbox.npy")
        files = [f.replace("\\", "/") for f in files]

        bboxTable = {}
        for file in files:
            subject = int(file.split("/")[-1].split("_")[0])
            cameraType = file.split("/")[-1].split("_")[1]
            if subject not in bboxTable:
                bboxTable[subject] = {}
            bboxTable[subject][cameraType] = np.load(file)

        np.save(self.hdf5Folder + "all_bbox.npy", bboxTable)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('hdf5Folder', action='store', type=str, default='P',
                        help="Hdf5 folder")
    args = parser.parse_args()
    generator = BMLBBoxGenerator(args.hdf5Folder)
    files = generator.generate()
    for f in tqdm(files):
        generator.generate_bbox_from_video(f)
    generator.coolect_all_bbox()