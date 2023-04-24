import argparse
import h5py
from ms_model_estimation.data_preparation.BBoxGenerator import BBoxGenerator
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path


class BMLBBoxGenerator(BBoxGenerator):

    def __init__(self, hdf5Folder):
        super(BMLBBoxGenerator, self).__init__()
        self.hdf5Folder = hdf5Folder if hdf5Folder.endswith("/") else hdf5Folder + "/"
        # self.subfolder = Path(self.hdf5Folder + "bounding_boxes")
        # if not self.subfolder.exists():
        #     self.subfolder.mkdir(parents=True)

    def generate_bbox_from_video(
            self, path):        

        with h5py.File(path, 'r') as f:

            numFrames = f["numFrames"][0]
            name = f["name"][0]
            bboxes = np.zeros((numFrames, 4))
            for i in tqdm(range(numFrames)):
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

    def generate_batched_bbox(self, path, batchsize=1):

        with h5py.File(path, 'r') as f:

            numFrames = f["numFrames"][0]
            name = f["name"][0]
            bboxes = np.zeros((numFrames, 4))
            start_idx = 0
            offset = 0

            progress_bar = tqdm(total=numFrames)
            while start_idx < numFrames:
                end_idx = min(start_idx + batchsize, numFrames)
                # check frames
                frames = []
                for i in range(start_idx, end_idx):
                    frame, offset = BBoxGenerator.check_ImgSize(f["images"][i, :, :, :])
                    frames.append(frame)

                np_frames = np.array(frames)

                # batch frames
                batch_results = self.generate_batched_bboxes(np_frames, offset)
                for result in batch_results:
                    bboxes.append(result.copy())

                progress_bar.update(end_idx - start_idx)

        path = path.replace("_img.hdf5", "_bbox.npy")
        np.save(path, bboxes)

    def generate(self):

        files = glob(self.hdf5Folder + "*_*_img.hdf5")
        files = [f.replace("\\", "/") for f in files]
        return files

    def collect_all_bbox(self):

        files = glob(self.hdf5Folder + "*_bbox.npy")
        files = [f.replace("\\", "/") for f in files]

        bboxTable = {}
        for file in files:
            if "all" in file:
                continue
            subject = int(file.split("/")[-1].split("_")[0])
            cameraType = file.split("/")[-1].split("_")[1]
            if subject not in bboxTable:
                bboxTable[subject] = {}
            bboxTable[subject][cameraType] = np.load(file)

        np.save(self.hdf5Folder + "all_bbox.npy", bboxTable)

    def cleanup(self):
        assert(Path(self.hdf5Folder + "all_bbox.npy").exists())            
            
        for bbox_file in Path(self.hdf5Folder).glob("[0-9]*.npy"):
            if "all" in bbox_file:
                continue
            bbox_file.unlink()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('hdf5Folder', action='store', type=str, default='P',
                        help="Hdf5 folder")
    args = parser.parse_args()
    generator = BMLBBoxGenerator(args.hdf5Folder)
    files = generator.generate()
    for f in tqdm(files):
        generator.generate_bbox_from_video(f)
    generator.collect_all_bbox()