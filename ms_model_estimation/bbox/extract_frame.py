import argparse
from glob import glob
import cv2
import h5py
import numpy as np
import multiprocessing as mp


def extract_frame_from_video(
        videoPath, outputFolder
):
    inf = videoPath.split("/")[-1]
    subjectID = int(inf.split("_")[-2])

    cap = cv2.VideoCapture(videoPath)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cameraType = videoPath.split("/")[-1].split("_")[1]
    assert fps > 0
    path = outputFolder + f'{subjectID}_{cameraType}_img.hdf5'
    with h5py.File(path, 'w') as f:
        nameD = f.create_dataset(f'name', (1,), dtype='S150', compression="gzip",
                                 compression_opts=9)
        nameD[:] = inf.encode()

        frameD = f.create_dataset(f'numFrames', (1,), dtype='i8', compression="gzip",
                                  compression_opts=9)
        frameD[:] = numFrames

        imgDataset = f.create_dataset(f'images', (numFrames, 600, 800, 3), dtype=np.uint8, compression="gzip",
                                      compression_opts=9, chunks=(1, 600, 800, 3))

        for i in range(numFrames):
            ret, frame = cap.read()
            assert ret
            imgDataset[i, :, :, :] = frame

    print(f'{path} is processed.')


def search(videoFolder):
    videos = glob(videoFolder + "*.avi")
    videos = [v.replace("\\", "/") for v in videos]
    return videos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('outputFolder', action='store', type=str, default='P',
                        help="Hdf5 folder")
    parser.add_argument('videoFolder', action='store', type=str, default='P',
                        help="videoFolder")
    parser.add_argument('--cores', action='store', type=int, default=0,
                        help="cores of cpu")
    args = parser.parse_args()

    videos = search(args.videoFolder)
    if args.cores > 0:
        print(f'{args.cores} cores')
        pool = mp.Pool(args.cores)
    else:
        print(f'{mp.cpu_count()} cores')
        pool = mp.Pool(mp.cpu_count())

    results = pool.starmap(extract_frame_from_video, [(videoPath, args.outputFolder) for videoPath in videos])

    pool.close()
