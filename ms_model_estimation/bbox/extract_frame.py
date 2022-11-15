import argparse
from glob import glob
import cv2
import h5py
import numpy as np
import multiprocessing as mp
from tqdm import tqdm


def extract_frame_from_video(
        videoPath, outputFolder
):
    _,cameraType,_,subjectID,_ = videoPath.stem.split("_")

    cap = cv2.VideoCapture(str(videoPath))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    assert fps > 0
    path = outputFolder / f'{subjectID}_{cameraType}_img.hdf5'
    with h5py.File(path, 'w') as f:
        nameD = f.create_dataset(f'name', (1,), dtype='S150', compression="gzip",
                                 compression_opts=9)
        nameD[:] = videoPath.name.encode()

        frameD = f.create_dataset(f'numFrames', (1,), dtype='i8', compression="gzip",
                                  compression_opts=9)
        frameD[:] = numFrames

        imgDataset = f.create_dataset(f'images', (numFrames, 600, 800, 3), dtype=np.uint8, compression="gzip",
                                      compression_opts=9, chunks=(1, 600, 800, 3))
        pbar = tqdm(range(numFrames))
        pbar.set_description(f"{videoPath.stem}")
        for i in pbar:
            
            ret, frame = cap.read()
            assert ret
            imgDataset[i, :, :, :] = frame
            # print(f"\r Processing frame {i} | {numFrames}")

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
