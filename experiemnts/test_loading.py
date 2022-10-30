from videoMuscle.models.dataset.H36MImgDataSet2 import H36MImgDataSet
from videoMuscle.models.config.config_h36m import get_cfg_defaults
from torch.utils.data import DataLoader
import time
import pickle

# NUM_CPU = multiprocessing.cpu_count()

if __name__ == "__main__":

    cfg = get_cfg_defaults()
    cfg.freeze()

    h5pyFolder = "/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/wyang/dataset/HDF5/H36M/"
    # h5pyFolder = "D:/mscWeitseYangLocal/dataset/HDF5/H36M/"
    cameraParameter = pickle.load(
        open("/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/wyang/dataset/cameras.pkl", "rb"))
    trainSet = H36MImgDataSet(cfg, h5pyFolder, cameraParameter, validation=False, test=False)
    '''
    augmentation = [
        ToTensor(),
        Occlusion(occlusionProb=cfg.DATASET.OCCLUSION.PROB,
                  rectangleSizeRatio=cfg.DATASET.OCCLUSION.RATIO),
        Zoom(zoomProb=cfg.DATASET.ZOOM.PROB, zoomRatio=cfg.DATASET.ZOOM.RATIO),
        Rescale(cfg.TRAINING.IMGSIZE),
        ColorJitter(brightness=cfg.DATASET.COLOR.BRIGHTNESS, contrast=cfg.DATASET.COLOR.CONTRAST,
                    saturation=cfg.DATASET.COLOR.SATURATION,
                    hue=cfg.DATASET.COLOR.HUE),
        HorizontalFlip(cfg.DATASET.HZ.PROB),
        Rotation(cfg.DATASET.RT.PROB, cfg.DATASET.RT.DEGREE),
        Translation(cfg.DATASET.TL.PROB, cfg.DATASET.TL.RATIO),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    transformCompose = [transforms.Compose([i]) for i in augmentation]
    description = [
        "Open text file",
        "Open hdf5 file",
        "Label loading",
        "Image loading",
        "ToTensor",
        "Occlusion",
        "Zoom",
        "Rescale",
        "Color Jitter",
        "Flip",
        "Rotation",
        "Translation",
        "Normalize"
    ]
    runningTime = [0] * (len(description))
    augmentationIndex = 4
    '''

    numIteration = 20
    print(f'{numIteration} iterations')

    trainLoader = DataLoader(trainSet, batch_size=cfg.TRAINING.BATCHSIZE, shuffle=True, num_workers=8)
    start = time.time()
    for i, inf in enumerate(trainLoader):
        if i == (numIteration - 1):
            break
    end = time.time()
    print(f'Total Time : {end - start}')

    '''
    path = h5pyFolder + "S1.hdf5"

    with torch.no_grad():
        for fileIdx in range(64 * numIteration):

            # open txt file
            start = time.time()
            with open("/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/wyang/test.txt", 'r') as f:
                pass
            end = time.time()
            runningTime[0] += (end - start)

            # open hdf5 file
            start = time.time()
            with h5py.File(path, 'r') as f:
                pass
            end = time.time()
            runningTime[1] += (end - start)

            # loading labels
            start = time.time()
            with h5py.File(path, 'r') as f:
                imagePath = f['ImgList'][fileIdx].decode()
                pos3d = f[f'{imagePath}_pos3d'][:, :]
                pos2d = f[f'{imagePath}_pos2d'][:, :]
                cameraType = f['cameraType'][fileIdx].decode()
                bbox = f[f'{imagePath}_bbox'][:]
            end = time.time()
            runningTime[2] += (end - start)

            # loading image
            imageIdx = int(fileIdx // 2000)
            start = time.time()
            with h5py.File(path.replace(".hdf5", f'_{imageIdx}.hdf5'), 'r') as f:
                image = f[imagePath][:, :, :]
            end = time.time()
            runningTime[3] += (end - start)

            sample = {}
            sample = {"image": image}

            for i in range(augmentationIndex, len(augmentation) + augmentationIndex):
                start = time.time()
                sample = transformCompose[i - augmentationIndex](sample)
                end = time.time()
                runningTime[i] += (end - start)

    report = ""
    totalTime = 0
    for title, t in zip(description, runningTime):
        report += f'{title} : {t} '
        totalTime += t
    report = f'Total Time : {totalTime} ' + report
    
    print(report)
    '''
