import sys
sys.path.append(r"E:\Users\Marian\Projects\VideoJointAngle\Code\Direct3DKinematicEstimation\ms_model_estimation")
## GT Generation imports
from ms_model_estimation.opensim.OpenSimModel import OpenSimModel
from ms_model_estimation.smplh_util.constants.scalingIKInf import IKTaskSet, scalingIKSet, scaleSet
from ms_model_estimation.opensim.DataReader import DataReader
from ms_model_estimation.opensim.OSLabelGenerator import BMLAmassOpenSimGTGenerator
from ms_model_estimation.pyOpenSim.TrcGenerator import TrcGenerator

## Dataset generation imports
from ms_model_estimation.bbox.extract_frame import extract_frame_from_video,search
from ms_model_estimation.bbox.BMLBBoxGenerator import BMLBBoxGenerator
from ms_model_estimation.models.hdf5.bml import search_bml_data_list,create_h5py
from ms_model_estimation.bbox.generate_index import generate_idx_file

import pickle
from pathlib import Path
from tqdm import tqdm


parent_dir = Path(__file__).parent

modelPath=str((parent_dir / "resources/opensim/BMLmovi/full_body.osim").as_posix())
amassFolder=str((parent_dir / "resources/amass/").as_posix())
v3dFolder = str((parent_dir / "resources/BMLmovi/v3d/F/").as_posix())

videoFolder = str(Path("E:/Users/Marian/Projects/VideoJointAngle/Data/BMLTestVideo/").as_posix())

outputFolder = str((parent_dir / "_dataset").as_posix())

opensimGTFolder = str((parent_dir / "resources/opensim/BMLmovi/BMLmovi").as_posix())

## GT Generation

dataReader=DataReader()
ikset=dataReader.read_ik_set(IKTaskSet)
scalingIKSet=dataReader.read_ik_set(scalingIKSet)
scaleset=dataReader.read_scale_set(scaleSet)

gtGenerator=BMLAmassOpenSimGTGenerator(
    v3dFolder, amassFolder, modelPath, scaleset , scalingIKSet , ikset,
    reScale=True, reIK=True, scaleOnly=False
)

npzPathList=gtGenerator.traverse_npz_files("BMLmovi")

# npzPathList = [path for path in npzPathList if "Subject_11" in path]

for path in npzPathList:
    gtGenerator.generate(path)

from ms_model_estimation import Postscaling_LockedCoordinates, Postscaling_UnlockedConstraints, ChangingParentMarkers

gtGenerator.opensimModel.postscaling_unlockedConstraints

## Generate HDF dataset
# Extact images to .hdf5 files
for videoPath in search(videoFolder):
    extract_frame_from_video(videoPath, outputFolder)

# Generate Bounding boxes
generator = BMLBBoxGenerator(outputFolder)
files = generator.generate()
for f in tqdm(files):
    generator.generate_bbox_from_video(f)
generator.collect_all_bbox()

# Generate movement index file
generate_idx_file(v3dFolder, outputFolder)

table = search_bml_data_list(outputFolder, opensimGTFolder, modelPath)

subject_ids = list(table['df'].subjectID.unique())
for s_id in subject_ids:
    create_h5py(outputFolder + f"Subject_{s_id}.hdf5", table["df"], table["bboxInf"])
