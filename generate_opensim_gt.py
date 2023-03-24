from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent / "ms_model_estimation"))
## GT Generation imports
from ms_model_estimation.opensim_utils.OpenSimModel import OpenSimModel
from ms_model_estimation.smplh_util.constants.scalingIKInf import IKTaskSet, scalingIKSet, scaleSet
from ms_model_estimation.opensim_utils.OpenSimDataReader import OpenSimDataReader
from ms_model_estimation.opensim_utils.OSLabelGenerator import BMLAmassOpenSimGTGenerator
from ms_model_estimation.pyOpenSim.TrcGenerator import TrcGenerator

import pickle as pkl
from tqdm import tqdm

parent_dir = Path(__file__).absolute().parent


# modelPath=str((parent_dir / "resources/opensim/BMLmovi/full_body.osim").as_posix())
modelPath=str((parent_dir / "resources/opensim/BMLmovi/full_body.osim").as_posix())
amassFolder=str((parent_dir / "resources/amass/").as_posix())
v3dFolder = str((parent_dir / "resources/V3D/F/").as_posix())

videoFolder = str(Path("E:/Users/Marian/Projects/VideoJointAngle/Data/BMLTestVideo/").as_posix())

outputFolder = str((parent_dir / "_dataset").as_posix())

opensimGTFolder = str((parent_dir / "resources/opensim/BMLmovi/BMLmovi").as_posix())

## GT Generation
if __name__ == "__main__":
    dataReader=OpenSimDataReader()
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

