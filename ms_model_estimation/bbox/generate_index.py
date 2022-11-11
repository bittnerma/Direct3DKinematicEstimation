import scipy.io
import numpy as np
from pathlib import Path
from tqdm import tqdm

def save_convert_path(path):
    if not isinstance(path, Path):
        path = Path(path)
    return path

def generate_idx_file(v3d_dir_path,output_dir_path):
    out_dict = {}

    v3d_dir_path = save_convert_path(v3d_dir_path)
    output_dir_path = save_convert_path(output_dir_path)

    for matFile in tqdm(v3d_dir_path.glob("*.mat")):
        subjectID = matFile.stem.split("_")[3]
        matData = scipy.io.loadmat(matFile)["Subject_" + str(subjectID) + "_F"]
        data = matData["move"][0,0][0,0]['flags30'][0][0]
        out_dict[int(subjectID)] = data.T
        
    np.save(output_dir_path / "videoIdx",out_dict)
