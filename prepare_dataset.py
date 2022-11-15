from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent / "ms_model_estimation"))

## Dataset generation imports
from ms_model_estimation.bbox.extract_frame import extract_frame_from_video,search
from ms_model_estimation.bbox.BMLBBoxGenerator import BMLBBoxGenerator
from ms_model_estimation.models.hdf5.bml import search_bml_data_list,create_h5py
from ms_model_estimation.bbox.generate_index import generate_idx_file

import pickle as pkl
from tqdm import tqdm
import argparse

parent_dir = Path(__file__).parent

# modelPath=str((parent_dir / "resources/opensim/BMLmovi/full_body.osim").as_posix())
modelPath=str((parent_dir / "resources/opensim/full_body_wo_hands.osim").as_posix())

v3dFolder = str((parent_dir / "resources/V3D/F/").as_posix())

outputFolder = str((parent_dir / "_dataset").as_posix())

opensimGTFolder = str((parent_dir / "resources/opensim/BMLmovi/BMLmovi").as_posix())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--BMLMoviDir', action='store',
                        default="", type=str,
                        help="Folder containing the BMLMovi videos from recording round F")

    args = parser.parse_args()    
    
    videoFolder= args.BMLMoviDir 
    videoFolder = Path(videoFolder)

    ## Generate HDF dataset
    # Extact images to .hdf5 files
    print("Extracting video frames into HDF files")
    for videoPath in videoFolder.glob('*.avi'):
        extract_frame_from_video(videoPath, Path(outputFolder))

    # Generate Bounding boxes
    print("Generate Bounding boxes for each video frame")
    generator = BMLBBoxGenerator(outputFolder)
    files = generator.generate()
    for f in tqdm(files):
        generator.generate_bbox_from_video(f)
    generator.collect_all_bbox()

    # Generate movement index file
    generate_idx_file(v3dFolder, outputFolder)

    table = search_bml_data_list(outputFolder, opensimGTFolder, modelPath)

    pkl.dump(table['pyBaseModel'],open(outputFolder+'/pyBaseModel.pkl','wb'))

    subject_ids = list(table['df'].subjectID.unique())
    for s_id in subject_ids:
        create_h5py(outputFolder + f"Subject_{s_id}.hdf5", table["df"], table["bboxInf"])
