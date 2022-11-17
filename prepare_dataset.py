from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent / "ms_model_estimation"))

## Dataset generation imports
from ms_model_estimation.data_preparation.extract_frame import extract_frame_from_video,search
from ms_model_estimation.data_preparation.BMLBBoxGenerator import BMLBBoxGenerator
from ms_model_estimation.training.hdf5.bml import search_bml_data_list,create_h5py,create_opensim_label_dataset
from ms_model_estimation.data_preparation.generate_index import generate_idx_file
from ms_model_estimation.training.hdf5.pascal import load_occluders

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
                        default=None, type=str,
                        help="Folder containing the BMLMovi videos from recording round F")

    parser.add_argument('--PascalDir', action='store',
                        default=None, type=str,
                        help="Folder containing the Pascal VOC dataset")

    args = parser.parse_args()    
        
    if args.BMLMoviDir is None:
        raise(Exception("No BMLMovi path provided"))

    videoFolder = Path(args.BMLMoviDir)

    if not videoFolder.exists():
        raise(FileNotFoundError("Cannot find provided BMLMovi directory"))


    if args.PascalDir is None:
        print("No pascal path provided, skipping pascal.hdf generation")
    
    else:    
        pascal_root = Path(args.PascalDir)

        if pascal_root.exists():
            print("Create pascal.hdf5")
            load_occluders(outputFolder, pascal_root)
        else:
            raise(FileNotFoundError("Cannot find provided Pascal directory"))

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
        create_opensim_label_dataset(outputFolder + f"subject_{s_id}_opensim.hdf5", table['df'], table['pyBaseModel'])
        create_h5py(outputFolder + f"subject_{s_id}.hdf5", table["df"], table["bboxInf"])
