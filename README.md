# Towards single camera human 3D-kinematics

<p class="callout info">This repository is currently under construction</p>

## Installation
1. Requirement

        Python 3.8.0 
        PyTorch 1.10.0
        OpenSim 4.3
        Notepad++

2. Python package

    Clone this repo and run the following:

        conda create --name <env> --file requirements.txt

3. OpenSim 4.3
    1. [Download OpenSim](https://simtk.org/frs/?group_id=91)
    2. [Set up Python environment](https://simtk-confluence.stanford.edu:8443/display/OpenSim/Scripting+in+Python)
        + In installation_folder/OpenSim 4.3/sdk/Python, run
        
                python -m pip install
            
    Note: Scripts requiring to import OpenSim are only verified on Windows.  

## Dataset and SMPL+H models
1. [BMLmovi](https://www.biomotionlab.ca/movi/)
    + Register to get access to the downloads section.
    + Download .avi videos of PG1 and PG2 cameras from the F round (F_PGX_Subject_X_L.avi).
    + Download Camera Parameters.tar.
    + Download v3d files (F_Subjects_1_45.tar).
2. [AMASS](https://amass.is.tue.mpg.de/index.html)
    + Download SMPL+H body data of BMLmovi.
3. [SMPL+H Models](https://mano.is.tue.mpg.de/index.html)
    + Register to get access to the downloads section.
    + Download the extended SMPL+H model (used in AMASS project).
4. [DMPLs](https://smpl.is.tue.mpg.de/index.html)
    + Register to get access to the downloads section.
    + Download DMPLs for AMASS.
5. [PASCAL Visual Object Classes](http://host.robots.ox.ac.uk/pascal/VOC/voc2012) (ONLY NECESSARY FOR TRAINING)    
    + Download the training/validation data

## Unpacking resources

1. Unpack the downloaded SMPL and DMPL archives into ```ms_model_estimation/resources```

2. Unpack the downloaded AMASS data into the top-level folder ```resources/amass```

3. Unpack the F_Subjects_1_45 folder and unpack content of **all subfolders** into ``resources/V3D/F``

## OpenSim GT Generation 

Run the [generate_opensim_gt](generate_opensim_gt.py) script:
```bash
python generate_opensim_gt.py
 ```

This process might take several hours!

Once the dataset is generated the scaled OpenSim model and motion files can be found in ``resources/opensim/BMLmovi/BMLmovi``

## Dataset Preparation 

After the ground truth has been generated, the dataset needs to be prepared. 

Run the [prepare_dataset](prepare_dataset.py) script and provide the location where the BMLMovi videos are stored:
```bash
python prepare_dataset.py path/to/bmlmovi/videos
 ```

This process might again take several hours!

## Evaluation

### Download models

### Run inference

Run the [run_inference](run_inference.py) script :
```bash
python run_inference.py
 ```

This will use D3KE to run predictions on the subset of BMLMovi used for testing.


## Training 


    
