# Estimating human musculoskeletal model using neural networks from monocular video


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
3. [SMPL+H Models](https://mano.is.tue.mpg.de/en)
    + Register to get access to the downloads section.
    + Download the extended SMPL+H model (used in AMASS project).
4. [DMPLs](https://smpl.is.tue.mpg.de/index.html)
    + Register to get access to the downloads section.
    + Download DMPLs for AMASS.
5. [PASCAL Visual Object Classes](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)


## OpenSim GT Generation 

## Dataset Preparation 

## Evaluation

## Training 


    
