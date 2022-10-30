DIR = "D:/mscWeitseYang/videoMuscle/"
BM_PATH_N = DIR + 'model/smplh/SMPLH_NEUTRAL.npz'
BM_PATH_F = DIR + 'model/smplh/SMPLH_FEMALE.npz'
BM_PATH_M = DIR + 'model/smplh/SMPLH_MALE.npz'

DMPL_PATH_N = DIR + 'model/dmpls/neutral/model.npz'
DMPL_PATH_F = DIR + 'model/dmpls/female/model.npz'
DMPL_PATH_M = DIR + 'model/dmpls/male/model.npz'

NUM_BETAS = 16
NUM_DMPLS = 8

OPENSIM_PULGIN_LIST = []

Postscaling_LockedCoordinates = [
    "ribs_bending",
    "elbow_y", "elbow_y_l",
]

Postscaling_UnlockedConstraints = [
    "sternoclavicular_r2_con", "sternoclavicular_r3_con",
    "acromioclavicular_r2_con",
    "sternoclavicular_r2_con_l", "sternoclavicular_r3_con_l",
    "acromioclavicular_r2_con_l",
]

ChangingParentMarkers = {

    # "LPSI": "lumbar5",
    # "RPSI": "lumbar5",
    "LKJC": "tibia_l",
    "LKNE": "tibia_l",
    "LKNI": "tibia_l",
    "RKJC": "tibia_r",
    "RKNE": "tibia_r",
    "RKNI": "tibia_r",
    "RANK": "talus_r",
    "RANI": "talus_r",
    "LANK": "talus_l",
    "LANI": "talus_l",
}

Global_Translation_Coordinates = [
    "pelvis_tx",
    "pelvis_ty",
    "pelvis_tz"
]

Right_Shoulder_Groups = [
    "clavicle",
    "clavphant",
    "scapula",
    "scapphant",
    "humphant",
    "humphant1",
]

Left_Shoulder_Groups = [
    "clavicle_l",
    "clavphant_l",
    "scapula_l",
    "scapphant_l",
    "humphant_l",
    "humphant1_l",
]

JointConnectionTable = {

    "ROOT": "Spine1",
    "Spine1": "Spine2",
    "Spine2": "Spine3",
    "Spine3": "NECK",
    "RCOL": "Spine3",
    "LCOL": "Spine3",
    "NECK": "HEAD",
    "RSHOULDER": "RCOL",
    "LSHOULDER": "LCOL",
    "RELC": "RSHOULDER",
    "LELC": "LSHOULDER",
    "RWRIST": "RELC",
    "LWRIST": "LELC",

    "LHIPJ": "ROOT",
    "RHIPJ": "ROOT",

    "LKJC": "LHIPJ",
    "RKJC": "RHIPJ",

    "LAJC": "LKJC",
    "RAJC": "RKJC",

    "LFOOT": "LAJC",
    "RFOOT": "RAJC",
}

MarkerConnectionTable = {
    # Pelvis
    'LPSI': "RPSI",
    'RASI': "LASI",

    # Thoratic
    'STRN': "CLAV",

    # HIP
    'RKNI': 'RKNE',
    'LKNE': 'LKNI',

    # shrank
    'LANI': 'LANK',
    'RANI': 'RANK',

    # Arm
    'LELB': 'LELBIN',
    'RELB': 'RELBIN',

    # Fore Arm
    'LIWR': 'LOWR',
    'RIWR': 'ROWR',
}
