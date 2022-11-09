'''
scaleSet is a dictionary to record all measurements in measurement-based body scaling.
The key: string
    the name of the measurement
The value: list
    MarkerPairSet: a 2D list.
        marker pairs to measure distances.
        it can have multiple marker pairs (the length >=1).
    bodies: a list.
        bodies which will be scaled.
    axes: a 2D list.
        the axes where the scaling apply to.
         0: x-axis, 1:y-axis and 2: z-axis.
         the length of axes == the length of bodies.
        if the length of the axes is equal to 1, it means the scaling use the same axes for whole bodies.
'''

scaleSet = {

    # Pelvis
    "pelvisX": {
        "MarkerPairSet": [
            ["MPSI", "MASI"]
        ],
        "bodies": ["pelvis"],
        "axes": [[0]]
    },
    "pelvisY": {
        "MarkerPairSet": [
            ["MHIPJ", "CENTER"]
        ],
        "bodies": ["pelvis"],
        "axes": [[1]]
    },
    "pelvisZ": {
        "MarkerPairSet": [
            # ["LHIPJ", "RHIPJ"],
            ["RASI", "LASI"],
            ["RPSI", "LPSI"]

        ],
        "bodies": ["pelvis"],
        "axes": [[2]]
    },

    # Spine
    "Torso_Y_L5L1": {
        "MarkerPairSet": [
            ["MPSI", "T12"],
        ],
        "bodies": [
            "lumbar5", "lumbar4", "lumbar3", "lumbar2", "lumbar1", "thoracic12"
        ],
        "axes": [
            [1], [1], [1], [1], [1], [1]
        ]
    },
    "Torso_Y_T12T11": {
        "MarkerPairSet": [
            ["T12", "T10"],
        ],
        "bodies": [
            "thoracic11", "thoracic10",
        ],
        "axes": [
            [1], [1]
        ]
    },

    "Torso_Y_T10T9": {
        "MarkerPairSet": [
            ["T10", "T8"],
        ],
        "bodies": [
            "thoracic9", "thoracic8",
        ],
        "axes": [
            [1], [1],
        ]
    },
    "Torso_Y_T8T1": {
        "MarkerPairSet": [
            ["T8", "C7"],
        ],
        "bodies": [
            "thoracic7", "thoracic6", "thoracic5",
            "thoracic4", "thoracic3", "thoracic2", "thoracic1",
        ],
        "axes": [
            [1], [1], [1],
            [1], [1], [1], [1]
        ]
    },
    "NeckY": {
        "MarkerPairSet": [
            ["C7", "HEAD"],
        ],
        "bodies": [
            "head_neck", "cerv6", "cerv5", "cerv4", "cerv3", "cerv2", "cerv1"
        ],
        "axes": [
            [1], [1], [1], [1], [1], [1], [1]
        ]
    },
    "SkullY": {
        "MarkerPairSet": [
            ["HEAD", "RFHD"],
            ["HEAD", "RBHD"],
            ["HEAD", "LFHD"],
            ["HEAD", "LBHD"],
        ],
        "bodies": [
            "skull", "jaw"
        ],
        "axes": [
            [1], [1],
        ]
    },
    "TorsoZ": {
        "MarkerPairSet": [
            ["RPSI", "LPSI"],
        ],
        "bodies": [
            "lumbar5", "lumbar4", "lumbar3", "lumbar2", "lumbar1", "thoracic12", "thoracic11",
            "thoracic10", "thoracic9", "thoracic8", "thoracic7", "thoracic6",
            "thoracic5", "thoracic4", "thoracic3", "thoracic2", "thoracic1",
            "head_neck", "cerv6", "cerv5", "cerv4", "cerv3", "cerv2", "cerv1"
        ],
        "axes": [
            [2], [2], [2], [2], [2], [2], [2],
            [2], [2], [2], [2], [2],
            [2], [2], [2], [2], [2],
            [2], [2], [2], [2], [2], [2], [2]
        ]
    },
    "TorsoX": {
        "MarkerPairSet": [
            ["MPSI", "CENTER"],
        ],
        "bodies": [
            "lumbar5", "lumbar4", "lumbar3", "lumbar2", "lumbar1", "thoracic12", "thoracic11",
            "thoracic10", "thoracic9", "thoracic8", "thoracic7", "thoracic6",
            "thoracic5", "thoracic4", "thoracic3", "thoracic2", "thoracic1",
        ],
        "axes": [
            [0], [0], [0], [0], [0], [0], [0],
            [0], [0], [0], [0], [0],
            [0], [0], [0], [0], [0],
        ]
    },
    "cerveX": {
        "MarkerPairSet": [
            ["CLAV", "C7"],
        ],
        "bodies": [
            "head_neck", "cerv6", "cerv5", "cerv4", "cerv3", "cerv2", "cerv1"
        ],
        "axes": [
            [0], [0], [0], [0], [0], [0], [0]
        ]
    },
    "SkullX": {
        "MarkerPairSet": [
            ["LBHD", "LFHD"],
            ["RBHD", "RFHD"],
        ],
        "bodies": [
            "skull", "jaw"
        ],
        "axes": [
            [0], [0]
        ]
    },
    "SkullZ": {
        "MarkerPairSet": [
            #["rear", "lear"],
            ["LBHD", "RBHD"],
            ["LFHD", "RFHD"],
        ],
        "bodies": [
            "skull", "jaw"
        ],
        "axes": [
            [2], [2]
        ]
    },

    # Femur
    "femurY": {
        "MarkerPairSet": [
            ["RHIPJ", "Virtual_RKJC"]
        ],
        "bodies": ["femur_r"],
        "axes": [[1]]
    },
    "femurY_l": {
        "MarkerPairSet": [
            ["LHIPJ", "Virtual_LKJC"],
        ],
        "bodies": ["femur_l"],
        "axes": [[1]]
    },
    "femurXZ": {
        "MarkerPairSet": [
            ["RKNI", "RKNE"]
        ],
        "bodies": ["femur_r"],
        "axes": [[0, 2]]
    },
    "femurXZ_l": {
        "MarkerPairSet": [
            ["LKNE", "LKNI"],
        ],
        "bodies": ["femur_l"],
        "axes": [[0, 2]]
    },
    "tibiaY": {
        "MarkerPairSet": [
            ["RKJC", "Virtual_RAJC"],
        ],
        "bodies": ["tibia_r", "talus_r"],
        "axes": [[1], [1]]
    },
    "tibiaY_l": {
        "MarkerPairSet": [
            ["LKJC", "Virtual_LAJC"],
        ],
        "bodies": ["tibia_l", "talus_l"],
        "axes": [[1], [1]]
    },
    "tibiaXZ": {
        "MarkerPairSet": [
            ["RANI", "RANK"]
        ],
        "bodies": ["tibia_r", "talus_r"],
        "axes": [[0, 2], [0, 2]]
    },
    "tibiaXZ_l": {
        "MarkerPairSet": [
            ["LANK", "LANI"],
        ],
        "bodies": ["tibia_l", "talus_l"],
        "axes": [[0, 2], [0, 2]]
    },
    "calcnX": {
        "MarkerPairSet": [
            ["Virtual_Ground_RHEE", "Virtual_Ground_RFOOT"],
        ],
        "bodies": ["calcn_r"],
        "axes": [[0]]
    },
    "footY": {
        "MarkerPairSet": [
            ["Virtual_Ground_RAJC", "Virtual_RAJC"],
        ],
        "bodies": ["calcn_r", "toes_r"],
        "axes": [[1], [1]]
    },
    "calcnZ": {
        "MarkerPairSet": [
            ["Virtual_Ground_RMT1", "Virtual_Ground_RMT5"],
        ],
        "bodies": ["calcn_r"],
        "axes": [[2]]
    },
    "calcnX_l": {
        "MarkerPairSet": [
            ["Virtual_Ground_LHEE", "Virtual_Ground_LFOOT"],
        ],
        "bodies": ["calcn_l"],
        "axes": [[0]]
    },
    "footY_l": {
        "MarkerPairSet": [
            ["Virtual_Ground_LAJC", "Virtual_LAJC"],
        ],
        "bodies": ["calcn_l", "toes_l"],
        "axes": [[1], [1]]
    },
    "calcnZ_l": {
        "MarkerPairSet": [
            ["Virtual_Ground_LMT1", "Virtual_Ground_LMT5"],
        ],
        "bodies": ["calcn_l"],
        "axes": [[2]]
    },
    "toesX": {
        "MarkerPairSet": [
            ["Virtual_Ground_RMT5", "Virtual_Ground_RMT1"],
        ],
        "bodies": ["toes_r"],
        "axes": [[0]]
    },
    "toesX_l": {
        "MarkerPairSet": [
            ["Virtual_Ground_LMT1", "Virtual_Ground_LMT5"],
        ],
        "bodies": ["toes_l"],
        "axes": [[0]]
    },
    "toesYZ": {
        "MarkerPairSet": [
            ["Virtual_Ground_RFOOT", "Virtual_Ground_RTOE"]
        ],
        "bodies": ["toes_r"],
        "axes": [[1, 2]]
    },
    "toesYZ_l": {
        "MarkerPairSet": [
            ["Virtual_Ground_LFOOT", "Virtual_Ground_LTOE"]
        ],
        "bodies": ["toes_l"],
        "axes": [[1, 2]]
    },




    "ribsX": {
        "MarkerPairSet": [
            ["CLAV", "C7"],
        ],
        "bodies": ["thorax"],
        "axes": [[0]]
    },
    "ribsY": {
        "MarkerPairSet": [
            ["T10", "C7"],
        ],
        "bodies": ["thorax"],
        "axes": [[1]]
    },
    "ribsZ": {
        "MarkerPairSet": [
            ["RSHO", "LSHO"],
        ],
        "bodies": ["thorax"],
        "axes": [[2]]
    },

    "calvY": {
        "MarkerPairSet": [
            ["C7", "CLAV"],
        ]
        ,
        "bodies": ['clavicle', 'clavphant',
                   'clavicle_l', 'clavphant_l', ],
        "axes": [
            [1],
            [1],
            [1],
            [1],
        ]
    },
    "scapulaY": {
        "MarkerPairSet": [
            ["C7", "T10"],
        ]
        ,
        "bodies": ['scapula', 'scapphant', 'humphant', 'humphant1',
                   'scapula_l', 'scapphant_l', 'humphant_l', 'humphant1_l'],
        "axes": [
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1]
        ]
    },
    "calvX": {
        "MarkerPairSet": [
            ["C7", "CLAV"],
            ["RSHO", "CLAV"],
        ]
        ,
        "bodies": ['clavicle', 'clavphant'],
        "axes": [
            [0],
            [0],
        ]
    },
    "calv_l_X": {
        "MarkerPairSet": [
            ["C7", "CLAV"],
            ["LSHO", "CLAV"],
        ]
        ,
        "bodies": ['clavicle_l', 'clavphant_l'],
        "axes": [

            [0],
            [0],
        ]
    },
    "scapulaX": {
        "MarkerPairSet": [
            ["STRN", "T10"],
        ]
        ,
        "bodies": ['scapula', 'scapphant', 'humphant', 'humphant1'],
        "axes": [
            [0],
            [0],
            [0],
            [0],
        ]
    },
    "scapula_l_X": {
        "MarkerPairSet": [
            ["STRN", "T10"],
        ]
        ,
        "bodies": ['scapula_l', 'scapphant_l', 'humphant_l', 'humphant1_l'],
        "axes": [
            [0],
            [0],
            [0],
            [0],
        ]
    },
    "calvZ": {
        "MarkerPairSet": [
            ["C7", "RSHO"]
        ]
        ,
        "bodies": ['clavicle', 'clavphant', 'scapula', 'scapphant', 'humphant', 'humphant1'],
        "axes": [
            [2],
            [2],
            [2],
            [2],
            [2],
            [2],
        ]
    },
    "calv_l_Z": {
        "MarkerPairSet": [
            ["LSHO", "C7"]
        ]
        ,
        "bodies": ['clavicle_l', 'clavphant_l', 'scapula_l', 'scapphant_l', 'humphant_l', 'humphant1_l'],
        "axes": [
            [2],
            [2],
            [2],
            [2],
            [2],
            [2],
        ]
    },

    # Arms
    "ArmY": {
        "MarkerPairSet": [
            ["Virtual_RELC", "RSHO"],
        ],
        "bodies": ["humerus"],
        "axes": [[1]]
    },
    "Arm_l_Y": {
        "MarkerPairSet": [
            ["Virtual_LELC", "LSHO"]
        ],
        "bodies": ["humerus_l"],
        "axes": [[1]]
    },
    "ArmXZ": {
        "MarkerPairSet": [
            ["RELB", "RELBIN"],
        ],
        "bodies": ["humerus"],
        "axes": [[0, 2]]
    },
    "Arm_l_XZ": {
        "MarkerPairSet": [
            ["LELB", "LELBIN"]
        ],
        "bodies": ["humerus_l"],
        "axes": [[0, 2]]
    },
    "formArmY": {
        "MarkerPairSet": [
            ["Virtual_RWRIST", "RELC"],
        ],
        "bodies": ["ulna", "radius"],
        "axes": [[1], [1]]
    },
    "formArm_l_Y": {
        "MarkerPairSet": [
            ["Virtual_LWRIST", "LELC"]
        ],
        "bodies": ["ulna_l", "radius_l"],
        "axes": [[1], [1]]
    },
    "formArmXZ": {
        "MarkerPairSet": [
            ["RIWR", "ROWR"],
        ],
        "bodies": ["ulna", "radius"],
        "axes": [[0, 2], [0, 2]]
    },
    "formArm_l_XZ": {
        "MarkerPairSet": [
            ["LIWR", "LOWR"]
        ],
        "bodies": ["ulna_l", "radius_l"],
        "axes": [[0, 2], [0, 2]]
    },

    # Hands
    "Hands_XYZ": {
        "MarkerPairSet": [
            ["Virtual_RWRIST", "right_thumb1"],
            ["Virtual_RWRIST", "right_index1"],
            ["Virtual_RWRIST", "right_middle1"],
            ["Virtual_RWRIST", "right_pinky1"],
            ["Virtual_RWRIST", "right_ring1"],
        ],
        "bodies": ["lunate", "scaphoid", "pisiform", "triquetrum", "capitate", "trapezium",
                   "trapezoid", "hamate", "firstmc1", "secondmc", "thirdmc", "fourthmc", "fifthmc"],
        "axes": [[0,1,2]]
    },
    "Hands_XYZ_l": {
        "MarkerPairSet": [
            ["Virtual_LWRIST", "left_thumb1"],
            ["Virtual_LWRIST", "left_index1"],
            ["Virtual_LWRIST", "left_middle1"],
            ["Virtual_LWRIST", "left_pinky1"],
            ["Virtual_LWRIST", "left_ring1"],
        ],
        "bodies": ["lunate_l", "scaphoid_l", "pisiform_l", "triquetrum_l", "capitate_l", "trapezium_l",
                   "trapezoid_l", "hamate_l", "firstmc1_l", "secondmc_l", "thirdmc_l", "fourthmc_l", "fifthmc_l"],
        "axes": [[0,1,2]]
    },


    "thumb1_XYZ_l": {
        "MarkerPairSet": [
            ["left_thumb1", "left_thumb2"],
        ],
        "bodies": ["firstmc_l"],
        "axes": [[0,1,2]]
    },
    "thumb2_XYZ_l": {
        "MarkerPairSet": [
            ["left_thumb2", "left_thumb3"],
        ],
        "bodies": ["proximal_thumb_l"],
        "axes": [[0,1,2]]
    },
    "thumb3_XYZ_l": {
        "MarkerPairSet": [
            ["left_thumb3", "lthumb"],
        ],
        "bodies": ["distal_thumb_l"],
        "axes": [[0,1,2]]
    },
    "index1_XYZ_l": {
        "MarkerPairSet": [
            ["left_index1", "left_index2"],
        ],
        "bodies": ["2proxph_l"],
        "axes": [[0,1,2]]
    },
    "index2_XYZ_l": {
        "MarkerPairSet": [
            ["left_index2", "left_index3"],
        ],
        "bodies": ["2midph_l"],
        "axes": [[0,1,2]]
    },
    "index3_XYZ_l": {
        "MarkerPairSet": [
            ["left_index3", "lindex"],
        ],
        "bodies": ["2distph_l"],
        "axes": [[0,1,2]]
    },
    "middle1_XYZ_l": {
        "MarkerPairSet": [
            ["left_middle1", "left_middle2"],
        ],
        "bodies": ["3proxph_l"],
        "axes": [[0,1,2]]
    },
    "middle2_XYZ_l": {
        "MarkerPairSet": [
            ["left_middle2", "left_middle3"],
        ],
        "bodies": ["3midph_l"],
        "axes": [[0,1,2]]
    },
    "middle3_XYZ_l": {
        "MarkerPairSet": [
            ["left_middle3", "lmiddle"],
        ],
        "bodies": ["3distph_l"],
        "axes": [[0,1,2]]
    },
    "ring1_XYZ_l": {
        "MarkerPairSet": [
            ["left_ring1", "left_ring2"],
        ],
        "bodies": ["4proxph_l"],
        "axes": [[0,1,2]]
    },
    "ring2_XYZ_l": {
        "MarkerPairSet": [
            ["left_ring2", "left_ring3"],
        ],
        "bodies": ["4midph_l"],
        "axes": [[0,1,2]]
    },
    "ring3_XYZ_l": {
        "MarkerPairSet": [
            ["left_ring3", "lring"],
        ],
        "bodies": ["4distph_l"],
        "axes": [[0,1,2]]
    },
    "pinky1_XYZ_l": {
        "MarkerPairSet": [
            ["left_pinky1", "left_pinky2"],
        ],
        "bodies": ["5proxph_l"],
        "axes": [[0,1,2]]
    },
    "pinky2_XYZ_l": {
        "MarkerPairSet": [
            ["left_pinky2", "left_pinky3"],
        ],
        "bodies": ["5midph_l"],
        "axes": [[0,1,2]]
    },
    "pinky3_XYZ_l": {
        "MarkerPairSet": [
            ["left_pinky3", "lpinky"],
        ],
        "bodies": ["5distph_l"],
        "axes": [[0,1,2]]
    },

    "thumb1_XYZ": {
        "MarkerPairSet": [
            ["right_thumb1", "right_thumb2"],
        ],
        "bodies": ["firstmc"],
        "axes": [[0, 1, 2]]
    },
    "thumb2_XYZ": {
        "MarkerPairSet": [
            ["right_thumb2", "right_thumb3"],
        ],
        "bodies": ["proximal_thumb"],
        "axes": [[0, 1, 2]]
    },
    "thumb3_XYZ": {
        "MarkerPairSet": [
            ["right_thumb3", "rthumb"],
        ],
        "bodies": ["distal_thumb"],
        "axes": [[0, 1, 2]]
    },
    "index1_XYZ": {
        "MarkerPairSet": [
            ["right_index1", "right_index2"],
        ],
        "bodies": ["2proxph"],
        "axes": [[0, 1, 2]]
    },
    "index2_XYZ": {
        "MarkerPairSet": [
            ["right_index2", "right_index3"],
        ],
        "bodies": ["2midph"],
        "axes": [[0, 1, 2]]
    },
    "index3_XYZ": {
        "MarkerPairSet": [
            ["right_index3", "rindex"],
        ],
        "bodies": ["2distph"],
        "axes": [[0, 1, 2]]
    },
    "middle1_XYZ": {
        "MarkerPairSet": [
            ["right_middle1", "right_middle2"],
        ],
        "bodies": ["3proxph"],
        "axes": [[0, 1, 2]]
    },
    "middle2_XYZ": {
        "MarkerPairSet": [
            ["right_middle2", "right_middle3"],
        ],
        "bodies": ["3midph"],
        "axes": [[0, 1, 2]]
    },
    "middle3_XYZ": {
        "MarkerPairSet": [
            ["right_middle3", "rmiddle"],
        ],
        "bodies": ["3distph"],
        "axes": [[0, 1, 2]]
    },
    "ring1_XYZ": {
        "MarkerPairSet": [
            ["right_ring1", "right_ring2"],
        ],
        "bodies": ["4proxph"],
        "axes": [[0, 1, 2]]
    },
    "ring2_XYZ": {
        "MarkerPairSet": [
            ["right_ring2", "right_ring3"],
        ],
        "bodies": ["4midph"],
        "axes": [[0, 1, 2]]
    },
    "ring3_XYZ": {
        "MarkerPairSet": [
            ["right_ring3", "rring"],
        ],
        "bodies": ["4distph"],
        "axes": [[0, 1, 2]]
    },
    "pinky1_XYZ": {
        "MarkerPairSet": [
            ["right_pinky1", "right_pinky2"],
        ],
        "bodies": ["5proxph"],
        "axes": [[0, 1, 2]]
    },
    "pinky2_XYZ": {
        "MarkerPairSet": [
            ["right_pinky2", "right_pinky3"],
        ],
        "bodies": ["5midph"],
        "axes": [[0, 1, 2]]
    },
    "pinky3_XYZ": {
        "MarkerPairSet": [
            ["right_pinky3", "rpinky"],
        ],
        "bodies": ["5distph"],
        "axes": [[0, 1, 2]]
    },
}

'''
Marker weights for body scaling.
Weights should be between 0 and 20.
'''

scalingIKSet = {

    # Virtual Markers
    "MASI": 20,
    "MPSI": 20,
    "MHIPJ": 20,
    "CENTER": 20,
    "Virtual_LWRIST": 20,
    "Virtual_RWRIST": 20,
    "Virtual_LAJC": 20,
    "Virtual_RAJC": 20,
    "Virtual_LKJC": 20,
    "Virtual_RKJC": 20,
    "Virtual_Ground_RAJC": 20,
    "Virtual_Ground_LAJC": 20,
    "Virtual_Ground_RFOOT": 1,
    "Virtual_Ground_LFOOT": 1,
    "Virtual_Ground_RTOE": 1,
    "Virtual_Ground_LTOE": 1,
    "Virtual_Ground_RMT1": 1,
    "Virtual_Ground_LMT1": 1,
    "Virtual_Ground_RMT5": 1,
    "Virtual_Ground_LMT5": 1,
    "Virtual_Ground_RHEE": 1,
    "Virtual_Ground_LHEE": 1,
    "Virtual_LELC" : 0,
    "Virtual_RELC" : 0,

    # Pelvis
    'ROOT': 0,
    'LPSI': 20,
    'RPSI': 20,
    'RASI': 20,
    'LASI': 20,
    'LHIPJ': 20,
    'RHIPJ': 20,

    # Markers on Spine
    'T12': 10,
    'T10': 20,
    'T8': 10,
    "T1": 10,


    # Lumbar
    'Spine1': 0,


    # Thoratic
    'STRN': 20,
    'CLAV': 20,
    'Spine2': 0,
    'Spine3': 0,
    'LCOL': 0,
    'RCOL': 0,


    # Neck
    'C7': 20,
    'HEAD': 0,
    'NECK': 0,


    # Head
    'LBHD': 1,
    'LFHD': 1,
    'RBHD': 1,
    'RFHD': 1,



    # HIP
    'LKJC': 20,
    'RKJC': 20,
    'RKNE': 20,
    'RKNI': 20,
    'LKNE': 20,
    'LKNI': 20,


    # shrank
    'LAJC': 0,
    'RAJC': 0,
    'LANK': 20,
    'RANK': 20,
    'LANI': 20,
    'RANI': 20,


    # Foot
    'LFOOT': 20,
    'RFOOT': 20,
    'LHEE': 20,
    'RHEE': 20,
    'LMT5': 1,
    'RMT5': 1,
    'RMT1': 1,
    'LMT1': 1,
    'RTOE': 10,
    'LTOE': 10,


    # Calvicle
    'LCLAV': 0,
    'LSHO': 20,
    'RSHO': 20,
    'RCLAV': 0,


    # Arm
    'LELC': 0,
    'RELC': 0,
    'LELB': 20,
    'RELB': 20,
    #'LELB2': 0,
    #'RELB2': 0,
    'LELBIN': 20,
    'RELBIN': 20,
    'LSHOULDER': 0,
    'RSHOULDER': 0,


    # Fore Arm
    'LWRIST': 0,
    'RWRIST': 0,
    'LIWR': 20,
    'RIWR': 20,
    'LOWR': 20,
    'ROWR': 20,


    # Hands
    'left_index1': 20,
    'left_index2': 20,
    'left_index3': 20,
    'left_middle1': 20,
    'left_middle2': 20,
    'left_middle3': 20,
    'left_pinky1': 20,
    'left_pinky2': 20,
    'left_pinky3': 20,
    'left_ring1': 20,
    'left_ring2': 20,
    'left_ring3': 20,
    'left_thumb1': 20,
    'left_thumb2': 20,
    'left_thumb3': 20,

    'right_index1': 20,
    'right_index2': 20,
    'right_index3': 20,
    "right_middle1": 20,
    "right_middle2": 20,
    "right_middle3": 20,
    'right_pinky1': 20,
    'right_pinky2': 20,
    'right_pinky3': 20,
    "right_ring1": 20,
    'right_ring2': 20,
    'right_ring3': 20,
    'right_thumb1': 20,
    'right_thumb2': 20,
    'right_thumb3': 20,

    'rthumb': 20,
    'rindex': 20,
    'rmiddle': 20,
    'rring': 20,
    'rpinky': 20,
    'lthumb': 20,
    'lindex': 20,
    'lmiddle': 20,
    'lring': 20,
    'lpinky': 20,
}


'''
Marker weights for inverse kinematics.
Weights should be between 0 and 20.
'''

IKTaskSet = {

    # Virtual Markers
    "MASI": 0,
    "MPSI": 0,
    "MHIPJ": 0,
    "CENTER": 0,
    "Virtual_LWRIST": 0,
    "Virtual_RWRIST": 0,
    "Virtual_LAJC": 0,
    "Virtual_RAJC": 0,
    "Virtual_LKJC": 0,
    "Virtual_RKJC": 0,
    "Virtual_Ground_RAJC": 0,
    "Virtual_Ground_LAJC": 0,
    "Virtual_Ground_RFOOT": 0,
    "Virtual_Ground_LFOOT": 0,
    "Virtual_Ground_RTOE": 0,
    "Virtual_Ground_LTOE": 0,
    "Virtual_Ground_RMT1": 0,
    "Virtual_Ground_LMT1": 0,
    "Virtual_Ground_RMT5": 0,
    "Virtual_Ground_LMT5": 0,
    "Virtual_Ground_RHEE": 0,
    "Virtual_Ground_LHEE": 0,
    "Virtual_LELC" : 0,
    "Virtual_RELC" : 0,

    # Pelvis
    'ROOT': 10,
    'LPSI': 20,
    'RPSI': 20,
    'RASI': 20,
    'LASI': 20,
    'LHIPJ': 10,
    'RHIPJ': 10,

    # Tracking Markers on Spine
    "Tracking_L5": 1,
    "Tracking_L4": 1,
    "Tracking_L3": 1,
    "Tracking_L2": 1,
    "Tracking_L1": 1,
    'T12': 10,
    "Tracking_T11": 1,
    "Tracking_T9": 1,
    'T10': 20,
    "Tracking_T7": 1,
    'T8': 10,
    "Tracking_T6": 1,
    "Tracking_T5": 1,
    "Tracking_T4": 1,
    "Tracking_T3": 1,
    "Tracking_T2": 1,
    "T1": 10,

    # Lumbar
    'Spine1': 1,

    # Spine
    'Spine2': 1,
    'Spine3': 1,
    'LCOL': 1,
    'RCOL': 1,

    # Thoratic
    'STRN': 20,
    'CLAV': 20,

    # Neck
    'C7': 20,
    'NECK': 10,
    'HEAD': 10,

    # Head
    'LBHD': 10,
    'LFHD': 10,
    'RBHD': 10,
    'RFHD': 10,


    # HIP
    'LKJC': 1,
    'RKJC': 1,
    'RKNE': 20,
    'RKNI': 20,
    'LKNE': 20,
    'LKNI': 20,

    # shrank
    'LAJC': 1,
    'RAJC': 1,
    'LANK': 20,
    'RANK': 20,
    'LANI': 20,
    'RANI': 20,

    # Foot
    'LFOOT': 20,
    'RFOOT': 20,
    'LHEE': 20,
    'RHEE': 20,
    'LMT5': 1,
    'RMT5': 1,
    'RMT1': 1,
    'LMT1': 1,
    'RTOE': 10,
    'LTOE': 10,

    #  Calvicle
    'LCLAV': 1,
    'RCLAV': 1,
    'LSHO': 20,
    'RSHO': 20,


    # Arm
    'LELC': 1,
    'RELC': 1,
    'LELB': 20,
    'RELB': 20,
    'LELBIN': 20,
    'RELBIN': 20,
    'LSHOULDER': 0,
    'RSHOULDER': 0,


    # Fore Arm
    #'LELB2': 1,
    #'RELB2': 1,
    'LWRIST': 1,
    'RWRIST': 1,
    'LIWR': 20,
    'RIWR': 20,
    'LOWR': 20,
    'ROWR': 20,

    # Hands
    'left_index1': 20,
    'left_index2': 20,
    'left_index3': 20,
    'left_middle1': 20,
    'left_middle2': 20,
    'left_middle3': 20,
    'left_pinky1': 20,
    'left_pinky2': 20,
    'left_pinky3': 20,
    'left_ring1': 20,
    'left_ring2': 20,
    'left_ring3': 20,
    'left_thumb1': 20,
    'left_thumb2': 20,
    'left_thumb3': 20,
    'right_index1': 20,
    'right_index2': 20,
    'right_index3': 20,
    "right_middle1": 20,
    "right_middle2": 20,
    "right_middle3": 20,
    'right_pinky1': 20,
    'right_pinky2': 20,
    'right_pinky3': 20,
    "right_ring1": 20,
    'right_ring2': 20,
    'right_ring3': 20,
    'right_thumb1': 20,
    'right_thumb2': 20,
    'right_thumb3': 20,
    'rthumb': 20,
    'rindex': 20,
    'rmiddle': 20,
    'rring': 20,
    'rpinky': 20,
    'lthumb': 20,
    'lindex': 20,
    'lmiddle': 20,
    'lring': 20,
    'lpinky': 20,
}
