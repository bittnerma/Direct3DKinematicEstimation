from ms_model_estimation.models.H36MUtils import SELECTED_JOINTS as H36M_SELECTED_JOINTS

JOINTS = {
    0: 'RANK',
    1: 'RKNE',
    2: 'RHIP',
    3: "LHIP",
    4: "LKNE",
    5: "LANK",
    6: "PELVIS",
    7: "THORAX",
    8: "NECK",
    9: "HEAD",
    10: "RWRIST",
    11: "RELB",
    12: "RSHOULDER",
    13: "LSHOULDER",
    14: "LELB",
    15: "LWRIST"
}

SELECTED_JOINTS = [
    'RANK',
    'RKNE',
    'RHIP',
    "LHIP",
    "LKNE",
    "LANK",
    "PELVIS",
    "THORAX",
    "NECK",
    "HEAD",
    "RWRIST",
    "RELB",
    "RSHOULDER",
    "LSHOULDER",
    "LELB",
    "LWRIST"
]

MIRROR_JOINTS = []
for joint in SELECTED_JOINTS:
    if joint[0] == "R":
        joint = "L" + joint[1:]
        MIRROR_JOINTS.append(SELECTED_JOINTS.index(joint))
    elif joint[0] == "L":
        joint = "R" + joint[1:]
        MIRROR_JOINTS.append(SELECTED_JOINTS.index(joint))
    else:
        MIRROR_JOINTS.append(SELECTED_JOINTS.index(joint))


ARMS_PELVIS_LEGS = [6, 0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
ARMS_LEGS = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]

def create_table(usedJoints=None):
    if usedJoints is None:
        usedJoints = ARMS_PELVIS_LEGS
    mapping_list = []
    for idx in usedJoints:
        jointName = JOINTS[idx]
        idxH36M = H36M_SELECTED_JOINTS.index(jointName)
        mapping_list.append(idxH36M)
    return mapping_list

MAPPING_H36M_INDEX_TO_MPII = create_table()
