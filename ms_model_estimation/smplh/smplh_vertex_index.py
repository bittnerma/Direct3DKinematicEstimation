'''
Indices of SMPl joints.
'''
smplHJoint = {

    "ROOT": 0,
    "Spine1": 3,
    "Spine2": 6,
    "Spine3": 9,

    "LHIPJ": 1,
    "RHIPJ": 2,

    "LKJC": 4,
    "RKJC": 5,

    "LAJC": 7,
    "RAJC": 8,

    "LFOOT": 10,
    "RFOOT": 11,

    'NECK': 12,
    'LCOL': 13,
    'RCOL': 14,
    'HEAD': 15,

    "LSHOULDER": 16,
    "RSHOULDER": 17,

    "LELC": 18,
    "RELC": 19,

    "LWRIST": 20,
    "RWRIST": 21,

    'left_index1': 22,
    'left_index2': 23,
    'left_index3': 24,
    'left_middle1': 25,
    'left_middle2': 26,
    'left_middle3': 27,
    'left_pinky1': 28,
    'left_pinky2': 29,
    'left_pinky3': 30,
    'left_ring1': 31,
    'left_ring2': 32,
    'left_ring3': 33,
    'left_thumb1': 34,
    'left_thumb2': 35,
    'left_thumb3': 36,

    'right_index1': 37,
    'right_index2': 38,
    'right_index3': 39,
    "right_middle1": 40,
    "right_middle2": 41,
    "right_middle3": 42,
    'right_pinky1': 43,
    'right_pinky2': 44,
    'right_pinky3': 45,
    'right_ring1': 46,
    'right_ring2': 47,
    'right_ring3': 48,
    'right_thumb1': 49,
    'right_thumb2': 50,
    'right_thumb3': 51,
}

'''
a List to record the connection of SMPL's pose.
'''
JOINT_CONNECTION = {
    "ROOT": ["RHIPJ", "LHIPJ", "Spine1"],
    "RHIPJ": "RKJC",
    "RKJC": "RAJC",
    "RAJC": "RFOOT",
    "LHIPJ": "LKJC",
    "LKJC": "LAJC",
    "LAJC": "LFOOT",
    "Spine1": "Spine2",
    "Spine2": "Spine3",
    "NECK": "HEAD",
    "Spine3": ["NECK", "RCOL", "LCOL"],
    "LCOL": "LSHOULDER",
    "RCOL": "RSHOULDER",
    "LSHOULDER": "LELC",
    "LELC": "LWRIST",
    "RSHOULDER": "RELC",
    "RELC": "RWRIST",
}
CONNECTION_INDEX_LIST = []
for k, values in JOINT_CONNECTION.items():
    if isinstance(values, list):
        for v in values:
            CONNECTION_INDEX_LIST.append([smplHJoint[k], smplHJoint[v]])
    else:
        CONNECTION_INDEX_LIST.append([smplHJoint[k], smplHJoint[values]])


'''
Indices of virtual markers.
We choose vertices of SMPL body model as markers. We call them virtual markers. 
'''
smplHMarker = {

    "STRN": 3079,
    "T1": 3012,
    "T8": 3014,
    "T10": 3017,
    "T12": 3173,
    "C7": 828,
    "CLAV": 3171,

    "LANK": 3327,
    "RANK": 6728,
    "LANI": 3198,
    "RANI": 6598,

    "LASI": 1799,
    "RASI": 5262,
    "LPSI": 3097,
    "RPSI": 6521,

    "LSHO": 606,
    "RSHO": 4094,

    "LTOE": 3232,
    "RTOE": 6634,

    'RHEE': 6787,
    'LHEE': 3386,

    "LKNE": 1010,
    "RKNE": 4495,

    "LKNI": 1148,
    "RKNI": 4634,

    "RIWR": 5691,
    "LIWR": 2230,

    "RELB": 5091,
    "LELB": 1620,

    "RELBIN": 5131,
    "LELBIN": 1661,

    "LOWR": 2108,
    "ROWR": 5568,

    "RCLAV": 4780,
    "LCLAV": 1298,

    "RFHD": 3645,
    "LBHD": 272,
    "LFHD": 134,
    "RBHD": 3783,

    "LMT1": 3338,
    "LMT5": 3346,
    "RMT1": 6738,
    "RMT5": 6747,

    "Tracking_L5": 3021,
    "Tracking_L4": 3022,
    "Tracking_L3": 3502,
    "Tracking_L2": 3023,
    "Tracking_L1": 3024,
    "Tracking_T11": 3016,
    "Tracking_T9": 3505,
    "Tracking_T7": 3015,
    "Tracking_T6": 3027,
    "Tracking_T5": 3029,
    "Tracking_T4": 1755,
    "Tracking_T3": 2877,
    "Tracking_T2": 1305,

    # SMPL+H Landmark from SMPL-X Github Repository
    'rthumb': 6191,
    'rindex': 5782,
    'rmiddle': 5905,
    'rring': 6016,
    'rpinky': 6133,
    'lthumb': 2746,
    'lindex': 2319,
    'lmiddle': 2445,
    'lring': 2556,
    'lpinky': 2673,
}
