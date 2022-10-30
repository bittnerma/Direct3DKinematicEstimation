from ms_model_estimation.models.BMLUtils import PredictedMarkers, smplHJoint, Joints_Inside_Cylinder_Body
import numpy as np
from ms_model_estimation.smplh.scalingIKInf import IKTaskSet

# predicted coordinates and their mirror
PredictedCoordinates = [
    "L5_S1_FE",
    "L5_S1_LB",
    "L5_S1_AR",
    "L4_L5_FE",
    "L4_L5_LB",
    "L4_L5_AR",
    "T12_L1_FE",
    "T12_L1_LB",
    "T12_L1_AR",
    "T11_T12_FE",
    "T11_T12_LB",
    "T11_T12_AR",
    "T10_T11_FE",
    "T10_T11_LB",
    "T10_T11_AR",
    "T9_T10_FE",
    "T9_T10_LB",
    "T9_T10_AR",
    "T8_T9_FE",
    "T8_T9_LB",
    "T8_T9_AR",
    "T1_T2_FE",
    "T1_T2_LB",
    "T1_T2_AR",
    "T1_head_neck_FE",
    "T1_head_neck_LB",
    "T1_head_neck_AR",
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_angle_r",
    "knee_angle_y_r",
    "knee_angle_x_r",
    "ankle_angle_r",
    "subtalar_angle_r",
    "mtp_angle_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_angle_l",
    "knee_angle_y_l",
    "knee_angle_x_l",
    "ankle_angle_l",
    "subtalar_angle_l",
    "mtp_angle_l",
    "skull_z",
    "skull_y",
    "skull_x",
    "sternoclavicular_r2",
    "sternoclavicular_r3",
    "acromioclavicular_r2",
    "elv_angle",
    "shoulder_elv",
    "shoulder1_r2",
    "shoulder_rot",
    "elbow_flexion",
    "elbow_x",
    "pro_sup",
    "sternoclavicular_r2_l",
    "sternoclavicular_r3_l",
    "acromioclavicular_r2_l",
    "elv_angle_l",
    "shoulder_elv_l",
    "shoulder1_r2_l",
    "shoulder_rot_l",
    "elbow_flexion_l",
    "elbow_x_l",
    "pro_sup_l",
    "ribs_bending",
]
MirrorPredictedCoordinates = {
    "hip_flexion_r": "hip_flexion_l",
    "hip_adduction_r": "hip_adduction_l",
    "hip_rotation_r": "hip_rotation_l",
    "knee_angle_r": "knee_angle_l",
    "knee_angle_y_r": "knee_angle_y_l",
    "knee_angle_x_r": "knee_angle_x_l",
    "ankle_angle_r": "ankle_angle_l",
    "subtalar_angle_r": "subtalar_angle_l",
    "mtp_angle_r": "mtp_angle_l",
    "sternoclavicular_r2": "sternoclavicular_r2_l",
    "sternoclavicular_r3": "sternoclavicular_r3_l",
    "acromioclavicular_r2": "acromioclavicular_r2_l",
    "elv_angle": "elv_angle_l",
    "shoulder_elv": "shoulder_elv_l",
    "shoulder1_r2": "shoulder1_r2_l",
    "shoulder_rot": "shoulder_rot_l",
    "elbow_flexion": "elbow_flexion_l",
    "elbow_x": "elbow_x_l",
    "pro_sup": "pro_sup_l"
}
tmp = {}
for k, v in MirrorPredictedCoordinates.items():
    tmp[v] = k
MirrorPredictedCoordinates.update(tmp)

MIRROR_COORDINATES = []
for idx, coordinate in enumerate(PredictedCoordinates):
    if coordinate in MirrorPredictedCoordinates:
        coordinate = MirrorPredictedCoordinates[coordinate]
        MIRROR_COORDINATES.append(PredictedCoordinates.index(coordinate))
    else:
        MIRROR_COORDINATES.append(idx)
MIRROR_SIGN_COORDINATES = []
for coordinate in PredictedCoordinates:
    if "AR" == coordinate[-2:] or "LB" == coordinate[-2:] or "skull_y" == coordinate or "skull_x" == coordinate:
        MIRROR_SIGN_COORDINATES.append(-1)
    else:
        MIRROR_SIGN_COORDINATES.append(1)
MIRROR_SIGN_COORDINATES = np.array(MIRROR_SIGN_COORDINATES)

# predicted bone scales
PredictedBones = [
    "pelvis",
    "lumbar5",
    "lumbar4",
    "lumbar3",
    "lumbar2",
    "lumbar1",
    "thoracic12",
    "thoracic11",
    "thoracic10",
    "thoracic9",
    "thoracic8",
    "thoracic7",
    "thoracic6",
    "thoracic5",
    "thoracic4",
    "thoracic3",
    "thoracic2",
    "thoracic1",
    "head_neck",
    "femur_r",
    "tibia_r",
    "talus_r",
    "calcn_r",
    "toes_r",
    "femur_l",
    "tibia_l",
    "talus_l",
    "calcn_l",
    "toes_l",
    "cerv6",
    "cerv5",
    "cerv4",
    "cerv3",
    "cerv2",
    "cerv1",
    "jaw",
    "skull",
    "thorax",
    "clavicle",
    "scapula",
    "humerus",
    "ulna",
    "radius",
    "clavicle_l",
    "scapula_l",
    "humerus_l",
    "ulna_l",
    "radius_l",
]
MirrorPredictedBones = {
    "femur_r": "femur_l",
    "tibia_r": "tibia_l",
    "talus_r": "talus_l",
    "calcn_r": "calcn_l",
    "toes_r": "toes_l",
    "clavicle": "clavicle_l",
    "scapula": "scapula_l",
    "humerus": "humerus_l",
    "ulna": "ulna_l",
    "radius": "radius_l",
}
tmp = {}
for k, v in MirrorPredictedBones.items():
    tmp[v] = k
MirrorPredictedBones.update(tmp)
MIRROR_BONES = []
for idx, body in enumerate(PredictedBones):
    if body in MirrorPredictedBones:
        body = MirrorPredictedBones[body]
        MIRROR_BONES.append(PredictedBones.index(body))
    else:
        MIRROR_BONES.append(idx)

# predicted openSim joints
PredictedOSJoints = [
    "ground_pelvis",
    "L5_S1_IVDjnt",
    "L4_L5_IVDjnt",
    "L3_L4_IVDjnt",
    "L2_L3_IVDjnt",
    "L1_L2_IVDjnt",
    "T12_L1_IVDjnt",
    "T11_T12_IVDjnt",
    "T10_T11_IVDjnt",
    "T9_T10_IVDjnt",
    "T8_T9_IVDjnt",
    "T7_T8_IVDjnt",
    "T6_T7_IVDjnt",
    "T5_T6_IVDjnt",
    "T4_T5_IVDjnt",
    "T3_T4_IVDjnt",
    "T2_T3_IVDjnt",
    "T1_T2_IVDjnt",
    "T1_head_neck",
    "hip_r",
    "knee_r",
    "ankle_r",
    "subtalar_r",
    "mtp_r",
    "hip_l",
    "knee_l",
    "ankle_l",
    "subtalar_l",
    "mtp_l",
    "C7_cerv6",
    "C7_cerv5",
    "C7_cerv4",
    "C7_cerv3",
    "C7_cerv2",
    "C7_cerv1",
    "skull_jaw",
    "C7_skull",
    "sternoclavicular",
    "acromioclavicular",
    "shoulder2",
    "elbow",
    "radioulnar",
    "sternoclavicular_l",
    "acromioclavicular_l",
    "shoulder2_l",
    "elbow_l",
    "radioulnar_l",
    "T1_rib",
]
MIRRORPredictedOSJoints = {
    "hip_r": "hip_l",
    "knee_r": "knee_l",
    "ankle_r": "ankle_l",
    "subtalar_r": "subtalar_l",
    "mtp_r": "mtp_l",
    "sternoclavicular": "sternoclavicular_l",
    "acromioclavicular": "acromioclavicular_l",
    "shoulder2": "shoulder2_l",
    "elbow": "elbow_l",
    "radioulnar": "radioulnar_l",
}
tmp = {}
for k, v in MIRRORPredictedOSJoints.items():
    tmp[v] = k
MIRRORPredictedOSJoints.update(tmp)

LeafJoints = ["radiocarpal",
              "radiocarpal_l"]

PredictedJoints = smplHJoint.copy()
tmp = []
for marker in PredictedMarkers:
    if "virtual_ground" in marker.lower():
        continue
    tmp.append(marker)
PredictedMarkers = tmp
PredictedMarkers = PredictedJoints + PredictedMarkers
SELECTED_JOINTS = PredictedOSJoints + LeafJoints + PredictedMarkers

MIRROR_JOINTS = []
for idx, joint in enumerate(PredictedOSJoints):

    if joint in MIRRORPredictedOSJoints:
        joint = MIRRORPredictedOSJoints[joint]
        MIRROR_JOINTS.append(SELECTED_JOINTS.index(joint))
    else:
        MIRROR_JOINTS.append(SELECTED_JOINTS.index(joint))

MIRROR_JOINTS.append(SELECTED_JOINTS.index("radiocarpal_l"))
MIRROR_JOINTS.append(SELECTED_JOINTS.index("radiocarpal"))
for idx, joint in enumerate(PredictedMarkers):
    if joint[0] == "R" and joint != "ROOT":
        joint = "L" + joint[1:]
        MIRROR_JOINTS.append(SELECTED_JOINTS.index(joint))
    elif joint[0] == "L":
        joint = "R" + joint[1:]
        MIRROR_JOINTS.append(SELECTED_JOINTS.index(joint))
    elif "_R" in joint:
        joint = joint.replace("_R", "_L")
        MIRROR_JOINTS.append(SELECTED_JOINTS.index(joint))
    elif "_L" in joint:
        joint = joint.replace("_L", "_R")
        MIRROR_JOINTS.append(SELECTED_JOINTS.index(joint))
    else:
        MIRROR_JOINTS.append(SELECTED_JOINTS.index(joint))

CONNECTION_TABLE = {
    "L5_S1_IVDjnt": "ground_pelvis",
    "L4_L5_IVDjnt": "L5_S1_IVDjnt",
    "L3_L4_IVDjnt": "L4_L5_IVDjnt",
    "L2_L3_IVDjnt": "L3_L4_IVDjnt",
    "L1_L2_IVDjnt": "L2_L3_IVDjnt",
    "T12_L1_IVDjnt": "L1_L2_IVDjnt",
    "T11_T12_IVDjnt": "T12_L1_IVDjnt",
    "T10_T11_IVDjnt": "T11_T12_IVDjnt",
    "T9_T10_IVDjnt": "T10_T11_IVDjnt",
    "T8_T9_IVDjnt": "T9_T10_IVDjnt",
    "T7_T8_IVDjnt": "T8_T9_IVDjnt",
    "T6_T7_IVDjnt": "T7_T8_IVDjnt",
    "T5_T6_IVDjnt": "T6_T7_IVDjnt",
    "T4_T5_IVDjnt": "T5_T6_IVDjnt",
    "T3_T4_IVDjnt": "T4_T5_IVDjnt",
    "T2_T3_IVDjnt": "T3_T4_IVDjnt",
    "T1_T2_IVDjnt": "T2_T3_IVDjnt",
    "T1_head_neck": "T1_T2_IVDjnt",
    "hip_r": "ground_pelvis",
    "knee_r": "hip_r",
    "ankle_r": "knee_r",
    "subtalar_r": "ankle_r",
    "mtp_r": "subtalar_r",
    "hip_l": "ground_pelvis",
    "knee_l": "hip_l",
    "ankle_l": "knee_l",
    "subtalar_l": "ankle_l",
    "mtp_l": "subtalar_l",
    "C7_cerv6": "T1_head_neck",
    "C7_cerv5": "C7_cerv6",
    "C7_cerv4": "C7_cerv5",
    "C7_cerv3": "C7_cerv4",
    "C7_cerv2": "C7_cerv3",
    "C7_cerv1": "C7_cerv2",
    "skull_jaw": "C7_skull",
    "C7_skull": "skull_jaw",
    "sternoclavicular": "T1_rib",
    "acromioclavicular": "sternoclavicular",
    "shoulder2": "acromioclavicular",
    "elbow": "shoulder2",
    "radioulnar": "elbow",
    "sternoclavicular_l": "T1_rib",
    "acromioclavicular_l": "sternoclavicular_l",
    "shoulder2_l": "acromioclavicular_l",
    "elbow_l": "shoulder2_l",
    "radioulnar_l": "elbow_l",
    "T1_rib": "T1_head_neck",
    "radiocarpal": "radioulnar",
    "radiocarpal_l": "radioulnar_l",
}
CONNECTION_TABLE_INDEX_LIST = []
for k, v in CONNECTION_TABLE.items():
    CONNECTION_TABLE_INDEX_LIST.append([SELECTED_JOINTS.index(k), SELECTED_JOINTS.index(v)])

# weights for PredictedMarkers
PredictedMarkersType = []
for marker in PredictedMarkers:
    if IKTaskSet[marker] == 20:
        PredictedMarkersType.append(20)
    elif IKTaskSet[marker] == 10:
        PredictedMarkersType.append(10)
    elif IKTaskSet[marker] == 1:
        PredictedMarkersType.append(1)
    elif IKTaskSet[marker] == 0:
        PredictedMarkersType.append(0)
'''
# Unique body scale mapping
BODY_SCALE_MAPPING = np.zeros((len(PredictedBones), 3), dtype=np.int16) - 1
scales = DataReader.read_scale_set(scaleSet)
for scaleIdx, scale in enumerate(scales.scales):
    for body, axe in zip(scale.bodies, scale.axes):
        if body in PredictedBones:
            idx = PredictedBones.index(body)
            BODY_SCALE_MAPPING[idx, axe] = scaleIdx
assert np.sum(BODY_SCALE_MAPPING == -1) == 0
'''

OS_Joints_Inside_Cylinder_Body = {
    "ground_pelvis": "Torso",
    "L5_S1_IVDjnt": "Torso",
    "L4_L5_IVDjnt": "Torso",
    "L3_L4_IVDjnt": "Torso",
    "L2_L3_IVDjnt": "Torso",
    "L1_L2_IVDjnt": "Torso",
    "T12_L1_IVDjnt": "Torso",
    "T11_T12_IVDjnt": "Torso",
    "T10_T11_IVDjnt": "Torso",
    "T9_T10_IVDjnt": "Torso",
    "T8_T9_IVDjnt": "Torso",
    "T7_T8_IVDjnt": "Torso",
    "T6_T7_IVDjnt": "Torso",
    "T5_T6_IVDjnt": "Torso",
    "T4_T5_IVDjnt": "Torso",
    "T3_T4_IVDjnt": "Torso",
    "T2_T3_IVDjnt": "Torso",
    "T1_T2_IVDjnt": "Torso",
    "T1_head_neck": "HEAD",
    "hip_r": "RightLeg",
    "knee_r": "RightLeg",
    "ankle_r": "RightShrank",
    "subtalar_r": "RightShrank",
    "mtp_r": "RightFOOT",
    "hip_l": "LeftLeg",
    "knee_l": "LeftLeg",
    "ankle_l": "LeftShrank",
    "subtalar_l": "LeftShrank",
    "mtp_l": "LeftFOOT",
    "C7_cerv6": "HEAD",
    "C7_cerv5": "HEAD",
    "C7_cerv4": "HEAD",
    "C7_cerv3": "HEAD",
    "C7_cerv2": "HEAD",
    "C7_cerv1": "HEAD",
    "skull_jaw": "HEAD",
    "C7_skull": "HEAD",
    "sternoclavicular": "RightArm",
    "acromioclavicular": "RightArm",
    "shoulder2": "RightArm",
    "elbow": "RightArm",
    "radioulnar": "RightForearm",
    "sternoclavicular_l": "LeftArm",
    "acromioclavicular_l": "LeftArm",
    "shoulder2_l": "LeftArm",
    "elbow_l": "LeftArm",
    "radioulnar_l": "LeftForearm",
    "T1_rib": "Torso",
    "radiocarpal": "RightForearm",
    "radiocarpal_l": "LeftForearm"
}
Joints_Inside_Cylinder_Body.update(OS_Joints_Inside_Cylinder_Body)
