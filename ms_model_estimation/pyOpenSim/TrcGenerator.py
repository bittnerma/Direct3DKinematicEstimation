import pickle
from ms_model_estimation.smplh.smplh_vertex_index import smplHJoint, smplHMarker
from ms_model_estimation.smplh.SMPLHModel import SMPLHModel
import numpy as np


class TrcGenerator:
    '''
    A class to generate .trc files(used in OpenSim) from AMASS data or raw marker data.
    '''

    '''
    OpenSim's virtual markers(not SMPL+H vertex position in our research). They are usually estimated by 
    the median point of two markers. They are used in body scaling. 
    '''
    VirtualMarkers = {
        "MASI": ["RASI", "LASI"],
        "MPSI": ["RPSI", "LPSI"],
        "MHIPJ": ["LHIPJ", "RHIPJ"],
        "CENTER": ["RASI", "LASI", "RPSI", "LPSI"],
        "Virtual_LWRIST": ["LIWR", "LOWR"],
        "Virtual_RWRIST": ["RIWR", "ROWR"],
        "Virtual_LAJC": ["LANK", "LANI"],
        "Virtual_RAJC": ["RANK", "RANI"],
        "Virtual_LKJC": ["LKNE", "LKNI"],
        "Virtual_RKJC": ["RKNE", "RKNI"],
        "Virtual_LELC": ["LELB", "LELBIN"],
        "Virtual_RELC": ["RELB", "RELBIN"]
    }

    '''
    Virtual ground markers. They are projected to the ground from markers on foot and ankles. 
    They are used to scale foot.
    '''
    GroundMarkers = {
        "Virtual_Ground_RAJC": "Virtual_RAJC",
        "Virtual_Ground_LAJC": "Virtual_LAJC",
        "Virtual_Ground_RFOOT": "RFOOT",
        "Virtual_Ground_LFOOT": "LFOOT",
        "Virtual_Ground_RTOE": "RTOE",
        "Virtual_Ground_LTOE": "LTOE",
        "Virtual_Ground_RMT1": "RMT1",
        "Virtual_Ground_LMT1": "LMT1",
        "Virtual_Ground_RMT5": "RMT5",
        "Virtual_Ground_LMT5": "LMT5",
        "Virtual_Ground_RHEE": "RHEE",
        "Virtual_Ground_LHEE": "LHEE",
    }

    MotionMarkerIndexTable = {}
    for idx, key in enumerate(smplHMarker):
        MotionMarkerIndexTable[key] = idx
    for idx, key in enumerate(smplHJoint):
        MotionMarkerIndexTable[key] = idx + len(smplHMarker)

    StaticMarkerIndexTable = MotionMarkerIndexTable.copy()
    for key in VirtualMarkers:
        StaticMarkerIndexTable[key] = len(StaticMarkerIndexTable)
    for key in GroundMarkers:
        StaticMarkerIndexTable[key] = len(StaticMarkerIndexTable)

    UsedVerticesIndices = [idx for _, idx in smplHMarker.items()]
    UsedJointsIndices = [idx for _, idx in smplHJoint.items()]

    @staticmethod
    def generate_static_marker_trc_file_from_amass(
            outputPath: str, bdata, origDataStartFrame=1,
            defaultpose=True, eulerAngle=None, addingVirtualMarkers=True, handDown=False,
            DMPL=False, zeroShape=-1, gender="N"
    ):

        vertices, joints = SMPLHModel.get_smplH_vertices_position(
            bdata, defaultpose=defaultpose, eulerAngle=eulerAngle,
            handDown=handDown, DMPL=DMPL, zeroShape=zeroShape, gender=gender
        )

        dataRate = int(bdata['mocap_framerate'])
        cameraRate = int(bdata['mocap_framerate'])
        numFrames = int(0.55 * cameraRate)
        origDataRate = int(bdata['mocap_framerate'])
        origDataStartFrame = origDataStartFrame
        origNumFrames = bdata['poses'].shape[0]

        points = np.concatenate(
            (vertices[TrcGenerator.UsedVerticesIndices, :], joints[TrcGenerator.UsedJointsIndices, :]), axis=0)

        # add OpenSim's virtual Markers
        if addingVirtualMarkers:
            groundValue = np.min(vertices[:, 1])
            points = TrcGenerator.add_virtual_markers(
                points, groundValue, TrcGenerator.VirtualMarkers, TrcGenerator.GroundMarkers,
                TrcGenerator.StaticMarkerIndexTable
            )

        TrcGenerator.write_static_marker_trc_file(
            outputPath, points, TrcGenerator.StaticMarkerIndexTable, dataRate, cameraRate, numFrames,
            points.shape[0],
            origDataRate, origDataStartFrame, origNumFrames
        )

    @staticmethod
    def add_virtual_markers(points, groundValue, virtualMarkers, groundMarkers, staticMarkerIndexTable, ):

        if len(points.shape) == 2:
            for idx, name in enumerate(virtualMarkers):
                position = np.zeros((1, 3))
                for tempM in virtualMarkers[name]:
                    position += points[staticMarkerIndexTable[tempM]]
                position /= len(virtualMarkers[name])
                points = np.concatenate((points, position), axis=0)

            for name, marker in groundMarkers.items():
                position = points[staticMarkerIndexTable[marker]].copy()
                position = np.reshape(position, (1, 3))
                position[0, 1] = groundValue
                points = np.concatenate((points, position), axis=0)

        elif len(points.shape) == 3:

            for idx, name in enumerate(virtualMarkers):
                position = np.zeros((points.shape[0], 1, 3))
                for tempM in virtualMarkers[name]:
                    position += points[:, staticMarkerIndexTable[tempM]:staticMarkerIndexTable[tempM] + 1, :]
                position /= len(virtualMarkers[name])
                points = np.concatenate((points, position), axis=1)

            for name, marker in groundMarkers.items():
                position = points[:, staticMarkerIndexTable[marker]:staticMarkerIndexTable[marker] + 1,
                           :].copy()
                position[:, :, 1] = groundValue
                points = np.concatenate((points, position), axis=1)
        else:
            assert False

        return points

    @staticmethod
    def write_static_marker_trc_file(
            outputPath: str, points: np.array, staticMarkerIndexTable: dict, dataRate: int, cameraRate: int,
            numFrames: int, numMarkers: int, origDataRate: int, origDataStartFrame: int,
            origNumFrames: int
    ):
        '''
        A function converts marker data of T-pose to .trc file.
        :param points: 2D numpy array (the number of markers x 3)
        '''
        unit = "mm"
        convert = 1000

        with open(outputPath, "w") as f:
            f.write("PathFileType\t4\t(X/Y/Z)\tclone_scale.trc\n")
            f.write(
                "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
            f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\n".format(dataRate, cameraRate, numFrames, numMarkers,
                                                                unit, origDataRate, origDataStartFrame, origNumFrames))
            string = "Frame#\tTime\t"
            for k in staticMarkerIndexTable:
                string += str(k) + "\t\t\t"
            string = string[:-1]
            string += "\n"
            f.write(string)

            string = "\t\t"
            for i in range(1, numMarkers + 1):
                string += "X{}\tY{}\tZ{}\t".format(i, i, i)
            string = string[:-1]
            string += "\n"
            f.write(string)

            f.write("\n")

            for i in range(1, numFrames + 1):

                string = "{}\t{:.4f}\t".format(i, (i) * 1 / dataRate)

                for _, index in staticMarkerIndexTable.items():
                    v = points[index] * convert
                    string += "{:.6f}\t{:.6f}\t{:.6f}\t".format(v[0], v[1], v[2])

                string = string[:-1]
                string += "\n"
                f.write(string)

    @staticmethod
    def generate_motion_marker_trc_file_from_amass(
            outputPath: str, bdata, origDataStartFrame=1, eulerAngle=None, saveMarkerLocationsAsPkl=False, gender="N",
            addingVirtualMarkers=True
    ):
        '''
        Generate .trc file given a sequence of SMPL+H body data.
        '''
        dataRate = int(bdata['mocap_framerate'])
        cameraRate = int(bdata['mocap_framerate'])
        numFrames = bdata['poses'].shape[0]
        numMarkers = len(smplHJoint) + len(smplHMarker)
        origDataRate = int(bdata['mocap_framerate'])
        origDataStartFrame = origDataStartFrame
        origNumFrames = numFrames

        MarkerLocations = np.zeros((numFrames, len(TrcGenerator.MotionMarkerIndexTable), 3))
        for fId in range(0, numFrames):
            vertices, joints = SMPLHModel.get_smplH_vertices_position(
                bdata, defaultpose=False, frame=fId, translation=True, eulerAngle=eulerAngle, gender=gender
            )
            point = np.concatenate(
                (vertices[TrcGenerator.UsedVerticesIndices, :], joints[TrcGenerator.UsedJointsIndices, :]), axis=0)
            MarkerLocations[fId, :, :] = point

        if addingVirtualMarkers:
            groundValue = np.min(MarkerLocations[:, :, 1])
            MarkerLocations = TrcGenerator.add_virtual_markers(
                MarkerLocations, groundValue, TrcGenerator.VirtualMarkers, TrcGenerator.GroundMarkers,
                TrcGenerator.StaticMarkerIndexTable
            )
            TrcGenerator.write_motion_marker_trc_file(
                outputPath, MarkerLocations, TrcGenerator.StaticMarkerIndexTable, dataRate, cameraRate, numFrames,
                len(TrcGenerator.StaticMarkerIndexTable), origDataRate, origDataStartFrame, origNumFrames
            )

        else:

            TrcGenerator.write_motion_marker_trc_file(
                outputPath, MarkerLocations, TrcGenerator.MotionMarkerIndexTable, dataRate, cameraRate, numFrames,
                numMarkers, origDataRate, origDataStartFrame, origNumFrames
            )
        if saveMarkerLocationsAsPkl:
            outputFileName = outputPath.split(".")[0]
            pickle.dump(MarkerLocations, open(outputFileName + ".pkl", "wb"))

    @staticmethod
    def write_motion_marker_trc_file(
            outputPath: str, points: np.array, motionMarkerIndexTable, dataRate: int, cameraRate: int,
            numFrames: int, numMarkers: int, origDataRate: int, origDataStartFrame: int,
            origNumFrames: int
    ):
        """
        A function converts a sequence of marker data to .trc file.
        """
        assert len(points.shape) == 3
        unit = "mm"
        convert = 1000

        with open(outputPath, "w") as f:
            f.write("PathFileType\t4\t(X/Y/Z)\n")
            f.write(
                "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
            f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\n".format(dataRate, cameraRate, numFrames, numMarkers,
                                                                unit, origDataRate, origDataStartFrame, origNumFrames))
            string = "Frame#\tTime\t"
            for k in motionMarkerIndexTable:
                string += " " + str(k) + "\t\t\t"
            string = string[:-1]
            string += "\n"
            f.write(string)

            string = "\t\t"
            for i in range(1, numMarkers + 1):
                string += "X{}\tY{}\tZ{}\t".format(i, i, i)
            string = string[:-1]
            string += "\n"
            f.write(string)

            f.write("\n")

            for i in range(1, numFrames + 1):

                string = "{}\t{:.4f}\t".format(i, (i - 1) * 1 / dataRate)

                for _, index in motionMarkerIndexTable.items():
                    v = points[i - 1, index, :] * convert
                    string += "{:.6f}\t{:.6f}\t{:.6f}\t".format(v[0], v[1], v[2])

                string = string[:-1]
                string += "\n"
                f.write(string)
