import opensim
from ms_model_estimation.opensim.utils import set_axes, setFunction, unit_checking
from ms_model_estimation.pyOpenSim.PyOpenSimModel import PyOpenSimModel
from ms_model_estimation.pyOpenSim.ScaleIKSet import ScaleSet, IKSet
import numpy as np
import math
import collections
import time

class OpenSimModel:

    def __init__(
            self, modelPath: str,
            pyOpensimModel: PyOpenSimModel, scaleSet: ScaleSet, scaleIKSet: IKSet, ikSet: IKSet, unit="m",
            prescaling_lockedCoordinates=None, prescaling_unlockedConstraints=None, prescaling_defaultValues=None,
            postscaling_lockedCoordinates=None, postscaling_unlockedConstraints=None, changingParentMarkers=None
    ):

        if prescaling_defaultValues is None:
            prescaling_defaultValues = {}
        if changingParentMarkers is None:
            changingParentMarkers = {}
        if postscaling_unlockedConstraints is None:
            postscaling_unlockedConstraints = []
        if prescaling_unlockedConstraints is None:
            prescaling_unlockedConstraints = []
        if postscaling_lockedCoordinates is None:
            postscaling_lockedCoordinates = []
        if prescaling_lockedCoordinates is None:
            prescaling_lockedCoordinates = []

        pyOpensimModel.bodySet.update_bodiesDict()
        pyOpensimModel.jointSet.update_coordinatesDict()
        pyOpensimModel.jointSet.update_jointsDict()
        pyOpensimModel.markerSet.update_markerDict()
        if scaleIKSet:
            scaleIKSet.update_markerWeightDict()
        if ikSet:
            ikSet.update_markerWeightDict()

        self.modelPath = modelPath
        self.modelFolder = "/".join(modelPath.split("/")[:-1]) + "/"
        self.pyOpensimModel = pyOpensimModel
        self.scaleSet = scaleSet
        self.scaleIKSet = scaleIKSet
        self.ikSet = ikSet
        self.manualScaleSet = None
        self.scaledModelFolder = None
        self.scalingXMLFilePath = None
        self.scaledModelPath = None
        self.scaledStaticMotionFilePath = None
        self.ikXMLFilePath = None
        self.ikMotionFilePath = None
        self.unit = unit_checking(unit)

        # choose marker which weighting is 20 as bonylandmarks
        self.bonyLandMarkers = []
        if self.ikSet:
            for k, w in self.ikSet.markerWeightDict.items():
                if w >= 20:
                    self.bonyLandMarkers.append(k)

        # pre-processing of body scaling
        self.prescaling_lockedCoordinates = prescaling_lockedCoordinates
        self.prescaling_unlockedConstraints = prescaling_unlockedConstraints
        self.prescaling_defaultValues = prescaling_defaultValues

        # post-processing of body scales
        self.postscaling_lockedCoordinates = postscaling_lockedCoordinates
        self.postscaling_unlockedConstraints = postscaling_unlockedConstraints
        self.changingParentMarkers = changingParentMarkers

    def generate_opensim_model(self, baseModelPath=None, name="fullBody"):
        '''
        Generate .osim model from pyOpenSimModel
        '''

        if baseModelPath is not None:
            # Load Model
            model = opensim.Model(baseModelPath)
        else:
            # Create a new model
            model = opensim.Model()
            ground = opensim.Ground()
            frameGeo = opensim.FrameGeometry()
            ground.set_frame_geometry(frameGeo)
            model.set_ground(ground)
            model.set_force_units("N")
            model.set_gravity(opensim.Vec3(0, -9.8066499999999994, 0))
            model.set_length_units("meters")
            model.setName(name)

        self.addBody(model)

        self.addJoint(model)

        self.addMarker(model)

        self.addConstraints(model)

        # Export model
        model.finalizeConnections()
        model.printToXML(self.modelPath)

    def addBody(self, model):

        for body in self.pyOpensimModel.bodySet.bodies:

            name = body.name
            meshs = body.mesh
            scales = body.scale
            massCenter = body.massCenter
            inertia = body.inertia
            mass = body.mass
            representation = body.representation
            if mass == 0:
                mass = 10 ** -25

            massCenter = opensim.Vec3(massCenter[0], massCenter[1], massCenter[2])
            inertia = opensim.Inertia(inertia[0], inertia[1], inertia[2], inertia[3], inertia[4], inertia[5])
            body = opensim.Body()
            body.setName(name)
            body.setMass(mass)
            body.setMassCenter(massCenter)
            body.setInertia(inertia)
            frameGeo = opensim.FrameGeometry()

            body.set_frame_geometry(frameGeo)

            scale = opensim.Vec3(scales[0], scales[1], scales[2])

            for geoObj in meshs:

                if '.' not in geoObj:
                    geoObj += ".obj"
                else:
                    format = geoObj.split('.')[-1]
                    assert format == "vtp" or format == "obj" or format == "stl"

                mesh = opensim.Mesh(geoObj)
                mesh.set_scale_factors(scale)
                mesh.setOpacity(1)
                mesh.setColor(opensim.Vec3(1, 1, 1))
                mesh.setRepresentation(representation)

                body.append_attached_geometry(mesh)

            model.addBody(body)

    def addJoint(self, model):

        opensimBodySet = model.get_BodySet()

        for joint in self.pyOpensimModel.jointSet.joints:

            name = joint.name
            jointType = joint.jointType
            frames = joint.frames
            parentLoc = frames.parentLoc
            parentOrientation = frames.parentOrientation
            childFrame = frames.childFrame
            childLoc = frames.childLoc
            childOrientation = frames.childOrientation

            parentFrame = frames.parentFrame
            if "ground" in parentFrame:
                parents = model.get_ground()
            elif "/" in parentFrame:
                parentFrame = parentFrame.split("/")[-1]
                parents = opensimBodySet.get(parentFrame)
            else:
                parents = opensimBodySet.get(parentFrame)

            parentTrans = opensim.Vec3(parentLoc[0], parentLoc[1], parentLoc[2])
            parentOrient = opensim.Vec3(parentOrientation[0], parentOrientation[1], parentOrientation[2])

            if "ground" in childFrame:
                child = model.get_ground()
            elif "/" in childFrame:
                childFrame = childFrame.split("/")[-1]
                child = opensimBodySet.get(childFrame)
            else:
                child = opensimBodySet.get(childFrame)
            childTrans = opensim.Vec3(childLoc[0], childLoc[1], childLoc[2])
            childOrient = opensim.Vec3(childOrientation[0], childOrientation[1], childOrientation[2])

            if jointType == "WeldJoint":
                joint = opensim.WeldJoint(name, parents, parentTrans, parentOrient, child, childTrans, childOrient)

            elif jointType == "CustomJoint":

                # Define the spatial transform
                coords = {}

                for coordinate in joint.coordinates:
                    coord = opensim.Coordinate(coordinate.name, coordinate.coordinateType, coordinate.defaultValue,
                                               coordinate.minValue, coordinate.maxValue)
                    coord.set_clamped(True)
                    coord.set_locked(coordinate.locked)
                    coords[coordinate.name] = coord

                # Define transform axis and transform function
                TAs = {}
                for spatialTransform in joint.spatialTransform:
                    if spatialTransform.coordinateName:
                        A = opensim.ArrayStr()
                        A.append(spatialTransform.coordinateName)
                        ta = opensim.TransformAxis(A, opensim.Vec3(spatialTransform.axis[0], spatialTransform.axis[1],
                                                                   spatialTransform.axis[2]))
                    else:
                        ta = opensim.TransformAxis()
                        ta.set_axis(
                            opensim.Vec3(spatialTransform.axis[0], spatialTransform.axis[1], spatialTransform.axis[2]))

                    if spatialTransform.funcType == "MultiplierFunction":
                        ta.setFunction(setFunction(spatialTransform.funcType, spatialTransform.funcParameters))
                    else:
                        ta.set_function(setFunction(spatialTransform.funcType, spatialTransform.funcParameters))

                    TAs[spatialTransform.transformName] = ta

                ST = opensim.SpatialTransform()
                ST.set_rotation1(TAs["rotation1"])
                ST.set_rotation2(TAs["rotation2"])
                ST.set_rotation3(TAs["rotation3"])
                ST.set_translation1(TAs["translation1"])
                ST.set_translation2(TAs["translation2"])
                ST.set_translation3(TAs["translation3"])

                joint = opensim.CustomJoint(name, parents, parentTrans, parentOrient, child, childTrans,
                                            childOrient, ST)
                for _, cood in coords.items():
                    joint.append_coordinates(cood)

            elif jointType == "BallJoint":

                # Define the Coordinates
                coords = {}
                # STParameters = parameters["SpatialTransform"]
                for coordinate in joint.coordinates:
                    coord = opensim.Coordinate(coordinate.name, coordinate.coordinateType, coordinate.defaultValue,
                                               coordinate.minValue, coordinate.maxValue)
                    coord.set_clamped(True)
                    coord.set_locked(coordinate.locked)
                    coords[coordinate.name] = coord

                joint = opensim.BallJoint(name, parents, parentTrans, parentOrient, child, childTrans, childOrient)
                index = 0
                for _, cood in coords.items():
                    joint.set_coordinates(index, cood)
                    index += 1

            elif jointType == "PinJoint":

                # Define the Coordinates
                coordinate = joint.coordinates[0]
                coord = opensim.Coordinate(coordinate.name, coordinate.coordinateType, coordinate.defaultValue,
                                           coordinate.minValue, coordinate.maxValue)
                coord.set_locked(coordinate.locked)
                coord.set_clamped(True)

                joint = opensim.PinJoint(name, parents, parentTrans, parentOrient, child, childTrans, childOrient)
                joint.set_coordinates(0, coord)

            else:
                raise Exception("Joint type is not defined")

            model.addJoint(joint)

    def addMarker(self, model):

        opensimBodySet = model.get_BodySet()

        for marker in self.pyOpensimModel.markerSet.markers:

            parentFrame = marker.parentFrame
            relativeLoc = marker.relativeLoc
            location = opensim.Vec3(relativeLoc[0], relativeLoc[1], relativeLoc[2])
            fixed = marker.fixed

            if "ground" in parentFrame:
                parent = model.get_ground()
            elif "/" in parentFrame:
                parentFrame = parentFrame.split("/")[-1]
                parent = opensimBodySet.get(parentFrame)
            else:
                parent = opensimBodySet.get(parentFrame)

            marker = opensim.Marker(marker.name, parent, location)
            marker.set_fixed(fixed)

            model.addMarker(marker)

    def addConstraints(self, model):
        if self.pyOpensimModel.constraintSet:
            for constraint in self.pyOpensimModel.constraintSet.constraints:

                oepnsimConstraint = opensim.CoordinateCouplerConstraint()
                oepnsimConstraint.setName(constraint.name)
                oepnsimConstraint.set_isEnforced(constraint.isEnforced)
                oepnsimConstraint.setDependentCoordinateName(constraint.dependent_coordinate_name)

                temp = opensim.ArrayStr()
                for n in constraint.independent_coordinate_names:
                    temp.append(n)
                oepnsimConstraint.setIndependentCoordinateNames(temp)

                function = setFunction(constraint.funcType, constraint.funcParameters)
                function.setName(constraint.funcName)
                oepnsimConstraint.setFunction(function)

                model.addConstraint(oepnsimConstraint)

    def generate_scale_setup_file(self, path: str, scaleFileName: str, mass: float, height: float, age: float,
                                  modelFile: str, markerFile: str, timeRange: str,
                                  outputModelFilePath: str, outputScaleFilePath: str, outputMotionFilePath: str,
                                  outputMarkerFilePath: str, measurements=True):

        self.scalingXMLFilePath = path
        self.scaledModelPath = outputModelFilePath
        self.scaledStaticMotionFilePath = outputMotionFilePath

        with open(path, 'w') as f:

            f.writelines("<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n")
            f.writelines("<OpenSimDocument Version=\"40000\">\n")
            f.writelines("\t<ScaleTool name=\"" + scaleFileName + "\">\n")
            f.writelines("\t\t<mass> " + str(mass) + " </mass>\n")
            f.writelines("\t\t<height> " + str(height) + " </height>\n")
            f.writelines("\t\t<age> " + str(age) + " </age>\n")

            f.writelines("\t\t<GenericModelMaker name=\"\">\n")
            f.writelines("\t\t\t<model_file>" + modelFile + "</model_file>\n")
            f.writelines("\t\t</GenericModelMaker>\n")

            f.writelines("\t\t<ModelScaler name=\"\">\n")
            f.writelines("\t\t\t<apply> true </apply>\n")
            f.writelines("\t\t\t<scaling_order> manualScale measurements </scaling_order>\n")
            f.writelines("\t\t\t<MeasurementSet name=\"\">\n")
            f.writelines("\t\t\t\t<objects name=\"\">\n")

            if measurements:
                for scale in self.scaleSet.scales:
                    f.writelines("\t\t\t\t\t<Measurement name=\"" + scale.name + "\">\n")
                    f.writelines("\t\t\t\t\t\t<apply> true </apply>\n")
                    f.writelines("\t\t\t\t\t\t<MarkerPairSet name=\"" + scale.name + "\">\n")
                    f.writelines("\t\t\t\t\t\t\t<objects>\n")
                    for a, b in scale.markerPairSet:
                        f.writelines("\t\t\t\t\t\t\t\t<MarkerPair>\n")
                        f.writelines("\t\t\t\t\t\t\t\t\t<markers> " + a + " " + b + " </markers>\n")
                        f.writelines("\t\t\t\t\t\t\t\t</MarkerPair>\n")
                    f.writelines("\t\t\t\t\t\t\t</objects>\n")
                    f.writelines("\t\t\t\t\t\t\t<groups/>\n")
                    f.writelines("\t\t\t\t\t\t</MarkerPairSet>\n")

                    f.writelines("\t\t\t\t\t\t<BodyScaleSet name=\"\">\n")
                    f.writelines("\t\t\t\t\t\t\t<objects>\n")

                    if scale.bodies:
                        for body, axis in zip(scale.bodies, scale.axes):
                            a = set_axes(axis)
                            f.writelines("\t\t\t\t\t\t\t\t<BodyScale name=\"" + body + "\">\n")
                            f.writelines("\t\t\t\t\t\t\t\t\t<axes> " + a + " </axes>\n")
                            f.writelines("\t\t\t\t\t\t\t\t</BodyScale>\n")
                    else:
                        axes = scale.axes[0]
                        a = set_axes(axes)
                        f.writelines("\t\t\t\t\t\t\t\t<BodyScale name=\"" + scale.name + "\">\n")
                        f.writelines("\t\t\t\t\t\t\t\t\t<axes> " + a + " </axes>\n")
                        f.writelines("\t\t\t\t\t\t\t\t</BodyScale>\n")

                    f.writelines("\t\t\t\t\t\t\t</objects>\n")
                    f.writelines("\t\t\t\t\t\t\t<groups/>\n")
                    f.writelines("\t\t\t\t\t\t</BodyScaleSet>\n")

                    f.writelines("\t\t\t\t\t</Measurement>\n")

            f.writelines("\t\t\t\t</objects>\n")
            f.writelines("\t\t\t\t<groups/>\n")
            f.writelines("\t\t\t</MeasurementSet>\n")

            f.writelines("\t\t\t<ScaleSet name=\"\">\n")
            f.writelines("\t\t\t\t<objects name=\"\">\n")

            if self.manualScaleSet:
                for segment, scale in self.manualScaleSet.items():
                    f.writelines("\t\t\t\t\t<Scale>\n")
                    f.writelines("\t\t\t\t\t\t<scales> " + str(scale[0]) + " " + str(scale[1]) + " " + str(
                        scale[2]) + " </scales>\n")
                    f.writelines("\t\t\t\t\t\t<segment> " + segment + " </segment>\n")
                    f.writelines("\t\t\t\t\t\t<apply> true </apply>\n")
                    f.writelines("\t\t\t\t\t</Scale>\n")

            f.writelines("\t\t\t\t</objects>\n")
            f.writelines("\t\t\t\t<groups/>\n")
            f.writelines("\t\t\t</ScaleSet>\n")

            f.writelines("\t\t\t<marker_file> " + markerFile + " </marker_file>\n")
            f.writelines("\t\t\t<time_range> " + timeRange + " </time_range>\n")
            f.writelines("\t\t\t<preserve_mass_distribution> true </preserve_mass_distribution>\n")
            f.writelines("\t\t\t<output_model_file> " + outputModelFilePath + " </output_model_file>\n")
            f.writelines("\t\t\t<output_scale_file> " + outputScaleFilePath + " </output_scale_file>\n")

            f.writelines("\t\t</ModelScaler>\n")

            f.writelines("\t\t<MarkerPlacer name=\"\">\n")
            f.writelines("\t\t\t<apply> true </apply>\n")
            f.writelines("\t\t\t<IKTaskSet name=\"\">\n")
            f.writelines("\t\t\t\t<objects name=\"\">\n")
            for markerWeight in self.scaleIKSet.markerWeight:
                weight = markerWeight.weight
                if not weight:
                    weight = 0
                f.writelines("\t\t\t\t\t<IKMarkerTask name=\"" + markerWeight.name + "\">\n")
                f.writelines("\t\t\t\t\t\t<apply> true </apply>\n")
                f.writelines("\t\t\t\t\t\t<weight> " + str(weight) + " </weight>\n")
                f.writelines("\t\t\t\t\t</IKMarkerTask>\n")
            f.writelines("\t\t\t\t</objects>\n")
            f.writelines("\t\t\t\t<groups/>\n")
            f.writelines("\t\t\t</IKTaskSet>\n")

            f.writelines("\t\t\t<marker_file> " + markerFile + " </marker_file>\n")
            f.writelines("\t\t\t<time_range> " + timeRange + " </time_range>\n")
            f.writelines("\t\t\t<output_motion_file> " + outputMotionFilePath + " </output_motion_file>\n")
            f.writelines("\t\t\t<output_model_file> " + outputModelFilePath + " </output_model_file>\n")
            f.writelines("\t\t\t<output_marker_file> " + outputMarkerFilePath + " </output_marker_file>\n")
            f.writelines("\t\t\t<max_marker_movement> -1 </max_marker_movement>\n")

            f.writelines("\t\t</MarkerPlacer>\n")

            f.writelines("\t</ScaleTool>\n")
            f.writelines("</OpenSimDocument>\n")

    def scaling(self, post_scaling_processing=True):
        '''
        Conduct body scaling
        '''
        assert self.scalingXMLFilePath is not None
        assert self.scaledModelPath is not None

        scaleTool = opensim.ScaleTool(self.scalingXMLFilePath)

        try:
            success = scaleTool.run()
        except:
            raise Exception("Body scaling fails!")

        if post_scaling_processing:
            self.post_scaling_processing()
            self.change_marker_parent_frame()

    def generate_ik_setup_file(self, path: str, templateModelFileName: str, markerFile: str, timeRange: str,
                               outputMotionFilePath: str):
        '''
        Generate .xml file to set up inverse kinematics
        :param path: output .xml file path
        :param outputMotionFilePath: relative path of output .mot file.
        :return:
        '''
        self.ikXMLFilePath = path
        self.ikMotionFilePath = outputMotionFilePath
        self.scaledModelFolder = "/".join(path.split("/")[:-1]) + "/"

        with open(path, 'w') as f:
            f.writelines("<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n")
            f.writelines("<OpenSimDocument Version=\"40000\">\n")
            f.writelines("\t<InverseKinematicsTool name=\"\">\n")

            f.writelines("\t\t<model_file> " + templateModelFileName + " </model_file>\n")
            f.writelines("\t\t<constraint_weight> 20 </constraint_weight>\n")
            f.writelines("\t\t<accuracy> 1e-005 </accuracy>\n")

            f.writelines("\t\t<IKTaskSet name=\"\">\n")
            f.writelines("\t\t\t<objects name=\"\">\n")
            for markerWeight in self.ikSet.markerWeight:
                weight = markerWeight.weight
                if weight is None:
                    weight = 0
                f.writelines("\t\t\t\t<IKMarkerTask name=\"" + markerWeight.name + "\">\n")
                f.writelines("\t\t\t\t\t<apply> true </apply>\n")
                f.writelines("\t\t\t\t\t<weight> " + str(weight) + " </weight>\n")
                f.writelines("\t\t\t\t</IKMarkerTask>\n")
            f.writelines("\t\t\t</objects>\n")
            f.writelines("\t\t\t<groups/>\n")
            f.writelines("\t\t</IKTaskSet>\n")

            f.writelines("\t\t<marker_file> " + markerFile + " </marker_file>\n")
            f.writelines("\t\t<coordinate_file></coordinate_file>\n")
            f.writelines("\t\t<time_range> " + timeRange + " </time_range>\n")
            f.writelines("\t\t<output_motion_file> " + outputMotionFilePath + " </output_motion_file>\n")
            f.writelines("\t\t<report_errors>true</report_errors>\n")
            f.writelines("\t\t<report_marker_locations>true</report_marker_locations>\n")
            f.writelines("\t</InverseKinematicsTool>\n")
            f.writelines("</OpenSimDocument>\n")

    def inverseKinematics(self):
        '''
        Conduct inverse kinematics.
        If the model has not conducted body scaling before, opensim would return errors.
        '''
        assert self.ikXMLFilePath is not None

        ik = opensim.InverseKinematicsTool(self.ikXMLFilePath, True)
        success = ik.run()
        if not success:
            raise Exception("IK fails!")

    def read_ik_mot_file(self, filePath: str) -> dict:
        '''
        Read .mot file of ik results, and save coordinates to a dictionary.
        :param filePath: .mot file path
        :return: {coordinate_name: a sequence of value}
        '''
        inDegrees = False
        coordinates = collections.defaultdict(list)

        with open(filePath, 'r') as f:

            inf = f.readline()
            # Only support to read Coordinates
            assert inf == "Coordinates\n"

            while True:
                text = f.readline()
                if "inDegrees" in text:
                    inDegrees = False if "no" in text.split("=")[-1] else True
                elif "endheader" in text:
                    break

            keys = f.readline().split("\t")
            keys = [k.split('\n')[0] for k in keys]

            while True:
                text = f.readline()
                if not text:
                    break
                text = text.split("\t")
                for k, v in zip(keys, text):
                    v = float(v.split("\n")[0])
                    coordinates[k].append(v)

        # Convert to radius
        if inDegrees:
            for coord in coordinates:
                if coord not in self.pyOpensimModel.jointSet.coordinatesDict:
                    continue
                # Coordinate is rotational motion
                tempJoint, k = self.pyOpensimModel.jointSet.coordinatesDict[coord]
                Jtype = tempJoint.coordinates[k].coordinateType
                if Jtype == 1:
                    temp = coordinates[coord]
                    temp = [t / 180 * math.pi for t in temp]
                    coordinates[coord] = temp

        return coordinates

    def pre_scaling_processing(self):

        opensimModel = opensim.Model(self.modelPath)

        for coordinate in self.pyOpensimModel.jointSet.coordinatesDict:
            joint, coordinateIdx = self.pyOpensimModel.jointSet.coordinatesDict[coordinate]

            # set joint angle to default value
            if coordinate in self.prescaling_defaultValues:
                opensimModel.get_JointSet().get(joint.name).get_coordinates(coordinateIdx).set_default_value(
                    self.prescaling_defaultValues[coordinate])

            # lock joint angles
            if coordinate in self.prescaling_lockedCoordinates:
                opensimModel.get_JointSet().get(joint.name).get_coordinates(coordinateIdx).set_locked(True)
            else:
                opensimModel.get_JointSet().get(joint.name).get_coordinates(coordinateIdx).set_locked(False)
            opensimModel.get_JointSet().get(joint.name).upd_coordinates(coordinateIdx)

        # unlock constraints
        for constraintName in self.prescaling_unlockedConstraints:
            opensimModel.get_ConstraintSet().get(constraintName).set_isEnforced(False)

        opensimModel.upd_JointSet()
        opensimModel.finalizeConnections()
        opensimModel.printToXML(self.modelPath)

    def post_scaling_processing(self):
        '''
        Set the joint angle with the angle under T-pose and lock the joint angle.
        '''
        opensimModel = opensim.Model(self.modelFolder + self.scaledModelPath)
        #print(self.modelFolder + self.scaledStaticMotionFilePath)
        staticPose = OpenSimModel.read_scaled_mot_file(self.modelFolder + self.scaledStaticMotionFilePath)

        for coordinate in self.pyOpensimModel.jointSet.coordinatesDict:
            joint, coordinateIdx = self.pyOpensimModel.jointSet.coordinatesDict[coordinate]
            opensimModel.get_JointSet().get(joint.name).get_coordinates(coordinateIdx).set_default_value(
                staticPose[coordinate])

            # set joint angle to angle of T-pose and lock joint angle
            if coordinate in self.postscaling_lockedCoordinates:
                opensimModel.get_JointSet().get(joint.name).get_coordinates(coordinateIdx).set_locked(True)
            else:
                opensimModel.get_JointSet().get(joint.name).get_coordinates(coordinateIdx).set_locked(False)
            opensimModel.get_JointSet().get(joint.name).upd_coordinates(coordinateIdx)

        for constraintName in self.postscaling_unlockedConstraints:
            opensimModel.get_ConstraintSet().get(constraintName).set_isEnforced(False)

        opensimModel.upd_JointSet()
        opensimModel.finalizeConnections()
        opensimModel.printToXML(self.modelFolder + self.scaledModelPath)

    def change_marker_parent_frame(self):
        '''
        Change anchored body of markers.
        '''
        opensimModel = opensim.Model(self.modelFolder + self.scaledModelPath)
        state = opensimModel.initSystem()

        for changingParentMarker in self.changingParentMarkers:

            # get kinematic transform (to the ground) of the old anchored body
            originalParent = self.pyOpensimModel.markerSet.markerDict[changingParentMarker].parentFrame.split("/")[-1]
            originalTransform = opensimModel.get_BodySet().get(originalParent).getTransformInGround(state)
            originalTransform = OpenSimModel.generate_homogenous_matrix(originalTransform)

            # get relative location of the marker
            markerLocation = opensimModel.get_MarkerSet().get(changingParentMarker).get_location()
            markerLocation = np.array([markerLocation[0], markerLocation[1], markerLocation[2], 1])
            markerLocation = np.reshape(markerLocation, (4, 1))
            markerLocation = np.dot(originalTransform, markerLocation)

            # get kinematic transform (to the ground) of the new anchored body
            newParentTransform = opensimModel.get_BodySet().get(
                self.changingParentMarkers[changingParentMarker]).getTransformInGround(
                state)
            newParentTransform = OpenSimModel.generate_homogenous_matrix(newParentTransform)

            # calculate new relative location
            newParentTransform = np.linalg.inv(newParentTransform)
            markerLocation = np.dot(newParentTransform, markerLocation)
            markerLocation = list(markerLocation[:3, 0])

            # change the anchored body and set the new relative location
            marker = opensimModel.get_MarkerSet().get(changingParentMarker)
            marker.setParentFrame(opensimModel.get_BodySet().get(self.changingParentMarkers[changingParentMarker]))
            marker.set_location(opensim.Vec3(markerLocation[0], markerLocation[1], markerLocation[2]))

            # update marker set
            opensimModel.upd_MarkerSet()

        # export .osim model
        opensimModel.finalizeConnections()
        opensimModel.printToXML(self.modelFolder + self.scaledModelPath)

    @staticmethod
    def read_trc_file(filePath: str) -> dict:

        assert filePath.split(".")[-1] == "trc"

        with open(filePath, "r") as f:
            _ = f.readline()
            _ = f.readline()
            inf = f.readline().split()
            numFrames = int(inf[2])
            numMarkers = int(inf[3])
            unit = unit_checking(inf[4])

            markerNames = f.readline()[:-1].split()[1:]
            assert len(markerNames) == (numMarkers + 1)

            _ = f.readline()
            _ = f.readline()

            markerLocations = collections.defaultdict(list)
            for _ in range(numFrames):

                positions = f.readline()[:-1].split()[1:]
                positions = [float(p) for p in positions]
                # convert to meter except time
                positions[1:] = [p * unit for p in positions[1:]]

                assert len(positions) == (numMarkers * 3 + 1)

                markerLocations["time"].append(positions[0])
                for i, name in zip(range(1, len(positions), 3), markerNames[1:]):
                    markerLocations[name].append(positions[i:i + 3])

            return markerLocations

    @staticmethod
    def read_ik_marker_location(filePath: str):

        assert filePath[-3:] == "sto"

        with open(filePath, 'r') as f:

            inf = f.readline()
            # Only support to .sto file form Model Marker Locations from IK
            assert inf == "Model Marker Locations from IK\n"

            nRows = 0
            nCols = 0

            while True:
                text = f.readline()
                if "nRows" in text:
                    nRows = int(text.split("=")[-1].split("\n")[0])
                elif "nColumns" in text:
                    nCols = int(text.split("=")[-1].split("\n")[0])
                elif "endheader" in text:
                    break

            assert (nCols - 1) % 3 == 0

            markerNames = f.readline().split()
            keys = [markerNames[i].split("\n")[0][:-3] for i in range(1, nCols, 3)]
            markersLocations = collections.defaultdict(list)

            for i in range(nRows):
                text = f.readline().split()

                for idx, markerName in enumerate(keys):
                    locations = text[idx * 3 + 1:idx * 3 + 4]
                    locations = [float(t.split("\n")[0]) for t in locations]
                    markersLocations[markerName].append(locations)

            return markersLocations

    @staticmethod
    def read_scaled_mot_file(filePath: str) -> dict:

        assert filePath[-3:] == "mot"

        with open(filePath, 'r') as f:

            assert "static pose" in f.readline()

            nRows = 0
            nCols = 0
            inDegrees = False

            while True:
                text = f.readline()
                if "nRows" in text:
                    nRows = int(text.split("=")[-1].split("\n")[0])
                elif "inDegrees" in text:
                    inDegrees = False if "no" in text.split("=")[-1] else True
                elif "nColumns" in text:
                    nCols = int(text.split("=")[-1].split("\n")[0])
                elif "endheader" in text:
                    break

            keys = f.readline().split("\t")
            values = f.readline().split("\t")

        assert len(keys) == len(values)

        coordinates = dict()
        for k, v in zip(keys, values):
            if "speed" in k or "time" in k:
                continue

            v = float(v.split("\n")[0])
            if inDegrees:
                v = v / 180 * math.pi

            coordinates[k.split('/')[-2].split("\n")[0]] = v

        return coordinates

    @staticmethod
    def generate_homogenous_matrix(opensimTransform):
        rotation = opensimTransform.R()
        rotation = rotation.asMat33()
        transform = opensimTransform.T()

        matrix = np.eye(4)
        for i in range(3):
            for j in range(3):
                matrix[i][j] = float(rotation.get(i, j))
        for i in range(3):
            matrix[i, 3] = transform[i]

        return matrix

    def write_ik_mot_file(self, fielPath, timeList, data):

        assert fielPath.split(".")[-1] == "mot"
        with open(fielPath, "w") as f:
            f.write(f'Coordinates\n')
            f.write(f'version=1\n')
            f.write(f'nRows={data.shape[0]}\n')
            f.write(f'nColumns={data.shape[1]+1}\n')
            f.write(f'inDegrees=yes\n')
            f.write(f'endheader\n')

            tmp = "time\t"
            keys = list(self.pyOpensimModel.jointSet.coordinatesDict)
            for name in keys:
                tmp = tmp + name + "\t"
            tmp = tmp + "\n"
            f.write(tmp)

            for i in range(data.shape[0]):
                tmp = f'\t\t{timeList[i]}\t'
                for j in range(data.shape[1]):
                    tmp = tmp + f'{data[i, j]}\t'
                tmp = tmp + "\n"
                f.write(tmp)
