import opensim
from ms_model_estimation.pyOpenSim.BodySet import *
from ms_model_estimation.pyOpenSim.JointSet import *
from ms_model_estimation.pyOpenSim.MarkerSet import *
from ms_model_estimation.pyOpenSim.ScaleIKSet import *
from ms_model_estimation.pyOpenSim.ConstraintSet import ConstraintSet, Constraint
from ms_model_estimation.pyOpenSim.PyOpenSimModel import PyOpenSimModel
from ms_model_estimation.pyOpenSim.MarkerSetTransform import *
import numpy as np


class OpenSimDataReader:

    @staticmethod
    def read_scale_set(table: dict):
        scaleSet = ScaleSet([])
        for name, paramters in table.items():
            if "bodies" not in paramters:
                bodies = [name]
            else:
                bodies = paramters["bodies"]

            scale = Scale(name, paramters["MarkerPairSet"], bodies, paramters["axes"])
            if len(bodies) != len(paramters["axes"]):
                scale.create_axes()
            scaleSet.scales.append(scale)

        return scaleSet

    @staticmethod
    def read_ik_set(table: dict):

        ikset = IKSet([])
        for name, weights in table.items():
            markerWeight = MarkerWeight(name, weights)
            ikset.markerWeight.append(markerWeight)

        return ikset

    @staticmethod
    def read_opensim_model(opensimModelPath: str, getBodyInGround=False):
        '''
        Convert .osim model to pyOpenSimModel
        :param opensimModelPath: the file path for .osim model.
        :param getBodyInGround: read body in the ground.
        :return: pyOpenSimModel
        '''
        assert opensimModelPath[-5:] == ".osim"

        opensimModel = opensim.Model(opensimModelPath)

        # read body set
        bodySet = OpenSimDataReader.read_opensim_bodySet(opensimModel)
        if getBodyInGround:
            bodyInGround = OpenSimDataReader.read_opensim_ground(opensimModel)
            if bodyInGround is not None:
                bodySet.bodies.append(bodyInGround)
                bodySet.update_bodiesDict()

        # read joint set
        jointSet = OpenSimDataReader.read_opnesim_jointSet(opensimModel)

        # read marker set
        markerSet = OpenSimDataReader.read_opensim_markerSet(opensimModel)

        # read constraint set
        constraintSet = OpenSimDataReader.read_opensim_constraint_set(opensimModelPath)

        return PyOpenSimModel(bodySet, jointSet, markerSet, constraintSet)

    @staticmethod
    def read_opensim_ground(opensimModel):

        opensimGround = opensimModel.getGround()
        name = opensimGround.getName()
        inertia = [0., 0., 0., 0., 0., 0.]

        mass = 0.
        massCenter = [0., 0., 0.]

        bodyScale = None
        meshes = []
        temp = opensimGround.getPropertyByIndex(2).toString()
        if "no objects" in temp.lower():
            return None

        for i in range(len(temp.split())):
            mesh = opensimGround.get_attached_geometry(i)
            meshFile = mesh.getPropertyByName("mesh_file").toString()
            meshes.append(meshFile)

            scale = mesh.get_scale_factors()
            scale = OpenSimDataReader.convert_opensim_object_to_list(scale)
            if bodyScale is None:
                bodyScale = scale
            else:
                assert bodyScale == scale

        body = Body(name, meshes, mass, massCenter, inertia, bodyScale)
        return body

    @staticmethod
    def read_opensim_bodySet(opensimModel):
        '''
        Read bodySet in a .osim model.
        :return: a bodySet in pyOpenSimModel.
        '''
        opensimBodySet = opensimModel.get_BodySet()
        bodies = []
        for idx in range(opensimBodySet.getSize()):

            body = opensimBodySet.get(idx)
            name = body.getName()
            inertia = body.get_inertia()
            inertia = OpenSimDataReader.convert_opensim_object_to_list(inertia)

            mass = body.get_mass()
            massCenter = body.get_mass_center()
            massCenter = OpenSimDataReader.convert_opensim_object_to_list(massCenter)

            bodyScale = None
            meshes = []
            temp = body.getPropertyByIndex(2).toString()
            if "no objects" in temp.lower():
                bodyScale = [1., 1., 1.]
                body = Body(name, meshes, mass, massCenter, inertia, bodyScale)
            else:
                for i in range(len(temp.split())):
                    mesh = body.get_attached_geometry(i)
                    meshFile = mesh.getPropertyByName("mesh_file").toString()
                    meshes.append(meshFile)

                    scale = mesh.get_scale_factors()
                    scale = OpenSimDataReader.convert_opensim_object_to_list(scale)
                    if bodyScale is None:
                        bodyScale = scale
                    else:
                        assert bodyScale == scale

                body = Body(name, meshes, mass, massCenter, inertia, bodyScale)
            bodies.append(body)
        pyBodySet = BodySet(bodies)
        return pyBodySet

    @staticmethod
    def read_opnesim_jointSet(opensimModel):
        opensimJointSet = opensimModel.get_JointSet()
        pyJoints = []

        for idx in range(opensimJointSet.getSize()):
            joint = opensimJointSet.get(idx)

            jointName = joint.getName()
            jointType = joint.getConcreteClassName()

            opensimParentFrame = joint.get_frames(0)
            parentOrientaion = opensimParentFrame.get_orientation()
            parentOrientaion = OpenSimDataReader.convert_opensim_object_to_list(parentOrientaion)
            parentLocation = opensimParentFrame.get_translation()
            parentLocation = OpenSimDataReader.convert_opensim_object_to_list(parentLocation)

            opensimChildFrame = joint.get_frames(1)
            childOrientaion = opensimChildFrame.get_orientation()
            childOrientaion = OpenSimDataReader.convert_opensim_object_to_list(childOrientaion)

            childLocation = opensimChildFrame.get_translation()
            childLocation = OpenSimDataReader.convert_opensim_object_to_list(childLocation)

            parentPath = opensimParentFrame.getAbsolutePathString()
            parentPath = OpenSimDataReader.convert_to_bodyset_path(parentPath)
            childPath = opensimChildFrame.getAbsolutePathString()
            childPath = OpenSimDataReader.convert_to_bodyset_path(childPath)

            pyFrame = Frame(parentPath, parentLocation, parentOrientaion, childPath, childLocation, childOrientaion)

            temp = joint.getPropertyByIndex(3).toString()
            if "no object" in temp.lower():
                pyCoordinates = None
            else:
                pyCoordinates = []
                for i in range(len(temp.split())):
                    opensimCoordinate = joint.get_coordinates(i)
                    defaultValue = opensimCoordinate.get_default_value()
                    coordinateName = opensimCoordinate.getName()
                    tempMinValue = opensimCoordinate.get_range(0)
                    tempMaxValue = opensimCoordinate.get_range(1)
                    coordinateType = opensimCoordinate.getMotionType()
                    locked = opensimCoordinate.get_locked()

                    minValue = min(tempMinValue, tempMaxValue)
                    maxValue = max(tempMinValue, tempMaxValue)

                    if defaultValue < minValue or defaultValue > maxValue:
                        defaultValue = minValue

                    coordinate = Coordinate(coordinateName, coordinateType, defaultValue, minValue, maxValue,
                                            locked=locked)
                    pyCoordinates.append(coordinate)

            pySpatialTransforms = None
            otherProperties = None
            if jointType == "CustomJoint":
                spatialTransforms = joint.getPropertyByName("SpatialTransform").getValueAsObject()
                pySpatialTransforms = []

                for i in range(6):
                    spatialTransform = spatialTransforms.getPropertyByIndex(i)

                    # Transform Name
                    transformName = spatialTransform.getValueAsObject().getName()

                    # Coordinate Name
                    coordinateName = spatialTransform.getValueAsObject().getPropertyByIndex(0).toString()
                    coordinateName = coordinateName[1:-1]

                    # Axis
                    axis = spatialTransform.getValueAsObject().getPropertyByIndex(1).toString()
                    axis = axis[1:-1].split()
                    axis = [float(a) for a in axis]

                    if coordinateName:

                        # Function Type
                        funcType = spatialTransform.getValueAsObject().getPropertyByIndex(2).toString()
                        paramters = spatialTransform.getValueAsObject().getPropertyByIndex(
                            2).getValueAsObject().getPropertyByIndex(0)
                        paramters = paramters.toString()

                        if funcType == "LinearFunction":
                            paramters = paramters[1:-1].split()
                            paramters = [float(p) for p in paramters]

                            # convert to 1 0
                            if len(paramters) == 2:
                                if paramters[0] == -1 and paramters[1] == 0:
                                    axis[0] *= -1
                                    axis[1] *= -1
                                    axis[2] *= -1
                                    # paramters= [ 1, 0]
                                    pass
                                elif paramters[0] == 1 and paramters[1] == 0:
                                    pass
                                else:
                                    assert False, print("Func paramters will cause bugs in other classes."
                                                        " Parameters of linear func must be [-1,0] or [1,0]")

                        elif funcType == "SimmSpline":
                            paramtersX = spatialTransform.getValueAsObject().getPropertyByIndex(
                                2).getValueAsObject().getPropertyByIndex(0)
                            paramtersX = paramtersX.toString()
                            paramtersY = spatialTransform.getValueAsObject().getPropertyByIndex(
                                2).getValueAsObject().getPropertyByIndex(1)
                            paramtersY = paramtersY.toString()
                            paramters = [paramtersX[1:-1], paramtersY[1:-1]]

                        elif funcType == "MultiplierFunction":
                            paramtersX = spatialTransform.getValueAsObject().getPropertyByIndex(2). \
                                getValueAsObject().getPropertyByIndex(0).getValueAsObject().getPropertyByIndex(
                                0).toString()
                            if paramtersX == '0':
                                funcType = "Constant"
                                paramters = [0]
                            else:
                                paramtersY = spatialTransform.getValueAsObject().getPropertyByIndex(2). \
                                    getValueAsObject().getPropertyByIndex(0).getValueAsObject().getPropertyByIndex(
                                    1).toString()

                                scale = spatialTransform.getValueAsObject().getPropertyByIndex(
                                    2).getValueAsObject().getPropertyByIndex(1).toString()

                                paramters = [paramtersX[1:-1], paramtersY[1:-1], float(scale)]
                        else:
                            funcType = "LinearFunction"
                            paramters = [1, 0]
                    else:
                        funcType = "Constant"
                        paramters = [0]

                    pySpatialTransform = SpatialTransform(transformName, axis, coordinateName, funcType, paramters)
                    pySpatialTransforms.append(pySpatialTransform)

            pyJoint = Joint(jointName, jointType, pyFrame, pyCoordinates, pySpatialTransforms,
                            otherProperties=otherProperties)
            pyJoints.append(pyJoint)
        pyJointSet = JointSet(pyJoints)
        return pyJointSet

    @staticmethod
    def read_opensim_markerSet(opensimModel):
        opensimMarkerSet = opensimModel.get_MarkerSet()
        pyMarkers = []
        for idx in range(opensimMarkerSet.getSize()):
            marker = opensimMarkerSet.get(idx)
            fixed = marker.get_fixed()
            location = marker.get_location()
            location = OpenSimDataReader.convert_opensim_object_to_list(location)
            name = marker.getName()
            parentFrame = marker.getParentFrameName().split("/")[-1]

            marker = Marker(name, parentFrame, location, fixed=fixed)
            pyMarkers.append(marker)
        pyMarkerSet = MarkerSet(pyMarkers)
        return pyMarkerSet

    @staticmethod
    def convert_opensim_object_to_list(obj):
        return [float(obj.get(i)) for i in range(len(obj))]

    @staticmethod
    def convert_to_bodyset_path(path):
        path = path.split('/')[-1]
        path = path.replace('_offset', '')

        if 'ground' in path:
            path = '/ground'

        return path

    @staticmethod
    def read_opensim_constraint_set(opensimModelPath):

        constraints = []
        notext = False
        with open(opensimModelPath, 'r') as f:
            while True:
                text = f.readline()
                if not text:
                    notext = True
                    break
                if "ConstraintSet" in text:
                    break

            # read <objects>
            while not notext:
                text = f.readline()
                if "<objects>" in text:
                    break

            while not notext:
                ending = False

                # Find name in <CoordinateCouplerConstraint name="xxxx">
                while True:
                    title = f.readline()
                    if "</objects>" in title:
                        ending = True
                        break
                    elif "<CoordinateCouplerConstraint" in title:
                        title = title.replace("<CoordinateCouplerConstraint", "")
                        title = title.replace(">", "")
                        title = title.replace("name", "")
                        title = title.replace('"', "")
                        title = title.replace('=', "")
                        title = title.split()
                        constraintName = title[0]
                        break
                if ending:
                    break

                # Find isEnforced in <isEnforced>true</isEnforced>
                while True:
                    text = f.readline()
                    if "<isEnforced>" in text:
                        text = text.replace("<isEnforced>", "")
                        text = text.replace("</isEnforced>", "")
                        text = text.split()
                        isEnforced = text[0]
                        isEnforced = True if isEnforced.lower() == "true" else False
                        break

                # Find <coupled_coordinates_function>
                while True:
                    text = f.readline().split()
                    if "<coupled_coordinates_function>" in text[0]:
                        break

                # Find functionType and functionName
                text = f.readline().split()
                funcType = text[0][1:]
                funcName = text[1].split('"')[1]

                if "SimmSpline" == funcType:
                    # Find xValue and yValue
                    xValue = f.readline().split()
                    if ">" in xValue[-1]:
                        xValue[-1] = xValue[-1][:-4]
                    xValue = xValue[1:]
                    xValue = " ".join(xValue)

                    yValue = f.readline().split()
                    if ">" in yValue[-1]:
                        yValue[-1] = yValue[-1][:-4]
                    yValue = yValue[1:]
                    yValue = " ".join(yValue)
                else:
                    assert False

                # Find <coupled_coordinates_function>
                independent_coordinate_names = []
                while True:
                    text = f.readline()
                    if "<independent_coordinate_names>" in text:
                        text = text.replace("<independent_coordinate_names>", "")
                        text = text.replace("</independent_coordinate_names>", "")
                        text = text.split()
                        for name in text:
                            independent_coordinate_names.append(name)
                        break

                while True:
                    text = f.readline()
                    if "<dependent_coordinate_name>" in text:
                        text = text.replace("<dependent_coordinate_name>", "")
                        text = text.replace("</dependent_coordinate_name>", "")
                        text = text.split()
                        dependent_coordinate_name = text[0]
                        break

                constraint = Constraint(constraintName, isEnforced, funcType, [xValue, yValue], funcName,
                                        independent_coordinate_names, dependent_coordinate_name)
                constraints.append(constraint)

        return ConstraintSet(constraints)
