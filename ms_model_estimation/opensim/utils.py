import numpy as np
import opensim
#import vtk
import os
import math


def setFunction(funcType: str, values: list):
    if funcType == "LinearFunction":

        assert len(values) == 2

        a = opensim.ArrayDouble()
        for v in values:
            a.append(v)

        return opensim.LinearFunction(a)

    elif funcType == "Constant":

        assert len(values) == 1

        return opensim.Constant(values[0])

    elif funcType == "SimmSpline":

        assert len(values) == 2

        Xstring = values[0]
        Ystring = values[1]

        X = []
        for x in Xstring.split(" "):
            if not x:
                continue
            x = float(x)
            X.append(x)

        Y = []
        for y in Ystring.split(" "):
            if not y:
                continue
            y = float(y)
            Y.append(y)

        assert len(X) == len(Y)

        func = opensim.SimmSpline()
        for x, y in zip(X, Y):
            func.addPoint(x, y)

        return func

    elif funcType == "MultiplierFunction":
        Xstring = values[0]
        Ystring = values[1]
        scale = values[2]

        X = []
        for x in Xstring.split(" "):
            if not x:
                continue
            x = float(x)
            X.append(x)

        Y = []
        for y in Ystring.split(" "):
            if not y:
                continue
            y = float(y)
            Y.append(y)

        assert len(X) == len(Y)

        func = opensim.SimmSpline()
        for x, y in zip(X, Y):
            func.addPoint(x, y)

        return opensim.MultiplierFunction(func, scale)

    else:
        assert False


def set_axes(axes: list) -> str:
    a = []
    if 0 in axes:
        a.append("X")
    if 1 in axes:
        a.append("Y")
    if 2 in axes:
        a.append("Z")
    a = " ".join(a)

    return a

'''
def convert_vtp_to_obj(folder: str, meshFolder: str, meshList: list, prefix="videoMuscle_"):
    for file in meshList:
        if '.' not in file:
            file += '.vtp'
            meshFormat = "vtp"
        else:
            meshFormat = file.split('.')[-1]

        name = prefix + file.split('.')[0]

        inputPath = os.path.join(meshFolder, file)
        outputPath = os.path.join(folder, name)

        if meshFormat == "obj":
            if meshFolder != folder:
                os.system('copy ' + inputPath + ' ' + outputPath + '.obj')

        else:
            if meshFormat == "vtp":
                reader = vtk.vtkXMLPolyDataReader()
            elif meshFormat == "stl":
                reader = vtk.vtkSTLReader()

            reader.SetFileName(inputPath)
            reader.Update()
            mapper = vtk.vtkPolyDataMapper()
            if vtk.VTK_MAJOR_VERSION <= 5:
                mapper.SetInput(reader.GetOutput())
            else:
                mapper.SetInputConnection(reader.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # Create a rendering window and renderer
            ren = vtk.vtkRenderer()
            renWin = vtk.vtkRenderWindow()
            renWin.AddRenderer(ren)

            # Create a renderwindowinteractor
            iren = vtk.vtkRenderWindowInteractor()
            iren.SetRenderWindow(renWin)

            # Assign actor to the renderer
            ren.AddActor(actor)
            renWin.Render()

            # Export
            exporter = vtk.vtkOBJExporter()
            exporter.SetRenderWindow(renWin)
            exporter.SetFilePrefix(outputPath)  # create mtl and obj file.
            exporter.Write()

            os.remove(folder + "/" + name + ".mtl")
'''

def get_time_range(dataPath: str):
    bdata = np.load(dataPath)
    DataRate = int(bdata['mocap_framerate'])
    NumFrames = bdata['poses'].shape[0]

    return "0 " + str(1 / DataRate * NumFrames)


def amss_data_faces_z(vertices):
    # if value range in x-axis (from left hand to right hand) is more than that in z-axis (from toe toe to heel/ body thickness),
    # the model faces z-axis. Rotate 90 degress along y-axis.
    if (max(vertices[:, 0]) - min(vertices[:, 0])) > (max(vertices[:, 2]) - min(vertices[:, 2])):
        return True
    else:
        return False


def unit_checking(unit):
    unit = str(unit)
    if unit == "mm":
        unit = 1 / 1000
    elif unit == "cm":
        unit = 1 / 100
    elif unit == "m":
        unit = 1
    else:
        print("unit is not defined")
        assert False

    return unit
