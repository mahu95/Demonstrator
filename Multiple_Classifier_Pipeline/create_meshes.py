import trimesh
import gmsh
import sys
import numpy as np


def createMeshes(stepFilePath, stlFilePath, objFilePath): 
    mesh = trimesh.Trimesh(**trimesh.interfaces.gmsh.load_gmsh(stepFilePath))
    mesh.export(stlFilePath)
    mesh.export(objFilePath)


if __name__ == "__main__":
    stepStorageFilePath = sys.argv[1]
    objStorageFilePath = sys.argv[2]
    stlStorageFilePath = sys.argv[3]

    createMeshes(stepStorageFilePath, stlStorageFilePath, objStorageFilePath)


