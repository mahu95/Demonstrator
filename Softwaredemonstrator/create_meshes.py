import trimesh
import gmsh
import sys
import numpy as np
from pyvirtualdisplay import Display
from PIL import Image

def render(mesh, pngPath):
    # Create a virtual display
    display = Display(visible=0, size=(800, 600))
    display.start()

    # Load your trimesh mesh

    # Set a background color for the scene
    background_color = (255, 255, 255, 255)  # White color, adjust as needed

    # Render the mesh
    image = mesh.to_image(background=background_color)

    # Save the image as PNG
    
    image.save('path_to_save_image.png')


def createMeshes(stepFilePath, stlFilePath, objFilePath): 
    mesh = trimesh.Trimesh(**trimesh.interfaces.gmsh.load_gmsh(stepFilePath))
    mesh.export(stlFilePath)
    mesh.export(objFilePath)


if __name__ == "__main__":
    stepStorageFilePath = sys.argv[1]
    objStorageFilePath = sys.argv[2]
    stlStorageFilePath = sys.argv[3]

    createMeshes(stepFilePath=stepStorageFilePath, stlFilePath=stlStorageFilePath, objFilePath=objStorageFilePath)


