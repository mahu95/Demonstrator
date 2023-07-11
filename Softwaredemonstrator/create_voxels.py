import os
import subprocess

def voxelization(objFilePath, voxelFilePath): 
    #subprocess.Popen(["Xvfb", ":99", "-screen", "0", "640x480x24"])
    #os.environ['DISPLAY'] = ':99'                                       
    #os.system( "./utils/binvox -d 64 " + objFilePath)

    subprocess.run(["./utils/binvox_win -d 64 " + objFilePath])

    # mesh = trimesh.load_mesh(objFilePath)
    # [x,y,z] = mesh.bounding_box_oriented.extents
    # mesh.apply_scale((1/x, 1/y, 1/z))
    # v = mesh.voxelized(1/63.)
    # voxel_data = v.matrix
    # np.save(voxelFilePath, voxel_data)
