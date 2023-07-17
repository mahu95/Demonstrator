from utils import binvox_rw
import numpy as np
import torch

# model_Drahterodieren = torch.jit.load('./model/model_DE_2023-06-22_19-44-33.pt') # map_location=torch.device('cuda:0'), if model should be loaded to GPU, by default CPU
# model_Fraesen_Drehen = torch.jit.load('./model/model_F&D_2023-06-23_12-46-35.pt') # map_location=torch.device('cuda:0'), if model should be loaded to GPU, by default CPU
# model_Schleifen = torch.jit.load('./model/model_Sch_2023-06-22_19-44-33.pt')
# model_Drahterodieren.eval()
# model_Fraesen_Drehen.eval()
# model_Schleifen.eval()

model_class_all = torch.jit.load('./model/model_2023-07-14_05-20-22_M.pt')
model_class_all.eval()


def classification(voxelFilePath):
    List = []
    with open(voxelFilePath, 'rb') as file:
        voxel_object = binvox_rw.read_as_3d_array(file)
        voxel = voxel_object.data.astype(np.float32)
        voxel = np.expand_dims(voxel, axis=0)
        voxel = np.expand_dims(voxel, axis=0)
        voxel = torch.tensor(voxel)

    # voxel = np.load(voxelFilePath)
    # voxel = np.expand_dims(voxel, axis=0)
    # voxel = np.expand_dims(voxel, axis=0)
    # voxel = voxel.astype(np.float32)
    # voxel = torch.tensor(voxel)

    # with torch.no_grad():
    #     output = model_Drahterodieren(voxel)
    #     print(output)
    #     output = torch.round(output)
    #     output = torch.squeeze(output)
    #     if output.item() == 1:
    #         List.append('Drahterodieren')

    # with torch.no_grad():
    #     output = model_Fraesen_Drehen(voxel)
    #     output = torch.round(output)
    #     output = torch.squeeze(output)
    #     if output[0] == 1:
    #         List.append('Fräsen')
    #     if output[1] == 1:
    #         List.append('Drehen')

    # with torch.no_grad():
    #     output = model_Schleifen(voxel)
    #     output = torch.round(output)
    #     output = torch.squeeze(output)
    #     if output[0] == 1:
    #         List.append('Flachschleifen')
    #     if output[1] == 1:
    #         List.append('Rundschleifen')
    #     if output[2] == 1:
    #         List.append('Koordinatenschleifen')

    with torch.no_grad():
        output = model_class_all(voxel)
        output = torch.round(output)
        output = torch.squeeze(output)
        if output[0] == 1:
            List.append('Fräsen')
        if output[1] == 1:
            List.append('Drehen')
        if output[2] == 1:
            List.append('Drahterodieren')
        if output[3] == 1:
            List.append('Flachschleifen')
        if output[4] == 1:
            List.append('Rundschleifen')
        if output[5] == 1:
            List.append('Koordinatenschleifen')

    return List