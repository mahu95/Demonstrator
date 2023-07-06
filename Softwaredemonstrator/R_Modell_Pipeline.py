from utils import binvox_rw
import numpy as np
import torch

Dic1 = {'<start>': 0, '<PAD>': 1, '<end>': 2, 'Sägen': 3, 'Drehen': 4, 'Rundschleifen': 5, 'Fräsen': 6, 'Messen': 7, 'Laserbeschriftung': 8, 'Flachschleifen': 9, 'Härten/Oberfläche': 10, 'Koordinatenschleifen': 11, 'Drahterodieren': 12, 'Startlochbohren': 13, 'Senkerodieren': 14, 'HSC-Fräsen': 15, 'Polieren': 16, 'Fremdvergabe': 17, 'Honen': 18, 'DF Dreh/Fräs-Z.-Mitlaufzeit': 19, 'Konstr. Werkzeuge': 20, 'Hartdrehen-CNC': 21, 'Drehen-CNC-Mitlaufzeit': 22}
Dic2 = {0: '<start>', 1: '<PAD>', 2: '<end>', 3: 'Sägen', 4: 'Drehen', 5: 'Rundschleifen', 6: 'Fräsen', 7: 'Messen', 8: 'Laserbeschriftung', 9: 'Flachschleifen', 10: 'Härten/Oberfläche', 11: 'Koordinatenschleifen', 12: 'Drahterodieren', 13: 'Startlochbohren', 14: 'Senkerodieren', 15: 'HSC-Fräsen', 16: 'Polieren', 17: 'Fremdvergabe', 18: 'Honen', 19: 'DF Dreh/Fräs-Z.-Mitlaufzeit', 20: 'Konstr. Werkzeuge', 21: 'Hartdrehen-CNC', 22: 'Drehen-CNC-Mitlaufzeit'}

model_R_Modell = torch.jit.load('./model/best_model_cnn_lstm.pt', map_location=torch.device('cpu'))
model_R_Modell.eval()

def Sequenzierung(List, voxelFilePath):
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

    List.insert(0, '<start>')
    List.append('<end>')
    Input_Vorgänge = np.ones((len(List), 1))

    index1 = 0
    for words in List:
        index2 = Dic1[words]
        Input_Vorgänge[index1] = index2
        index1 += 1

    with torch.no_grad():
        outputs, features = model_R_Modell(voxel, torch.tensor(Input_Vorgänge[:-1], dtype=torch.int64), torch.tensor(Input_Vorgänge, dtype=torch.int64))
        outputs = torch.round(outputs)
        outputs = torch.squeeze(outputs)
        Values, Indexes = torch.max(outputs, dim=1)
        Result = []
        for index in range(0, len(Indexes)):
            Result.append(Dic2[int(Indexes[index])])

    set1 = set(Result)
    set2 = set(List)
    non_matching = set1 ^ set2
    Result_end = [x for x in Result if x not in non_matching]

    return Result