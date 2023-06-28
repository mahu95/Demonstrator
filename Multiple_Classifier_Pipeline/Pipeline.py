from utils import binvox_rw
import numpy as np
import torch
import trimesh
import os
import subprocess

mesh = trimesh.Trimesh(**trimesh.interfaces.gmsh.load_gmsh('./data/100_100.stp'))
mesh.export('./data/100_100.obj')

subprocess.Popen(["Xvfb", ":99", "-screen", "0", "640x480x24"]) # öffnet ein virtuelles Display, wird benötigt für Headless Linux Systeme
os.environ['DISPLAY'] = ':99'                                        # führt das virtuelle Display aus
os.system( "./utils/binvox -d 64 " + "./data/100_100.obj")

model_Drahterodieren = torch.jit.load('./model/model_DE_2023-06-22_19-44-33.pt') # map_location=torch.device('cuda:0'), if model should be loaded to GPU, by default CPU
model_Fraesen_Drehen = torch.jit.load('./model/model_F&D_2023-06-23_12-46-35.pt') # map_location=torch.device('cuda:0'), if model should be loaded to GPU, by default CPU
model_Schleifen = torch.jit.load('./model/model_Sch_2023-06-22_19-44-33.pt') # map_location=torch.device('cuda:0'), if model should be loaded to GPU, by default CPU

with open('./data/100_100.binvox', 'rb') as file:
     voxel_object = binvox_rw.read_as_3d_array(file)
     voxel = voxel_object.data.astype(np.float32)
     voxel = np.expand_dims(voxel, axis=0)
     voxel = np.expand_dims(voxel, axis=0)
     voxel = torch.tensor(voxel)
     print(voxel_object)

List = []

print("Ergebnisse Classifier")
model_Drahterodieren.eval()
with torch.no_grad():
    output = model_Drahterodieren(voxel)
    print(output)
    output = torch.round(output)
    output = torch.squeeze(output)
    if output.item() == 1:
      List.append('Drahterodieren')

model_Fraesen_Drehen.eval()
with torch.no_grad():
    output = model_Fraesen_Drehen(voxel)
    print(output)
    output = torch.round(output)
    output = torch.squeeze(output)
    if output[0] == 1:
      List.append('Fraesen')
    if output[1] == 1:
      List.append('Drehen')

model_Schleifen.eval()
with torch.no_grad():
    output = model_Schleifen(voxel)
    print(output)
    output = torch.round(output)
    output = torch.squeeze(output)
    if output[0] == 1:
      List.append('Flachschleifen')
    if output[1] == 1:
      List.append('Rundschleifen')
    if output[2] == 1:
      List.append('Koordinatenschleifen')
        
print('Ergebnis von ' + '100_100' + ' ist: ' + str(List))


print("Ergebnisse Reihenfolgenmodell")
model_R_Modell = torch.jit.load('./model/best_model_cnn_lstm.pt', map_location=torch.device('cpu'))
Dic1 = {'<start>': 0, '<PAD>': 1, '<end>': 2, 'Sägen': 3, 'Drehen': 4, 'Rundschleifen': 5, 'Fräsen': 6, 'Messen': 7, 'Laserbeschriftung': 8, 'Flachschleifen': 9, 'Härten/Oberfläche': 10, 'Koordinatenschleifen': 11, 'Drahterodieren': 12, 'Startlochbohren': 13, 'Senkerodieren': 14, 'HSC-Fräsen': 15, 'Polieren': 16, 'Fremdvergabe': 17, 'Honen': 18, 'DF Dreh/Fräs-Z.-Mitlaufzeit': 19, 'Konstr. Werkzeuge': 20, 'Hartdrehen-CNC': 21, 'Drehen-CNC-Mitlaufzeit': 22}
Dic2 = {0: '<start>', 1: '<PAD>', 2: '<end>', 3: 'Sägen', 4: 'Drehen', 5: 'Rundschleifen', 6: 'Fräsen', 7: 'Messen', 8: 'Laserbeschriftung', 9: 'Flachschleifen', 10: 'Härten/Oberfläche', 11: 'Koordinatenschleifen', 12: 'Drahterodieren', 13: 'Startlochbohren', 14: 'Senkerodieren', 15: 'HSC-Fräsen', 16: 'Polieren', 17: 'Fremdvergabe', 18: 'Honen', 19: 'DF Dreh/Fräs-Z.-Mitlaufzeit', 20: 'Konstr. Werkzeuge', 21: 'Hartdrehen-CNC', 22: 'Drehen-CNC-Mitlaufzeit'}

Vorgänge = ['<start>', 'Sägen', 'Drehen', 'Flachschleifen', 'Messen', '<end>']

Input_Vorgänge = np.ones((len(Vorgänge), 1))

index1 = 0
for words in Vorgänge:
	index2 = Dic1[words]
	Input_Vorgänge[index1] = index2
	index1 += 1


model_R_Modell.eval()
with torch.no_grad():
    outputs, features = model_R_Modell(voxel, torch.tensor(Input_Vorgänge[:-1], dtype=torch.int64), torch.tensor(Input_Vorgänge, dtype=torch.int64))
    outputs = torch.round(outputs)
    outputs = torch.squeeze(outputs)
    Values, Indexes = torch.max(outputs, dim=1)
    Result = []
    for index in range(0, len(Indexes)):
          Result.append(Dic2[int(Indexes[index])])

    print(Result)
   