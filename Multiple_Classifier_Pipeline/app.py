from flask import Flask, render_template, request, redirect, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid
import zipfile
import io
import tempfile
import subprocess
import os

from werkzeug.datastructures import FileStorage
from enum import Enum
from sqlalchemy import JSON
import trimesh
import gmsh
import threading
import sqlite3

from utils import binvox_rw
import numpy as np
import torch
import trimesh
import os
import subprocess

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///vorplanml.db'

model_Drahterodieren = torch.jit.load('./model/model_DE_2023-06-22_19-44-33.pt') # map_location=torch.device('cuda:0'), if model should be loaded to GPU, by default CPU
model_Fraesen_Drehen = torch.jit.load('./model/model_F&D_2023-06-23_12-46-35.pt') # map_location=torch.device('cuda:0'), if model should be loaded to GPU, by default CPU
model_Schleifen = torch.jit.load('./model/model_Sch_2023-06-22_19-44-33.pt')
model_R_Modell = torch.jit.load('./model/best_model_cnn_lstm.pt', map_location=torch.device('cpu'))
model_Drahterodieren.eval()
model_Fraesen_Drehen.eval()
model_Schleifen.eval()
model_R_Modell.eval()

Dic1 = {'<start>': 0, '<PAD>': 1, '<end>': 2, 'Sägen': 3, 'Drehen': 4, 'Rundschleifen': 5, 'Fräsen': 6, 'Messen': 7, 'Laserbeschriftung': 8, 'Flachschleifen': 9, 'Härten/Oberfläche': 10, 'Koordinatenschleifen': 11, 'Drahterodieren': 12, 'Startlochbohren': 13, 'Senkerodieren': 14, 'HSC-Fräsen': 15, 'Polieren': 16, 'Fremdvergabe': 17, 'Honen': 18, 'DF Dreh/Fräs-Z.-Mitlaufzeit': 19, 'Konstr. Werkzeuge': 20, 'Hartdrehen-CNC': 21, 'Drehen-CNC-Mitlaufzeit': 22}
Dic2 = {0: '<start>', 1: '<PAD>', 2: '<end>', 3: 'Sägen', 4: 'Drehen', 5: 'Rundschleifen', 6: 'Fräsen', 7: 'Messen', 8: 'Laserbeschriftung', 9: 'Flachschleifen', 10: 'Härten/Oberfläche', 11: 'Koordinatenschleifen', 12: 'Drahterodieren', 13: 'Startlochbohren', 14: 'Senkerodieren', 15: 'HSC-Fräsen', 16: 'Polieren', 17: 'Fremdvergabe', 18: 'Honen', 19: 'DF Dreh/Fräs-Z.-Mitlaufzeit', 20: 'Konstr. Werkzeuge', 21: 'Hartdrehen-CNC', 22: 'Drehen-CNC-Mitlaufzeit'}

db = SQLAlchemy(app)
    
class Material(Enum):
    MAT1 = 'Stahl - C45'
    MAT2 = 'Stahl - 42CrMo4'
    MAT3 = 'Aluminium - AlMgSi1'
    MAT4 = 'Gusseisen - GG25'
    MAT5 = 'Kupfer - CuZn37'
    MAT6 = 'Messing - Ms58'
    MAT7 = 'Bronze - CuSn8'
    MAT8 = 'Titan - Ti6Al4V'
    MAT9 = 'Nickellegierungen - Inconel 625'
    MAT10 = 'Kunststoffe - PA6'
    MAT11 = 'Nichteisenmetalle - Zink'
    MAT12 = 'Chromstahl - X12CrNi17-7'
    MAT13 = 'Werkzeugstahl - HSS'
    MAT14 = 'Stahlguss - GS-C25'
    MAT15 = 'Aluminiumlegierungen - AlSi1MgMn'
    MAT16 = 'Kugellagerstahl - 100Cr6'
    MAT17 = 'Rostfreier Stahl - X6Cr17'
    MAT18 = 'Titanlegierungen - TiAl6V4'
    MAT19 = 'Hochtemperaturlegierungen - Inconel 718'
    MAT20 = 'Zinklegierungen - ZL0410'
    MAT21 = 'Kohlenstoffstahl - C55'
    MAT22 = 'Aluminiumbronze - CuAl10Ni'
    MAT23 = 'Nickel-Chrom-Legierungen - Nichrome'
    MAT24 = 'Superlegierungen - Inconel 625'
    MAT25 = 'Verbundwerkstoffe - GFK'
    
     
class AdditionalData(Enum):
    DATA1 = "Sägen"
    DATA2 = "Messen"
    DATA3 = "Laserbeschriftung"
    DATA4 = "Härten"
    DATA5 = "Startlochbohren"
    DATA6 = "Senkerodieren"
    DATA7 = "Hohnen"
    DATA8 = "Polieren"
    
    
class Part(db.Model):
    __tablename__ = 'part'

    id = db.Column(db.Integer, primary_key=True)
    originalFilename = db.Column(db.String(100), nullable=False)
    stepStorageFilePath = db.Column(db.String(100), nullable=False)
    stlStorageFilePath = db.Column(db.String(100), nullable=False)
    objStorageFilePath = db.Column(db.String(100), nullable=False)
    voxelStorageFilePath = db.Column(db.String(100), nullable=False)
    givenName = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    material = db.Column(db.String(100), nullable=True)
    #sawing = db.Column(db.Boolean, default=False, nullable=True)


# Initialize the database
with app.app_context():
    db.create_all()

def createVoxel(objFilePath): 
    subprocess.Popen(["Xvfb", ":99", "-screen", "0", "640x480x24"])
    os.environ['DISPLAY'] = ':99'                                       
    os.system( "./utils/binvox -d 64 " + objFilePath)

def classification(voxelFilePath):
    List = []
    with open(voxelFilePath, 'rb') as file:
        voxel_object = binvox_rw.read_as_3d_array(file)
        voxel = voxel_object.data.astype(np.float32)
        voxel = np.expand_dims(voxel, axis=0)
        voxel = np.expand_dims(voxel, axis=0)
        voxel = torch.tensor(voxel)

    with torch.no_grad():
        output = model_Drahterodieren(voxel)
        print(output)
        output = torch.round(output)
        output = torch.squeeze(output)
        if output.item() == 1:
            List.append('Drahterodieren')

    with torch.no_grad():
        output = model_Fraesen_Drehen(voxel)
        output = torch.round(output)
        output = torch.squeeze(output)
        if output[0] == 1:
            List.append('Fräsen')
        if output[1] == 1:
            List.append('Drehen')

    with torch.no_grad():
        output = model_Schleifen(voxel)
        output = torch.round(output)
        output = torch.squeeze(output)
        if output[0] == 1:
            List.append('Flachschleifen')
        if output[1] == 1:
            List.append('Rundschleifen')
        if output[2] == 1:
            List.append('Koordinatenschleifen')

    return List
    
def Reihenfolge(List, voxelFilePath):
    with open(voxelFilePath, 'rb') as file:
        voxel_object = binvox_rw.read_as_3d_array(file)
        voxel = voxel_object.data.astype(np.float32)
        voxel = np.expand_dims(voxel, axis=0)
        voxel = np.expand_dims(voxel, axis=0)
        voxel = torch.tensor(voxel)

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

    print(Result)
    return Result
    
def createFile(file, givenName, material):
    # Save file to the upload folder
    originalFilename = file.filename
    
    filename = str(uuid.uuid4())
    
    stepStorageFilePath=os.path.join(app.config['UPLOAD_FOLDER'], filename+ '.stp')
    stlStorageFilePath=os.path.join(app.config['UPLOAD_FOLDER'], filename+ '.stl')
    objStorageFilePath=os.path.join(app.config['UPLOAD_FOLDER'], filename+ '.obj')
    voxelStorageFilePath=os.path.join(app.config['UPLOAD_FOLDER'], filename+ '.binvox')

    file.save(stepStorageFilePath)

    ## dirty, but Trimesh cannot run in flask in a thread
    with app.app_context():

        subprocess.run(
            ['python3', 'create_meshes.py', stepStorageFilePath, stlStorageFilePath, objStorageFilePath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
            )

        createVoxel(objFilePath=objStorageFilePath)
        #createVoxelTrimesh(objFilePath=objStorageFilePath, voxelFilePath=voxelStorageFilePath)

        # Store text input in SQLite database
        part = Part(originalFilename=originalFilename, 
                    stepStorageFilePath = stepStorageFilePath,
                    stlStorageFilePath  = stlStorageFilePath,
                    objStorageFilePath = objStorageFilePath,
                    voxelStorageFilePath = voxelStorageFilePath,
                    givenName=givenName,
                    material=material)
        db.session.add(part)
        db.session.commit()


@app.route('/parts/stl/<int:part_id>', methods=['GET'])
def get_stl(part_id):
    part = Part.query.get_or_404(part_id)
    stl_path = part.stlStorageFilePath
    return send_file(stl_path, as_attachment=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        return redirect(url_for('index'))
    
    data=[]
    
    return render_template('index.html', data=data)


@app.route('/upload', methods=['GET'])
def upload():
    return render_template('upload.html', material=Material)

@app.route('/upload', methods=['POST'])
def upload_page():
    # Get data from the form
    file = request.files['file']
    givenName = request.form['givenName']
    #sawing = request.form['sawing']


    material = request.form.get('material')
    if material == '':
        material=None

    createFile(file, givenName, material)
    
        
    return redirect(url_for('parts'))

@app.route('/uploadMultiple', methods=['GET'])
def upload_multiple_page():
    return render_template('uploadMultiple.html')

@app.route('/uploadMultiple', methods=['POST'])
def upload_multiple():
    # Get data from the form
    file = request.files['file']

    if file.filename.endswith('.zip'):
        with zipfile.ZipFile(file, 'r') as zip_ref:
            for member in zip_ref.infolist():
                if member.filename.endswith(('.stp', '.STEP')) and not member.filename.startswith('__MACOSX/'):
                    # Extract the part file from the zip
                    tmp_folder_name = str(uuid.uuid4())
                    tmp_folder_path = os.path.join('tmp', tmp_folder_name)

                    ## TODO: delete tmp folder
                    zip_ref.extract(member, path=tmp_folder_path)
                    extracted_file_path = os.path.join(tmp_folder_path, member.filename)
                
                    with open(extracted_file_path, 'rb') as f:
                        # Create a BytesIO object from the byte stream
                        file = io.BytesIO(f.read())
                        name = os.path.basename(extracted_file_path)
                        file_storage = FileStorage(stream=file, filename=name)

                    # Create a Part object using the byte stream
                    createFile(file_storage, None, None, None)
        
        return redirect(url_for('parts'))
    else:
        return 'Invalid file format. Please upload a ZIP file.'

@app.route('/parts/delete/<int:part_id>', methods=['GET', 'POST'])
def delete_part(part_id):
    part = Part.query.get(part_id)
    if part:
        db.session.delete(part)
        db.session.commit()
        
        try:
            os.remove( part.stepStorageFilePath)
            os.remove( part.stlStorageFilePath)
            os.remove( part.objStorageFilePath)
            os.remove( part.voxelStorageFilePath)
        except:
            pass

        return redirect(url_for('parts'))
    else:
        return 'Part not found!'
    
@app.route('/parts', methods=['GET'])
def parts():
    parts = Part.query.all()
    return render_template('parts.html', parts=parts)

@app.route('/parts/<int:part_id>', methods=['GET'])
def view_part(part_id):
    part = Part.query.get_or_404(part_id)

    #Zu der Liste Vorgaenge müssen noch die anderen Technologien hinzugeüfgt werden!!!
    Vorgaenge = []
    Vorgaenge_Classification = classification(voxelFilePath=part.voxelStorageFilePath)
    Vorgaenge = Vorgaenge_Classification
    Vorgangsfolge = Reihenfolge(List=Vorgaenge, voxelFilePath=part.voxelStorageFilePath)
    
    return render_template('part.html', part=part, vorgangsfolge=Vorgangsfolge)

@app.route('/parts/edit/<int:part_id>', methods=['POST'])
def update_part(part_id):
    part = Part.query.get_or_404(part_id)
    part.givenName = request.form['givenName']
    
    material = request.form.get('material')
    if material == '':
        material=None

    part.material = material

    db.session.commit()
    return redirect(url_for('parts'))

@app.route('/parts/edit/<int:part_id>', methods=['GET'])
def edit_part(part_id):
    part = Part.query.get_or_404(part_id)

    return render_template('edit_part.html', part=part, material=Material)
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5000)