from flask import Flask, render_template, request, redirect, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid
import zipfile
import io
import tempfile
import subprocess
import os
from collections import OrderedDict

from werkzeug.datastructures import FileStorage
from enum import Enum
from sqlalchemy import JSON

import os
import subprocess

from Classifier_Pipeline import classification
from R_Modell_Pipeline import Sequenzierung
from create_voxels import voxelization

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///vorplanml.db'

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
    objStorageFilePath = db.Column(db.String(100), nullable=False)
    voxelStorageFilePath = db.Column(db.String(100), nullable=False)
    comment = db.Column(db.String(500), nullable=True)
    customer = db.Column(db.String(100), nullable=True)
    drawingNumber = db.Column(db.String(500), nullable=True)
    orderNumber = db.Column(db.String(500), nullable=True)
    drawingStorageFilePath = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    material = db.Column(db.String(100), nullable=True)
    isSawing = db.Column(db.Boolean, nullable=True)
    isMeasuring = db.Column(db.Boolean, nullable=True)
    isLaserEngraving = db.Column(db.Boolean, nullable=True)
    isHardening = db.Column(db.Boolean, nullable=True)
    isStartholeDrilling = db.Column(db.Boolean, nullable=True)
    isSinkEroding = db.Column(db.Boolean, nullable=True)
    isHoning = db.Column(db.Boolean, nullable=True)
    isPolishing = db.Column(db.Boolean, nullable=True)

# Initialize the database
with app.app_context():
    db.create_all()
    
def createFile(file, comment, material,
               customer,
               drawingNumber,
               orderNumber,
               drawingFile,
               isSawing,
               isMeasuring,
               isLaserEngraving,
               isHardening,
               isStartholeDrilling,
               isSinkEroding,
               isHoning,
               isPolishing):

    # Save file to the upload folder
    originalFilename = file.filename
    
    if drawingFile:
        originalDrawingEnding = os.path.splitext(drawingFile.filename)[1] 
    else:
        originalDrawingEnding= None

    filename = str(uuid.uuid4())
    
    stepStorageFilePath=os.path.join(app.config['UPLOAD_FOLDER'], filename+ '.stp')
    objStorageFilePath=os.path.join(app.config['UPLOAD_FOLDER'], filename+ '.obj')
    voxelStorageFilePath=os.path.join(app.config['UPLOAD_FOLDER'], filename+ '.binvox')
    
    if drawingFile:
        drawingStorageFilePath=os.path.join(app.config['UPLOAD_FOLDER'], filename+ originalDrawingEnding)
        drawingFile.save(drawingStorageFilePath)
    else:
        drawingStorageFilePath=None
    file.save(stepStorageFilePath)

    ## dirty, but Trimesh cannot run in flask in a thread
    
    with app.app_context():
        #subprocess.run(['python', 'create_meshes.py', stepStorageFilePath, objStorageFilePath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        subprocess.run(['python3', 'create_meshes.py', stepStorageFilePath, objStorageFilePath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        voxelization(objFilePath=objStorageFilePath, voxelFilePath=voxelStorageFilePath)
    
        # Store text input in SQLite database
        part = Part(originalFilename=originalFilename, 
                    stepStorageFilePath = stepStorageFilePath,
                    objStorageFilePath = objStorageFilePath,
                    voxelStorageFilePath = voxelStorageFilePath,
                    comment=comment,
                    customer = customer,
                    drawingNumber = drawingNumber,
                    orderNumber = orderNumber,
                    drawingStorageFilePath = drawingStorageFilePath,
                    material=material,
                    isSawing=isSawing,
                    isMeasuring=isMeasuring,
                    isLaserEngraving=isLaserEngraving,
                    isHardening=isHardening,
                    isStartholeDrilling=isStartholeDrilling,
                    isSinkEroding=isSinkEroding,
                    isHoning=isHoning,
                    isPolishing=isPolishing)
        db.session.add(part)
        db.session.commit()

@app.route('/parts/obj/<int:part_id>', methods=['GET'])
def get_obj(part_id):
    part = Part.query.get_or_404(part_id)
    obj_path = part.objStorageFilePath
    return send_file(obj_path, as_attachment=True)

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
    comment = request.form['comment']

    drawingFile = request.files['drawingFile']

    material = request.form.get('material')
    if material == '':
        material=None
        
    isSawing = request.form.get('isSawing')
    isSawing = bool(isSawing) if isSawing else None

    isMeasuring = request.form.get('isMeasuring')
    isMeasuring = bool(isMeasuring) if isMeasuring else None
    
    isLaserEngraving = request.form.get('isLaserEngraving')
    isLaserEngraving = bool(isLaserEngraving) if isLaserEngraving else None
    
    isHardening = request.form.get('isHardening')
    isHardening = bool(isHardening) if isHardening else None
    
    isStartholeDrilling = request.form.get('isStartholeDrilling')
    isStartholeDrilling = bool(isStartholeDrilling) if isStartholeDrilling else None
    
    isSinkEroding = request.form.get('isSinkEroding')
    isSinkEroding = bool(isSinkEroding) if isSinkEroding else None

    isHoning = request.form.get('isHoning')
    isHoning = bool(isHoning) if isHoning else None

    isPolishing = request.form.get('isPolishing')
    isPolishing = bool(isPolishing) if isPolishing else None

    customer = request.form.get('customer')
    if customer == '':
        customer=None
        
    drawingNumber = request.form.get('drawingNumber')
    if drawingNumber == '':
        drawingNumber=None
        
    orderNumber = request.form.get('orderNumber')
    if orderNumber == '':
        orderNumber=None

    createFile(file, comment, material,
                customer,
                drawingNumber,
                orderNumber,
                drawingFile,
                isSawing,
                isMeasuring,
                isLaserEngraving,
                isHardening,
                isStartholeDrilling,
                isSinkEroding,
                isHoning,
                isPolishing)
    
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
                    createFile(file_storage, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
        
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

    Vorgaenge = []

    Vorgaenge_Classification = classification(voxelFilePath=part.voxelStorageFilePath)
    Vorgaenge = Vorgaenge_Classification

    if part.isSawing == True:
        Vorgaenge.append('Sägen')
    if part.isHardening == True:
        Vorgaenge.append('Härten/Oberfläche')
    if part.isMeasuring == True:
        Vorgaenge.append('Messen')
    if part.isLaserEngraving == True:
        Vorgaenge.append('Laserbeschriftung')
    if part.isStartholeDrilling == True:
        Vorgaenge.append('Startlochbohren')
    if part.isSinkEroding == True:
        Vorgaenge.append('Senkerodieren')
    if part.isHoning == True:
        Vorgaenge.append('Honen')
    if part.isPolishing == True:
        Vorgaenge.append('Polieren')

    Vorgangsfolge = Sequenzierung(List=Vorgaenge, voxelFilePath=part.voxelStorageFilePath)

    # Some hard-coded rules to solve for some prediction errors!
    set1 = set(Vorgaenge)
    set2 = set(Vorgangsfolge)

    non_matching = set2 - set1
    Vorgangsfolge = [x for x in Vorgangsfolge if x not in non_matching]

    Vorgangsfolge.extend(set1-set2)

    if 'Sägen' in Vorgangsfolge and Vorgangsfolge[0] != 'Sägen':
        index = Vorgangsfolge.index('Sägen')
        Vorgangsfolge.insert(0, Vorgangsfolge.pop(index))

    if 'Laserbeschriftung' in Vorgangsfolge and Vorgangsfolge[-1:] != 'Laserbeschriftung':
        index = Vorgangsfolge.index('Laserbeschriftung')
        Vorgangsfolge.append(Vorgangsfolge.pop(index))

    if 'Startlochbohren' in Vorgangsfolge:
        index1 = Vorgangsfolge.index('Startlochbohren')
        if Vorgangsfolge[index1+1] != 'Drahterodieren':
            index2 = Vorgangsfolge.index('Drahterodieren')
            Vorgangsfolge.insert(index2, Vorgangsfolge.pop(index1))

    seen = OrderedDict()
    removed_double = []
    for item in Vorgangsfolge:
        if item not in seen:
            seen[item] = None
            removed_double.append(item)
    
    Vorgangsfolge = removed_double
    
    return render_template('part.html', part=part, vorgangsfolge=Vorgangsfolge)

@app.route('/parts/edit/<int:part_id>', methods=['POST'])
def update_part(part_id):
    part = Part.query.get_or_404(part_id)
    
    isSawing = request.form.get('isSawing')
    isSawing = bool(isSawing) if isSawing else None
    part.isSawing = isSawing

    isMeasuring = request.form.get('isMeasuring')
    isMeasuring = bool(isMeasuring) if isMeasuring else None
    part.isMeasuring=isMeasuring
    
    isLaserEngraving = request.form.get('isLaserEngraving')
    isLaserEngraving = bool(isLaserEngraving) if isLaserEngraving else None
    part.isLaserEngraving = isLaserEngraving
    
    isHardening = request.form.get('isHardening')
    isHardening = bool(isHardening) if isHardening else None
    part.isHardening=isHardening
    
    isStartholeDrilling = request.form.get('isStartholeDrilling')
    isStartholeDrilling = bool(isStartholeDrilling) if isStartholeDrilling else None
    part.isStartholeDrilling= isStartholeDrilling
    
    isSinkEroding = request.form.get('isSinkEroding')
    isSinkEroding = bool(isSinkEroding) if isSinkEroding else None
    part.isSinkEroding=isSinkEroding
    
    isHoning = request.form.get('isHoning')
    isHoning = bool(isHoning) if isHoning else None
    part.isHoning=isHoning

    isPolishing = request.form.get('isPolishing')
    isPolishing = bool(isPolishing) if isPolishing else None
    part.isPolishing=isPolishing



    material = request.form.get('material')
    if material == '':
        material=None


    drawingNumber = request.form.get('drawingNumber')
    if drawingNumber == '':
        drawingNumber=None

    customer = request.form.get('customer')
    if customer == '':
        customer=None
        
    orderNumber = request.form.get('orderNumber')
    if orderNumber == '':
        orderNumber=None
        
    comment = request.form.get('comment')
    if comment == '':
        comment=None
        

    part.comment = comment
    part.customer = customer
    part.material = material
    part.drawingNumber = drawingNumber
    part.orderNumber = orderNumber



    db.session.commit()
    return redirect(url_for('parts'))

@app.route('/parts/edit/<int:part_id>', methods=['GET'])
def edit_part(part_id):
    part = Part.query.get_or_404(part_id)

    return render_template('edit_part.html', part=part, material=Material)
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5000)