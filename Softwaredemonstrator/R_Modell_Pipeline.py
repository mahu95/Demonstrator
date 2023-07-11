import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import numpy as np

from utils import binvox_rw

Dic1 = {'<start>': 0, '<PAD>': 1, '<end>': 2, 'Sägen': 3, 'Drehen': 4, 'Rundschleifen': 5, 'Fräsen': 6, 'Messen': 7, 'Laserbeschriftung': 8, 'Flachschleifen': 9, 'Härten/Oberfläche': 10, 'Koordinatenschleifen': 11, 'Drahterodieren': 12, 'Startlochbohren': 13, 'Senkerodieren': 14, 'Polieren': 15, 'Honen': 16}
Dic2 = {0: '<start>', 1: '<PAD>', 2: '<end>', 3: 'Sägen', 4: 'Drehen', 5: 'Rundschleifen', 6: 'Fräsen', 7: 'Messen', 8: 'Laserbeschriftung', 9: 'Flachschleifen', 10: 'Härten/Oberfläche', 11: 'Koordinatenschleifen', 12: 'Drahterodieren', 13: 'Startlochbohren', 14: 'Senkerodieren', 15: 'Polieren', 16: 'Honen'}

class CADEncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(CADEncoderCNN, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(1, 16)
        self.pool_layer1 = nn.MaxPool3d(3, stride=2)
        self.conv_layer2 = self._conv_layer_set(16, 32)
        self.pool_layer2 = nn.MaxPool3d(3, stride=2)
        self.conv_layer3 = self._conv_layer_set(32, 64)
        self.pool_layer3 = nn.MaxPool3d(3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(64, embed_size) #TODO: Remove hard coding 128
        self.batch1=nn.BatchNorm1d(embed_size) #128
        self.fc2 = nn.Linear(128, embed_size)
        self.relu = nn.LeakyReLU()
        self.batch2=nn.BatchNorm1d(embed_size)
        self.drop=nn.Dropout(p=0.5)
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3),stride=[1,1,1],padding=[0,0,0]),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2))
        )
        return conv_layer
    

    def forward(self, x):
        out = self.conv_layer1(x)
        # out = self.pool_layer1(out)
        out = self.conv_layer2(out)
        # out = self.pool_layer2(out)
        out = self.conv_layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.batch1(out)
        out = self.relu(out)
        #out = self.fc2(out)
        #out = self.batch2(out)
        #out = self.relu(out)
        return out
    
class SequenceEncoder(nn.Module):
    '''
    Enocde the random words into a single vector
    '''
    def __init__(self, hidden_size, vocab_size) -> None:
        super().__init__()
        self.hidden_sze = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input):
        '''
        Receive the randomized target as an input and returns
        the final hidden state, which act as decoder's hidden state(h_0)
        '''
        embed = self.embedding(input)
        output, hidden = self.gru(embed)
        return output, hidden

class SequenceDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, output_size, num_layers=1): # hardcoding num layers for attention, remove the parameter
        super(SequenceDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, features, captions, encoder_hidden):
        # sourcery skip: inline-immediately-returned-variable
        '''
        Receive the features from the encoderCNN, the captions from the dataloader, 
        and the encoder_hidden as a vector representation of the randomized caption
        '''
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        output, hidden = self.gru(embeddings, encoder_hidden)
        outputs = self.linear(output)
        return outputs
    
    def init_hidden(self, batch_size):
        return torch.zeros(1,batch_size,self.hidden_size, device=torch.device('cpu'))

class CADSequenceModel(nn.Module):
    '''
    This model combines the CNN and RNN models
    '''
    def __init__(self, hidden_size, vocab_size, output_size, num_layers):
        super(CADSequenceModel, self).__init__()
        self.encoderCNN = CADEncoderCNN(hidden_size)
        self.encoderRNN = SequenceEncoder(hidden_size, vocab_size)
        self.decoderRNN = SequenceDecoder(hidden_size, vocab_size, output_size, num_layers)

    def forward(self, images, captions, randomized_caption):
        features = self.encoderCNN(images)
        encoder_outputs, encoder_hidden = self.encoderRNN(randomized_caption)
        outputs = self.decoderRNN(features, captions, encoder_hidden)
        return outputs, features

    def caption_image(self, image, vocabulary, randomized_caption, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            encoder_outputs, encoder_hidden = self.encoderRNN(randomized_caption)

            for _ in range(max_length):
                out, encoder_hidden = self.decoderRNN.gru(x, encoder_hidden)
                output = self.decoderRNN.linear(out.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary[predicted.item()] == "<end>":
                    break

        return [vocabulary[idx] for idx in result_caption]


model_R_Modell = CADSequenceModel(hidden_size=32, vocab_size=17, output_size=17, num_layers=1).to(torch.device('cpu'))

model = torch.load('./model/best_model_cnn_lstm.pt', map_location=torch.device('cpu'))
model_R_Modell.load_state_dict(model['model_dict'])
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

    # List.insert(0, '<start>')
    # List.append('<end>')

    Input_Vorgänge = np.ones((len(List), 1))

    index1 = 0
    for words in List:
        index2 = Dic1[words]
        Input_Vorgänge[index1] = index2
        index1 += 1


    with torch.no_grad():
        outputs = model_R_Modell.caption_image(image=voxel, vocabulary=Dic2, randomized_caption= torch.tensor(Input_Vorgänge, dtype=torch.int64))

    return outputs

#List = ['Fräsen', 'Drehen', 'Rundschleifen', 'Drahterodieren', 'Flachschleifen']
#Ergebnis = Sequenzierung(List=List, voxelFilePath='C:/Users/mhussong/Desktop/100_100.binvox')
#print(Ergebnis)
