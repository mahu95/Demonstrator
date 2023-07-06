# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 17:39:52 2021

@author: mhussong
"""
#import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import InputLayer, LSTM, Dense, RepeatVector
import numpy as np
import os
import sys

import random


maxSequence = 24
numberOperations = 14
latent_dim = 5*numberOperations
IDToOperation = {0: 'Sägen', 1: 'Drehen', 2: 'Rundschleifen', 3: 'Fräsen', 4: 'Messen', 5: 'Laserbeschriftung', 6: 'Flachschleifen', 7: 'Härten/Oberfläche', 8: 'Koordinatenschleifen', 9: 'Drahterodieren', 10: 'Startlochbohren', 11: 'Senkerodieren', 12: 'Polieren', 13: 'Honen'}
OperationToID = {'Sägen': 0, 'Drehen': 1, 'Rundschleifen': 2, 'Fräsen': 3, 'Messen': 4, 'Laserbeschriftung': 5, 'Flachschleifen': 6, 'Härten/Oberfläche': 7, 'Koordinatenschleifen': 8, 'Drahterodieren': 9, 'Startlochbohren': 10, 'Senkerodieren': 11, 'Polieren': 12, 'Honen': 13}


def Encoding(OperationSequences):
        """Definition of input shape"""
        x = np.zeros((len(OperationSequences), maxSequence, numberOperations))
        index1 = 0
        for key in OperationSequences.keys():
                """Get the length of the longest operation sequence"""
                for index in range(0,len(OperationSequences[key])):
                    OS = OperationSequences[key]
                    """One-Hot-Encoding"""
                    for index2 in range(0,len(OS)): 
                        i = OperationToID[OS[index2]]
                        x[index1][index2+1][i] = 1
                          
                index1 += 1
        return x


def Decoding(matrix_single_sequence):
        x = np.where(matrix_single_sequence == 1)
        x = np.array(x)
        x = np.delete(x, 0,0)
        x = x.flatten()
        String = ", ".join(IDToOperation[x] for x in x)
        if not String:
            String = 'Es gab keine Decodierung, da alle Einträge Null sind!'
        return String

def Vorhersage(Vorgaenge):
        encoder_inputs = Input(shape=(None, numberOperations))
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        decoder_inputs = Input(shape=(None, numberOperations))
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(numberOperations, activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.load_weights('./model/LSTM.h5')

        Dic = {}
        Dic['123'] = Vorgaenge
        encodedTest = Encoding(Dic)

        encoder_inputs = model.input[0]  # input_1
        encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
        encoder_model = Model(encoder_inputs, encoder_states)
        decoder_inputs = model.input[1]  # input_2
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = model.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        states_value = encoder_model.predict(encodedTest)
        decoder_test_input = np.zeros(shape=(1, 1, numberOperations))
        stop_condition = False
        decoded_sentence = 1
        Result = []
        while not stop_condition:
                output_tokens, h, c = decoder_model.predict([decoder_test_input] + states_value)
                sampled_token_index = np.argmax(output_tokens)
                Result.append(IDToOperation[sampled_token_index])
                decoded_sentence += 1
                if decoded_sentence > len(Vorgaenge):
                        stop_condition = True
                
                #Update the target sequence (of length 1).
                decoder_test_input = np.zeros((1, 1, numberOperations))
                decoder_test_input[0, 0, sampled_token_index] = 1.0
                #print(decoder_test_input)

                # Update states
                states_value = [h, c]

        return Result


