import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.models import Model, Sequential
from keras.layers import GRU, Dense, Activation, Input, LSTM, TimeDistributed, Bidirectional, Lambda, Flatten, Permute, Concatenate, AvgPool2D
from AttentionWithContext import AttentionWithContext
from keras import backend as K
from keras import optimizers as KO
from keras import layers as KL
import pickle as pk
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

####
# Description: Seven Seq2Seq models with Bidirectional RNN
####

overall_encoder_input = Input(shape = (77, 6))
overall_decoder_input = Input(shape = (7, 1))
Outputs = []

for i in range(7):
    
    # The first part is unchanged
    encoder_inputs = Lambda(lambda x : x[:, 11 * i : 11 * (i + 1), :])(overall_encoder_input)
    # Permute the matrix from (11, 6) to (6, 11) and apply the every input
    p = Permute((2, 1))(encoder_inputs)
    encoder_outputs_1, state_1_f, state_1_b = Bidirectional(GRU(64, return_state=True, return_sequences=True))(p)

    # Set up the decoder, which will only process one timestep at a time.
    decoder_inputs = Lambda(lambda x : x[:, i : i + 1, :])(overall_decoder_input)
    decoder_gru_1 = Bidirectional(GRU(64, return_sequences=True, return_state=True))
    decoder_dense_1 = Dense(64, activation='relu')
    decoder_dense_2 = Dense(1, activation='relu')

    all_outputs = []
    inputs = decoder_inputs

    for _ in range(6):
        
        # Run the decoder on one timestep
        outputs, state_1_f, state_1_b = decoder_gru_1(inputs,  initial_state=(state_1_f, state_1_b))
        
        all_state_1_f = Lambda(lambda x : x[:, :, :64])(encoder_outputs_1)
        attention_state_f = AttentionWithContext()(all_state_1_f)
        all_state_1_b = Lambda(lambda x : x[:, :, 64:])(encoder_outputs_1)
        attention_state_b = AttentionWithContext()(all_state_1_b)
        
        attention_state_f = Lambda(lambda x : K.expand_dims(x, axis=1))(attention_state_f)
        attention_state_b = Lambda(lambda x : K.expand_dims(x, axis=1))(attention_state_b)
        
        outputs = Concatenate()([attention_state_f, attention_state_b, outputs])

        outputs = decoder_dense_1(outputs)
        outputs = decoder_dense_2(outputs)
        # Store the current prediction (we will concatenate all predictions later)
        all_outputs.append(outputs)
        # Reinject the outputs as inputs for the next loop iteration
        # as well as update the states
        inputs = outputs

    # Concatenate all predictions
    decoder_outputs = Concatenate(axis = 1)(all_outputs)
    Outputs.append(decoder_outputs)
    
cat = Concatenate(axis = 1)(Outputs)
lstm_flat = Flatten()(cat)

# Avg pooling the site map of the previous hour 
cnn_input = Input(shape=(11, 11, 1))
avg_pool = AvgPool2D(pool_size=3, strides=2)(cnn_input)
cnn_flat = Flatten()(avg_pool)

# Concat all of them
last_hr_pm25 = Input(shape=(1,))
concat = Concatenate()([lstm_flat, cnn_flat, last_hr_pm25])
dense = Dense(21, activation='relu')(concat)
overall_output = Dense(6, activation='relu')(dense)

# Define and compile model as previously
model = Model([overall_encoder_input, overall_decoder_input, last_hr_pm25, cnn_input], overall_output)
model.compile(optimizer='adam', loss='mae')
model.save_weights('original_weights.h5')
model.summary()

with open('training_cnn_site_map.pk', 'rb') as file:
    cnn_site_map = pk.load(file)
    
with open('val_cnn_site_map.pk', 'rb') as file:
    val_cnn_site_map = pk.load(file)

early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
    
for i in range(77):
    
    print(i + 1, '/ 77 stations')

    check_pointer = ModelCheckpoint("./Seq2Seq_with_attention/"+ str(i + 1) + ".h5", save_best_only=True)

    training_data = np.load("./history_npy/" + str(i + 1) + ".npy")
    label_data = np.load('./history_npy/' + str(i + 1) + '_label.npy')
    
    val_data = np.load('./history_npy/' + str(i + 1) + '_val.npy')
    val_label_data = np.load('./history_npy/' + str(i + 1) + '_val_label.npy')
    
    last_hr_pm25_data = label_data[:, 0:1]

    training_data = training_data[1:]
    label_data = label_data[1:]
    last_hr_pm25_data = last_hr_pm25_data[:-1]

    cnn_training = cnn_site_map[:, i]
    cnn_training = np.expand_dims(cnn_training, axis = 3)
    cnn_training = cnn_training[7:-6]

    decoder_training_data = np.zeros((training_data.shape[0], 7, 1))

    val_last_hr_pm25_data = val_label_data[:, 0:1]

    val_data = val_data[1:]
    val_label_data = val_label_data[1:]
    val_last_hr_pm25_data = val_last_hr_pm25_data[:-1]

    cnn_val = val_cnn_site_map[:, i]
    cnn_val = np.expand_dims(cnn_val, axis = 3)
    cnn_val = cnn_val[7:-6]

    decoder_val_data = np.zeros((val_data.shape[0], 7, 1))

    model.compile(optimizer='adam', loss='mae')
    model.load_weights('original_weights.h5')
    model.fit([training_data, decoder_training_data, last_hr_pm25_data, cnn_training], 
              label_data, 
              batch_size=10240, 
              epochs=500, 
              validation_data=([val_data, decoder_val_data, val_last_hr_pm25_data, cnn_val], val_label_data), 
              callbacks=[early_stopping, check_pointer])
