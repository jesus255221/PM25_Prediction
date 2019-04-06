import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

####
# Attention Layer
####

TIME_STEPS = 6

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = KL.Permute((2, 1))(inputs)
    a = KL.Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = KL.Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = KL.Permute((2, 1))(a)
    output_attention_mul = KL.Multiply()([inputs, a_probs])
    output_attention_sum = KL.Lambda(lambda x : K.sum(x, axis=1))(output_attention_mul)
    return output_attention_sum

####
# Description: One seq2seq with attention mech
####

encoder_inputs = Input(shape = (6, 77))
decoder_inputs = Input(shape = (1, 1))
Outputs = []
    
# The first part is unchanged
# Permute the matrix from (11, 6) to (6, 11) and apply the every input
encoder_outputs_1, state_1_f, state_1_b = \
    Bidirectional(GRU(64, return_state=True, return_sequences=True))(encoder_inputs)
encoder_outputs_2, state_2_f, state_2_b = \
    Bidirectional(GRU(64, return_state=True, return_sequences=True))(encoder_outputs_1)
encoder_outputs_3, state_3_f, state_3_b = \
    Bidirectional(GRU(64, return_state=True, return_sequences=True))(encoder_outputs_2)
encoder_outputs_4, state_4_f, state_4_b = \
    Bidirectional(GRU(64, return_state=True, return_sequences=True))(encoder_outputs_3)

# Set up the decoder, which will only process one timestep at a time.
decoder_gru_1 = Bidirectional(GRU(64, return_sequences=True, return_state=True))
decoder_gru_2 = Bidirectional(GRU(64, return_sequences=True, return_state=True))
decoder_gru_3 = Bidirectional(GRU(64, return_sequences=True, return_state=True))
decoder_gru_4 = Bidirectional(GRU(64, return_sequences=True, return_state=True))
decoder_dense = Dense(1, activation='relu')

all_outputs = []

inputs = decoder_inputs

for _ in range(6):

    # Run the decoder on one timestep
    outputs, state_1_f, state_1_b = decoder_gru_1(inputs, initial_state=(state_1_f, state_1_b))

    all_state_1_f = Lambda(lambda x : x[:, :, :64])(encoder_outputs_1)
    attention_state_1_f = attention_3d_block(all_state_1_f)
    all_state_1_b = Lambda(lambda x : x[:, :, 64:])(encoder_outputs_1)
    attention_state_1_b = attention_3d_block(all_state_1_b)

    attention_state_1_f = Lambda(lambda x : K.expand_dims(x, axis=1))(attention_state_1_f)
    attention_state_1_b = Lambda(lambda x : K.expand_dims(x, axis=1))(attention_state_1_b)

    outputs = Concatenate()([attention_state_1_f, attention_state_1_b, outputs])
    
    outputs, state_2_f, state_2_b = decoder_gru_2(outputs, initial_state=(state_2_f, state_2_b))
    
    all_state_2_f = Lambda(lambda x : x[:, :, :64])(encoder_outputs_2)
    attention_state_2_f = AttentionWithContext()(all_state_2_f)
    all_state_2_b = Lambda(lambda x : x[:, :, 64:])(encoder_outputs_2)
    attention_state_2_b = AttentionWithContext()(all_state_2_b)

    attention_state_2_f = Lambda(lambda x : K.expand_dims(x, axis=1))(attention_state_2_f)
    attention_state_2_b = Lambda(lambda x : K.expand_dims(x, axis=1))(attention_state_2_b)
    
    outputs = Concatenate()([attention_state_2_f, attention_state_2_b, outputs])
    
    outputs, state_3_f, state_3_b = decoder_gru_3(outputs, initial_state=(state_3_f, state_3_b))
    
    all_state_3_f = Lambda(lambda x : x[:, :, :64])(encoder_outputs_3)
    attention_state_3_f = AttentionWithContext()(all_state_3_f)
    all_state_3_b = Lambda(lambda x : x[:, :, 64:])(encoder_outputs_3)
    attention_state_3_b = AttentionWithContext()(all_state_3_b)

    attention_state_3_f = Lambda(lambda x : K.expand_dims(x, axis=1))(attention_state_3_f)
    attention_state_3_b = Lambda(lambda x : K.expand_dims(x, axis=1))(attention_state_3_b)
    
    outputs = Concatenate()([attention_state_3_f, attention_state_3_b, outputs])
    
    outputs, state_4_f, state_4_b = decoder_gru_4(outputs, initial_state=(state_4_f, state_4_b))
    
    all_state_4_f = Lambda(lambda x : x[:, :, :64])(encoder_outputs_4)
    attention_state_4_f = AttentionWithContext()(all_state_4_f)
    all_state_4_b = Lambda(lambda x : x[:, :, 64:])(encoder_outputs_4)
    attention_state_4_b = AttentionWithContext()(all_state_4_b)

    attention_state_4_f = Lambda(lambda x : K.expand_dims(x, axis=1))(attention_state_4_f)
    attention_state_4_b = Lambda(lambda x : K.expand_dims(x, axis=1))(attention_state_4_b)
    
    outputs = Concatenate()([attention_state_4_f, attention_state_4_b, outputs])

    outputs = decoder_dense(outputs)
    
    all_outputs.append(outputs)
    inputs = outputs

# Concatenate all predictions
decoder_outputs = Concatenate(axis = 1)(all_outputs)

# Define and compile model as previously
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.save_weights("./Seq2Seq_with_attention_v2/original_weights.h5")
model.summary()

with open('training_cnn_site_map.pk', 'rb') as file:
    cnn_site_map = pk.load(file)
    
with open('val_cnn_site_map.pk', 'rb') as file:
    val_cnn_site_map = pk.load(file)

reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5,  patience=50, verbose=1, min_lr=1e-7)
    
for i in range(62, 76, 1):
    
    print(i + 1, '/ 76 stations')

    check_pointer = ModelCheckpoint("./Seq2Seq_with_attention_v2/"+ str(i + 1) + ".h5", save_best_only=True)

    training_data = np.load("./history_npy/" + str(i + 1) + ".npy")
    label_data = np.load('./history_npy/' + str(i + 1) + '_label.npy')
    
    val_data = np.load('./history_npy/' + str(i + 1) + '_val.npy')
    val_label_data = np.load('./history_npy/' + str(i + 1) + '_val_label.npy')
    
    last_hr_pm25_data = label_data[:, 0:1]

    training_data = training_data[1:]
    training_data = np.swapaxes(training_data, 1, 2)
    label_data = label_data[1:]
    last_hr_pm25_data = last_hr_pm25_data[:-1]

    cnn_training = cnn_site_map[:, i]
    cnn_training = np.expand_dims(cnn_training, axis = 3)
    cnn_training = cnn_training[7:-6]
    
    decoder_training_data = np.expand_dims(last_hr_pm25_data, axis=1)

    val_last_hr_pm25_data = val_label_data[:, 0:1]

    val_data = val_data[1:]
    val_data = np.swapaxes(val_data, 1, 2)
    val_label_data = val_label_data[1:]
    val_last_hr_pm25_data = val_last_hr_pm25_data[:-1]

    cnn_val = val_cnn_site_map[:, i]
    cnn_val = np.expand_dims(cnn_val, axis = 3)
    cnn_val = cnn_val[7:-6]
    
    decoder_val_data = np.expand_dims(val_last_hr_pm25_data, axis=1)

    model.compile(optimizer=KO.Adam(lr=1e-5), loss='mae')
    model.load_weights("./Seq2Seq_with_attention_v2/original_weights.h5")
    model.fit([training_data, decoder_training_data], 
          np.expand_dims(label_data, axis=2), 
          batch_size=10240, 
          epochs=1000, 
          validation_data=([val_data, decoder_val_data], np.expand_dims(val_label_data, axis=2)), 
          callbacks=[check_pointer])
