'''
Example file for load and run deep learning model

'''
################################################################
# Load libraries

import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # scaler for normalizing data

import torch
from Model.MLP_mixer import SEP_MTMV_QMLP_MIXER_V4 #import torch model
from utils import *

#################################################################
# set initial values for model run
# values of following variables can be changed in the future
# INPUT_SCALE and OUTPUT_SCALE can be differ according to the neural network model
print('Initialize Variables')
INPUT_SCALE = 'SS' # 'SS' for standard scaler, 'MM' for minmax
OUTPUT_SCALE = 'SS' # 'SS' for standard scaler, 'MM' for minmax, 

#C_IN and OUT_LEN can be changed to other values according to the model training
C_IN = 49  # Number of input channel
OUT_LEN = 333 # Length of output time series data 
data_variables = ['PPSTRB(3)','PPS','FREL(1)','FREL(2)'] # List of output variables in order
# neural network model outputs [N, OUT_LEN, len(data_variables)] where N is the number of input for neural network

print('Input scaler type is {}'.format(INPUT_SCALE))
print('Output scaler type is {}'.format(OUTPUT_SCALE))
print('Number of input channel is {}'.format(C_IN))
print('Number of output variable is {}'.format(len(data_variables)))
print('Names of output variables are {}'.format(data_variables))
print('Length of output is {}'.format(OUT_LEN))
#############################################################
# Data loading part
# load saved scaler, scaler has following structures
# scaler - input - SS
#        |       |- MM
#        - output - SS - PPSTRB(3)
#                 |     |- PPS
#                 |     |- FREL(1)
#                 |     |- FREL(2)
#                 - MM - PPSTRB(3)
#                      |- PPS
#                      |- FREL(1)
#                      |- FREL(2)
scaler_root = './Data/all_scalers.pkl'
print('Loading scaler object from {}'.format(scaler_root))
with open(scaler_root, 'rb') as f:
    scalers = pickle.load(f)

# input_data : saved numpy array for example
# output_data is not required for the inference stage.
# input_data can be csv file but should be converted to numpy array
# shape of input data is (N, d), N is the number of input data

input_root = './Data/input_sample.npy'
input_data = np.load(input_root)
print('Loading input data from {}'.format(input_root))
print('Number of samples in loaded data is {}'.format(len(input_data)))

# output_data = np.load('./Data/output_sample.npy')
print('Scalinng input data with scaler {}'.format(INPUT_SCALE))
scaled_input_data = scalers['input'][INPUT_SCALE].transform(input_data)

#######################################################################
# Model loading part
# set model name which is the key of params dictionary
model_name = 'SMT-MV-QMIXER-V4'
params = {'SMT-MV-QMIXER-V4':{'d':128,'n':2,'h':256}}  #dictionary of hyperparameter for saved model
print('Model name is {}'.format(model_name))

# set best model path 
model_path = './Data/saved_model.ckpt'
print('Load NN model from {}'.format(model_path))
loaded_model = SEP_MTMV_QMLP_MIXER_V4.load_from_checkpoint(checkpoint_path = model_path, c_in = C_IN, d_model = params[model_name]['d'],
            out_len = OUT_LEN, n_layers = params[model_name]['n'], token_hdim = params[model_name]['h'],
            ch_hdim = params[model_name]['h'], dr_rates = 0.2, use_lr_scheduler = False)

###########################################################################
# Inference stage            
# 1. transform numpy data to torch.Tensor
# 2. call loaded_model with input data
# 3. output can be differ according to the model and the current model produces following tuples
# (output_of_median, (output_of_lower_quantile, output_of_upper_quantile))

print('Now Do Inference...')
output = loaded_model(torch.Tensor(scaled_input_data))
true_output = output[0].detach().numpy() # set output of interest to numpy array
denormalized_output = [] # output of neural network is normalized 
for i, variable in enumerate(data_variables):
    output_of_ith_variable = true_output[...,i].reshape(-1,1) # reshape to (L, 1) to use scaler
    denormalized_output_of_ith_variable =  scalers['output'][OUTPUT_SCALE][variable].inverse_transform(output_of_ith_variable)
    reshaped_denormed_output =     denormalized_output_of_ith_variable.reshape(-1,OUT_LEN)
    denormalized_output.append(reshaped_denormed_output)
print('Now Do Inverse Transform')
denormed_output_in_np = np.stack(denormalized_output,axis=-1)
print('Done')
#############################################################################
# result of i-th sample's j-th variable is 
# denormed_output_in_np[i,:,j]
i = 0 # i-th sample
j = 0 # j-th variable
print('sample_index :{}'.format(i))
print('variable_index :{}'.format(i))
result = denormed_output_in_np[i,:,j]
print(result)
