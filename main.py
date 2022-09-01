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
INPUT_SCALE = 'SS' # 'SS' for standard scaler, 'MM' for minmax
OUTPUT_SCALE = 'SS' # 'SS' for standard scaler, 'MM' for minmax, 

#C_IN and OUT_LEN can be changed to other values according to the model training
C_IN = 49  # Number of input channel
OUT_LEN = 333 # Length of output time series data 
data_variables = ['PPSTRB(3)','PPS','FREL(1)','FREL(2)'] # List of output variables in order
# neural network model outputs [N, OUT_LEN, len(data_variables)] where N is the number of input for neural network
params = {'SMT-MV-QMIXER-V4':{'d':128,'n':2,'h':256}}  #dictionary of hyperparameter for saved model

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

with open('./Data/all_scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

# input_data : saved numpy array for example
# output_data is not required for the inference stage.
# input_data can be csv file but should be converted to numpy array
# shape of input data is (N, d), N is the number of input data

input_data = np.load('./Data/input_sample.npy')
# output_data = np.load('./Data/output_sample.npy')
scaled_input_data = scalers['input'][INPUT_SCALE].transform(input_data)

#######################################################################
# Model loading part
# set model name which is the key of params dictionary
model_name = 'SMT-MV-QMIXER-V4'
# set best model path 


model_path = './Data/saved_model.ckpt'
loaded_model = SEP_MTMV_QMLP_MIXER_V4.load_from_checkpoint(checkpoint_path = model_path, c_in = C_IN, d_model = params[model_name]['d'],
            out_len = OUT_LEN, n_layers = params[model_name]['n'], token_hdim = params[model_name]['h'],
            ch_hdim = params[model_name]['h'], dr_rates = 0.2, use_lr_scheduler = False)

###########################################################################
# Inference stage            
# 1. transform numpy data to torch.Tensor
# 2. call loaded_model with input data
# 3. output can be differ according to the model and the current model produces following tuples
# (output_of_median, (output_of_lower_quantile, output_of_upper_quantile))

output = loaded_model(torch.Tensor(scaled_input_data))
true_output = output[0].detach().numpy() # set output of interest to numpy array
denormalized_output = [] # output of neural network is normalized 
for i, variable in enumerate(data_variables):
    output_of_ith_variable = true_output[...,i].reshape(-1,1) # reshape to (L, 1) to use scaler
    denormalized_output_of_ith_variable =  scalers['output'][OUTPUT_SCALE][variable].inverse_transform(output_of_ith_variable)
    reshaped_denormed_output =     denormalized_output_of_ith_variable.reshape(-1,OUT_LEN)
    denormalized_output.append(reshaped_denormed_output)
denormed_output_in_np = np.stack(denormalized_output,axis=-1)

#############################################################################
# result of i-th sample's j-th variable is 
# denormed_output_in_np[i,:,j]
i = 0 # i-th sample
j = 0 # j-th variable
result = denormed_output_in_np[i,:,j]
