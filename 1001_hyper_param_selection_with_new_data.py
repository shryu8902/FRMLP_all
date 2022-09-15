#%%
# library for data
from decimal import MAX_PREC
import os
import numpy as np
import pandas as pd

# library for data analytics

# library for deep learning
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from Model.MLP_mixer import Q_CONV_MIXER
from Model.CONV_MIXER import Q_CONV_MIXER_all
# from Model.FNN import FNN, LSTM

import gc

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold       # train, validation, test sets
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # training scalers

import pickle
from utils import *
import utils
import json

import time
#%%
BATCH = 256
np_inputs = np.load('./Data/new_data/all_input_v2.npy')
np_index = np.load('./Data/new_data/all_index_v2.npy').astype(int)
np_outputs = np.load('./Data/new_data/all_output_v2.npy')

with open('./Data/new_data/all_scalers_v2.pkl', 'rb') as f:
    scalers = pickle.load(f)
INPUT_SCALE = 'SS'
OUTPUT_SCALE = 'SS'
data_variables = ["PEX0(3)","PPS","FREL(1)","FREL(2)","WWINJ(11)","WSPTA","TCREXIT"]
np_scaled_inputs = scalers['input'][INPUT_SCALE].transform(np_inputs)
temp = []
for i, target in enumerate(data_variables):
    a = np_outputs[...,i].reshape(-1,1)
    b = scalers['output'][OUTPUT_SCALE][target].transform(a)
    temp.append(b.reshape(-1,1000,1))
np_scaled_outputs = np.concatenate(temp,axis=-1)

train_input, test_input, train_index, test_index, train_output,test_output = train_test_split(np_scaled_inputs,np_index,np_scaled_outputs, test_size=0.2, random_state=0, stratify=np_index[:,0])

x_tr = np.concatenate([train_input]*3,axis=0)
# x_te = np.concatenate([test_input]*3,axis=0)
x_te = test_input
y_tr = np.concatenate([train_output[:,::3,:][:,:-1,:],train_output[:,1::3,:], train_output[:,2::3,:]])
# y_te = np.concatenate([test_output[:,::3,:][:,:-1,:],test_output[:,1::3,:], test_output[:,2::3,:]])
y_te = test_output[:,::3,:][:,:-1,:]
train = TensorDataset(torch.Tensor(x_tr), torch.Tensor(y_tr))
test = TensorDataset(torch.Tensor(x_te), torch.Tensor(y_te))

loader_train = DataLoader(train, batch_size = BATCH, shuffle =True)
loader_test = DataLoader(test, batch_size= BATCH, shuffle = False)
#%%
C_IN = 48
OUT_LEN = 333
EPOCH = 20
results = []
d_models = [64,128,256,512]
n_layers = [1,2,3,5]
h_dims = [64,128]
lr_lists = [0.01,0.001,0.0001]
d_emb = 32
model_name = 'Q-CONV-MIXER'

for d in d_models:
    for n in n_layers:
        for h in h_dims:
            for lr in lr_lists:
                print(d, n, h, lr)
                utils.seed_everything(0)
                current_model = Q_CONV_MIXER_all(c_in = C_IN, d_model = d, d_emb = d_emb,
                out_len = OUT_LEN, n_layers = n, token_hdim = h, loss_weight=[1,1,1,1,1,1,1],
                ch_hdim = h, dr_rates = 0.2, use_lr_scheduler = False, lr =lr)

                checkpoint_callback = ModelCheckpoint(dirpath = './HYPE_ALL/d{}_n{}_h{}_lr{}'.format(d,n,h,lr))
                logger = pl.loggers.TensorBoardLogger(save_dir='./logs/HYPE_ALL/d{}_n{}_h{}_lr{}'.format(d,n,h,lr))
                trainer = pl.Trainer(gpus = 1, max_epochs = EPOCH, progress_bar_refresh_rate = 2, logger = logger,
                            callbacks = [checkpoint_callback])
                trainer.fit(current_model, loader_train)

                temp_prediction = trainer.predict(current_model, loader_test)    
                Y_HAT = np.concatenate([x[0] for x in temp_prediction])
                Y_TRUE = np.concatenate([x[1] for x in temp_prediction])

                se = (Y_HAT-Y_TRUE)**2
                mse = se.mean(axis=-1).mean(axis=-1)
                results.append({'model':model_name, 'in_scale':INPUT_SCALE, 'out_scale':OUTPUT_SCALE, 
                'mses':mse.mean(), 'mses_med':np.median(mse),
                'd_emb':d_emb, 'd_model':d, 'hdim':h, 'n_layer':n ,'lr':lr, 'epoch':EPOCH, 'batch':BATCH})
                del(current_model)
            gc.collect()
            torch.cuda.empty_cache()
df_results = pd.DataFrame.from_dict(results)
df_results.to_csv('./GridSearch.csv') #loss weight 0.5,0.3, 0.1,0.1

