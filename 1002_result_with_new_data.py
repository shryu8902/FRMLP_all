#%%
# library for data
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
#%%
params = {'Hype-Q-CONV-MIXER':{'d':256, 'n':2, 'h':128, 'lr':0.001, 'd_emb':32}}
C_IN = 48
OUT_LEN = 333
EPOCH = 200
BATCH = 128

#%%

skf = StratifiedKFold(n_splits = 5, random_state=0, shuffle = True)
results = []

for R_SEED, (tr_ind, te_ind) in enumerate(skf.split(range(len(np_scaled_inputs)), np_index[:,0])):
    inp_train = np_scaled_inputs[tr_ind,...]
    inp_ = np_scaled_inputs[te_ind,...]
    outp_train = np_scaled_outputs[tr_ind,...] 
    outp_ = np_scaled_outputs[te_ind,...] 
    ind_ = np_index[te_ind]
    inp_val, inp_test, outp_val, outp_test = train_test_split(inp_, outp_, test_size = 0.5, stratify = ind_[:,0],random_state=0)

    x_tr = np.concatenate([inp_train]*3,axis=0)
    x_val = np.concatenate([inp_val]*3,axis=0)
    x_te = np.concatenate([inp_test]*3,axis=0)
    y_tr = np.concatenate([outp_train[:,::3,:][:,:-1,:],outp_train[:,1::3,:], outp_train[:,2::3,:]])
    y_val = np.concatenate([outp_val[:,::3,:][:,:-1,:],outp_val[:,1::3,:], outp_val[:,2::3,:]])
    y_te = np.concatenate([outp_test[:,::3,:][:,:-1,:],outp_test[:,1::3,:], outp_test[:,2::3,:]])

    for tr_seed in range(5):
        for model_name in ['Hype-Q-CONV-MIXER']:
            utils.seed_everything(tr_seed)

            '''
            Prepare Input data (scenarios)
            '''
            
            train = TensorDataset(torch.Tensor(x_tr), torch.Tensor(y_tr))
            val = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
            test = TensorDataset(torch.Tensor(x_te), torch.Tensor(y_te))

            loader_train = DataLoader(train, batch_size = BATCH, shuffle =True)
            loader_val = DataLoader(val, batch_size= BATCH, shuffle = False)
            loader_test = DataLoader(test, batch_size= BATCH, shuffle = False)

            #Prepare model
            current_model = Q_CONV_MIXER_all(c_in = C_IN, d_model = params[model_name]['d'], d_emb = params[model_name]['d_emb'],
            out_len = OUT_LEN, n_layers = params[model_name]['n'], token_hdim = params[model_name]['h'], loss_weight=[1,1,1,1,1,1,1],
            ch_hdim = params[model_name]['h'], dr_rates = 0.2, use_lr_scheduler = True, lr =params[model_name]['lr'])

            checkpoint_callback = ModelCheckpoint(dirpath='./EXP2/R{}_T{}_in{}_out{}_{}'.format(R_SEED,str(tr_seed),INPUT_SCALE, OUTPUT_SCALE,  model_name), save_top_k=1, monitor='val_loss')
            logger = pl.loggers.TensorBoardLogger(save_dir='./logs/EXP2/R{}_T{}_in{}_out{}_{}'.format(R_SEED,str(tr_seed),INPUT_SCALE, OUTPUT_SCALE, model_name), version=1, name=str(tr_seed))
            trainer = pl.Trainer(gpus = 1, max_epochs = EPOCH, progress_bar_refresh_rate = 2, logger = logger,
                        callbacks = [EarlyStopping(monitor="val_loss",patience=20),checkpoint_callback])
            trainer.fit(current_model, loader_train, loader_val)


            loaded_model = Q_CONV_MIXER_all.load_from_checkpoint(checkpoint_path = checkpoint_callback.best_model_path, 
                                        c_in = C_IN, d_model = params[model_name]['d'],d_emb = params[model_name]['d_emb'],
            out_len = OUT_LEN, n_layers = params[model_name]['n'], token_hdim = params[model_name]['h'],loss_weight=[1,1,1,1,1,1,1],
            ch_hdim = params[model_name]['h'], dr_rates = 0.2, use_lr_scheduler = True, lr =params[model_name]['lr'])

            temp_prediction = trainer.predict(loaded_model, loader_test)    
            Y_HAT = np.concatenate([x[0] for x in temp_prediction])
            Y_TRUE = np.concatenate([x[1] for x in temp_prediction])
            np.save('./EXP2/R{}_T{}_in{}_out{}_{}'.format(R_SEED,str(tr_seed),INPUT_SCALE, OUTPUT_SCALE,  model_name), Y_HAT)
            np.save('./EXP2/R{}_in{}_out{}_{}_true.npy'.format(R_SEED, INPUT_SCALE, OUTPUT_SCALE,  model_name), Y_TRUE)

            for i, variable in enumerate(data_variables):
                a = Y_HAT[...,i].reshape(-1,1)
                DE_Y_HAT = scalers['output'][OUTPUT_SCALE][variable].inverse_transform(a).reshape(-1,OUT_LEN)
                DE_Y_TRUE = scalers['output'][OUTPUT_SCALE][variable].inverse_transform(Y_TRUE[...,i].reshape(-1,1)).reshape(-1,OUT_LEN)
                TEMP_DE_Y_HAT = DE_Y_HAT.copy()
                TEMP_DE_Y_HAT[TEMP_DE_Y_HAT<1e-3]=0
                mse =  ((DE_Y_HAT - DE_Y_TRUE)**2).mean(axis=1)
                mape = abs((DE_Y_HAT - DE_Y_TRUE)/(DE_Y_TRUE+DE_Y_TRUE*1e-6)).mean(axis=1)*100
                mape_drop = abs((TEMP_DE_Y_HAT - DE_Y_TRUE)/(DE_Y_TRUE+DE_Y_TRUE*1e-6)).mean(axis=1)*100
                smape = (abs(DE_Y_HAT - DE_Y_TRUE)/((abs(DE_Y_TRUE)+abs(DE_Y_HAT)))).mean(axis=1)*100
                nmae1 = abs((DE_Y_HAT - DE_Y_TRUE)/(DE_Y_TRUE.mean(axis=1)[...,np.newaxis])).mean(axis=1)*100

                results.append({'data':variable, 'model':model_name, 
                'in_scale':INPUT_SCALE, 'out_scale':OUTPUT_SCALE,
                'c_in':C_IN,'R_SEED':R_SEED,'T_SEED':tr_seed,
                'mses':mse.mean(), 'mapes':mape.mean(), 'mapes_drop':mape_drop.mean(), 'smapes':smape.mean(),'nmae_mean':nmae1.mean(),
                'mses_med':np.median(mse), 'mapes_med':np.median(mape), 'mapes_drop_med':np.median(mape_drop),'smapes_med':np.median(smape),'nmae_mean_med':np.median(nmae1),
                'd_model':params[model_name]['d'], 'hdim':params[model_name]['h'], 'nlayer':params[model_name]['n'],
                'batch': BATCH, 'lr':params[model_name]['lr'],
                'last_epoch': trainer.current_epoch
                })
            del(current_model)
    gc.collect()
    torch.cuda.empty_cache()
df_results = pd.DataFrame.from_dict(results)
df_results.to_csv('./result_new_data.csv') #loss weight 0.5,0.3, 0.1,0.1

