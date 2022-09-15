#%%
# library for data
import os
import numpy as np
import pandas as pd

# library for data analytics
import tqdm

# library for deep learning
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from Model.REGR_TRNSFMR import AR_TRANSFMR, AR_TRANSFMR_V2
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
params = {'AR-TRNSFMR':{'d':128, 'n':2, 'h':128, 'lr':0.001, 'd_emb':32}}
params = {'AR-TRNSFMR-V2':{'d':128, 'n':2, 'h':128, 'lr':0.001, 'd_emb':32}}

C_IN = 48
D_OUT = 7
D_IN = C_IN + D_OUT
# OUT_LEN = 333 
OUT_LEN = 100
# OUT_LEN = 10
EPOCH = 100
BATCH = 256
#%%
def train_concatenator(input_vec, output_seq):
    N, L, D = output_seq.shape
    input_seq = np.tile(input_vec[:,np.newaxis,:],(1,L,1))
    strt_tkn = -2*np.ones((N,1,D))
    outwtkn = np.concatenate((strt_tkn, output_seq), axis = 1)[:,:-1,:]
    new_input = np.concatenate((input_seq, outwtkn), axis = -1)
    return new_input, output_seq
def split_given_size(a, size):
    return np.split(a, np.arange(size,len(a),size))
#%%
skf = StratifiedKFold(n_splits = 5, random_state=0, shuffle = True)
results = []

for R_SEED, (tr_ind, te_ind) in enumerate(skf.split(range(len(np_scaled_inputs)), np_index[:,0])):
    if R_SEED == 0:
        inp_train = np_scaled_inputs[tr_ind,...]
        inp_ = np_scaled_inputs[te_ind,...]
        outp_train = np_scaled_outputs[tr_ind,...] 
        outp_ = np_scaled_outputs[te_ind,...] 
        ind_ = np_index[te_ind]
        inp_val, inp_test, outp_val, outp_test = train_test_split(inp_, outp_, test_size = 0.5, stratify = ind_[:,0],random_state=0)

        if OUT_LEN == 333:
            x_tr = np.concatenate([inp_train]*3,axis=0)
            y_tr = np.concatenate([outp_train[:,::3,:][:,:-1,:],outp_train[:,1::3,:], outp_train[:,2::3,:]])
            new_x_tr, new_y_tr = train_concatenator(x_tr, y_tr)
            
            x_val = np.concatenate([inp_val]*3,axis=0)
            y_val = np.concatenate([outp_val[:,::3,:][:,:-1,:],outp_val[:,1::3,:], outp_val[:,2::3,:]])
            new_x_val, new_y_val = train_concatenator(x_val, y_val)

            x_te = np.concatenate([inp_test]*3,axis=0)
            y_te = np.concatenate([outp_test[:,::3,:][:,:-1,:],outp_test[:,1::3,:], outp_test[:,2::3,:]])
        elif OUT_LEN == 100:
            x_tr = np.concatenate([inp_train]*10,axis=0)
            y_tr = np.concatenate([outp_train[:,k::10,:] for k in range(10)])
            new_x_tr, new_y_tr = train_concatenator(x_tr, y_tr)
            
            x_val = np.concatenate([inp_val]*10,axis=0)
            y_val = np.concatenate([outp_val[:,k::10,:] for k in range(10)])
            new_x_val, new_y_val = train_concatenator(x_val, y_val)

            x_te = np.concatenate([inp_test],axis=0)
            # y_te = np.concatenate([outp_test[:,::10,:] for k in range(10)])
            y_te = outp_test[:,::10,:]

        for tr_seed in range(1):
            # for model_name in ['AR-TRNSFMR']:
            for model_name in ['AR-TRNSFMR-V2']:

                utils.seed_everything(tr_seed)

                train = TensorDataset(torch.Tensor(new_x_tr), torch.Tensor(new_y_tr))
                val = TensorDataset(torch.Tensor(new_x_val), torch.Tensor(new_y_val))

                loader_train = DataLoader(train, batch_size = BATCH, shuffle =True)
                loader_val = DataLoader(val, batch_size= BATCH, shuffle = False)

                current_model = AR_TRANSFMR(c_in = D_IN, d_model = params[model_name]['d'], d_emb = params[model_name]['d_emb'],
                            n_out = D_OUT, out_len = OUT_LEN, n_layers = params[model_name]['n'], n_head = 8, loss_weight=[1,1,1,1,1,1,1],
                            dr_rates = 0.2, use_lr_scheduler = True, lr =params[model_name]['lr'])
                checkpoint_callback = ModelCheckpoint(dirpath='./EXP3/R{}_T{}_in{}_out{}_{}'.format(R_SEED,str(tr_seed),INPUT_SCALE, OUTPUT_SCALE,  model_name), save_top_k=1, monitor='val_loss')
                logger = pl.loggers.TensorBoardLogger(save_dir='./logs/EXP3/R{}_T{}_in{}_out{}_{}'.format(R_SEED,str(tr_seed),INPUT_SCALE, OUTPUT_SCALE, model_name), version=1, name=str(tr_seed))
                trainer = pl.Trainer(gpus = 1, max_epochs = EPOCH, progress_bar_refresh_rate = 2, logger = logger,
                            callbacks = [EarlyStopping(monitor="val_loss",patience=20),checkpoint_callback])
                trainer.fit(current_model, loader_train, loader_val)
    
                list_x_te_bathces = split_given_size(x_te, BATCH)
                list_y_te_hat = []
                for x_te_batch in list_x_te_bathces:
                    N,_ = x_te_batch.shape
                    for i in tqdm.tqdm(range(OUT_LEN)):
                        temp_input_seq = np.tile(x_te_batch[:,np.newaxis,:],(1,i+1,1))
                        if i == 0:
                            strt_tkn = -2 * np.ones((N,1,D_OUT))
                            outwtkn = strt_tkn
                        temp_x_te_input = np.concatenate((temp_input_seq, outwtkn), axis = -1)
                        out, _ = current_model(torch.Tensor(temp_x_te_input))
                        outwtkn = np.concatenate((outwtkn, out[:,-1,:].unsqueeze(1).cpu().detach().numpy()), axis= 1)
                    list_y_te_hat.append(out)

                list_y_te_hat = [kk.detach().numpy() for kk in list_y_te_hat]
                Y_HAT =  np.concatenate(list_y_te_hat, axis = 0)
                Y_TRUE = y_te
                np.save('./EXP3/R{}_T{}_in{}_out{}_{}'.format(R_SEED,str(tr_seed),INPUT_SCALE, OUTPUT_SCALE,  model_name), Y_HAT)
                np.save('./EXP3/R{}_in{}_out{}_{}_true.npy'.format(R_SEED, INPUT_SCALE, OUTPUT_SCALE,  model_name), Y_TRUE)
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
df_results.to_csv('./AR_TRNSFMR_V2_result_new_data.csv') #loss weight 0.5,0.3, 0.1,0.1
# #%%
# DENORM_Y_HAT = []
# DENORM_Y_TRUE = []
# for i, variable in enumerate(data_variables):
#     a = Y_HAT[...,i].reshape(-1,1)
#     DE_Y_HAT = scalers['output'][OUTPUT_SCALE][variable].inverse_transform(a).reshape(-1,OUT_LEN)
#     DE_Y_TRUE = scalers['output'][OUTPUT_SCALE][variable].inverse_transform(Y_TRUE[...,i].reshape(-1,1)).reshape(-1,OUT_LEN)
#     DENORM_Y_HAT.append(DE_Y_HAT)
#     DENORM_Y_TRUE.append(DE_Y_TRUE)

# DENORM_Y_HAT = np.stack(DENORM_Y_HAT, axis = -1)
# DENORM_Y_TRUE = np.stack(DENORM_Y_TRUE, axis = -1)
# DENORM_Y_TRUE = abs(DENORM_Y_TRUE)
# DENORM_Y_HAT[DENORM_Y_HAT<0]=1e-12
#%%
# from matplotlib import gridspec
# import tqdm
# for j, variable in enumerate(data_variables):
#     fig = plt.figure(figsize=(15, 6), facecolor='white') 
#     gs = gridspec.GridSpec(nrows=1, # row 몇 개 
#                        ncols=2, # col 몇 개 
#                        height_ratios=[1], 
#                        width_ratios=[1, 1]
#                       )
#     for i in tqdm.tqdm(range(len(DENORM_Y_HAT))):
#     # for i in range(20):
#         ax1 = plt.subplot(gs[0])
#         ax1.plot(DENORM_Y_TRUE[i,...,j])
#         ax2 = plt.subplot(gs[1])
#         ax2.plot(DENORM_Y_HAT[i,...,j])
#     ax1.set_title('{} (True)'.format(variable))
#     ax2.set_title('{} (Pred)'.format(variable))
#     ax2.set_ylim(ax1.get_ylim())
#     plt.tight_layout()
#     plt.savefig('./IMAGES/AR_full_examples_{}.png'.format(variable),dpi=300, transparent=False)
#     plt.close()
