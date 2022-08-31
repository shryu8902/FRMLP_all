#%%
import os
import numpy as np
import pandas as pd
import gc
# import tqdm, glob, pickle, datetime, re, time
# os.environ[ 'MPLCONFIGDIR' ] = './tmp/'

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold       # train, validation, test sets
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # training scalers

import pickle
from utils import *
import utils
import json

import torch
import time
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from Model.MLP_mixer import MLP_MIXER, SEP_MLP_MIXER, SEP_MT_QMLP_MIXER_V2, SEP_MTMV_QMLP_MIXER_V3, SEP_Multitask_MLP_MIXER, SEP_MT_QMLP_MIXER, SEP_MT_QMLP_MIXER_V2, SEP_MTMV_QMLP_MIXER , SEP_MTMV_QMLP_MIXER_V2, SEP_MTMV_QMLP_MIXER_V3
from Model.MLP_mixer import SEP_MTMV_QMLP_MIXER_V4, Q_CONV_MIXER, Q_CONV_MIXER_AVGPOOL, CASCADE_Q_CONV_MIXER_AVGPOOL
from Model.FNN import FNN, LSTM

#%%
#parameter setting
INPUT_SCALE = 'SS' # 'standard' or 'minmax
OUTPUT_SCALE = 'SS'
EPOCH = 200
BATCH = 128
C_IN = 49
OUT_LEN = 333
data_variables = ["PPSTRB(3)", "PPS", "FREL(1)", "FREL(2)"]
loss_weights = [0.5, 0.3, 0.1,0.1]
results = []
params = {'MIXER':{'d':64, 'n':2, 'h':256},
        'SEP_MIXER':{'d':64, 'n':2, 'h':256},
        'LSTM':{'d':256, 'n':1, 'h':256},
        'FNN':{'d':128, 'n':3, 'h':256},
        'SMT-MIXER':{'d':64, 'n':2, 'h':256},
        'SMT-MIXER_V3':{'d':128, 'n':2, 'h':512},
        'SMT-QMIXER':{'d':64, 'n':2, 'h':256},
        'SMT-QMIXER-V2':{'d':64, 'n':2, 'h':256},
        'SMT-MV-QMIXER':{'d':64, 'n':2, 'h':256}, # recon loss, diff loss
        'SMT-MV-QMIXER-V2':{'d':128, 'n':2, 'h':256},  # recon loss
        'SMT-MV-QMIXER-V3':{'d':64, 'n':2, 'h':256},  # recon loss
        'SMT-MV-QMIXER-V4':{'d':128, 'n':2, 'h':256},  # recon loss, diff_loss
        'Q-CONV-MIXER':{'d': 64, 'n':3, 'h':256},
        'Q-CONV-MIXER-V2':{'d': 64, 'n':1, 'h':256},
        'Q-CONV-MIXER-AVGPOOL':{'d': 64, 'n':1, 'h':256},
        'CASC-Q-CONV-MIXER-AVGPOOL':{'d': 64, 'n':1, 'h':256} 
        }
#%%
#load data
np_index = np.load('./Data/all_index.npy').astype(int)
np_inputs = np.load('./Data/all_input.npy')
np_outputs = np.load('./Data/all_output.npy')
with open('./Data/all_scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

scaled_inputs = scalers['input'][INPUT_SCALE].transform(np_inputs)
temp = []
for i, target in enumerate(data_variables):
    a = np_outputs[...,i].reshape(-1,1)
    b = scalers['output'][OUTPUT_SCALE][target].transform(a)
    temp.append(b.reshape(-1,1000,1))
scaled_outputs = np.concatenate(temp,axis=-1)

#%%
skf = StratifiedKFold(n_splits = 5, random_state=0, shuffle = True)

for R_SEED, (tr_ind, te_ind) in enumerate(skf.split(range(len(scaled_inputs)), np_index[:,1])):
    inp_train = scaled_inputs[tr_ind,...]
    inp_ = scaled_inputs[te_ind,...]
    outp_train = scaled_outputs[tr_ind,...] 
    outp_ = scaled_outputs[te_ind,...] 
    ind_ = np_index[te_ind]
    inp_val, inp_test, outp_val, outp_test = train_test_split(inp_, outp_, test_size = 0.5, stratify = ind_[:,1],random_state=0)

    x_tr = np.concatenate([inp_train]*3,axis=0)
    x_val = np.concatenate([inp_val]*3,axis=0)
    x_te = np.concatenate([inp_test]*3,axis=0)
    y_tr = np.concatenate([outp_train[:,::3,:][:,:-1,:],outp_train[:,1::3,:], outp_train[:,2::3,:]])
    y_val = np.concatenate([outp_val[:,::3,:][:,:-1,:],outp_val[:,1::3,:], outp_val[:,2::3,:]])
    y_te = np.concatenate([outp_test[:,::3,:][:,:-1,:],outp_test[:,1::3,:], outp_test[:,2::3,:]])

    for tr_seed in range(5):
        # for model_name in ['MIXER','SEP_MIXER']:
        # for model_name in ['SMT-MV-QMIXER-V4']:
        # for model_name in ['Q-CONV-MIXER']:
        # for model_name in ['Q-CONV-MIXER-V2']:
        # for model_name in ['Q-CONV-MIXER-AVGPOOL']:
        for model_name in ['CASC-Q-CONV-MIXER-AVGPOOL']:
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
            if model_name =='MIXER':
                current_model = MLP_MIXER(c_in = C_IN, d_model = params[model_name]['d'],
                out_len = OUT_LEN, n_layers = params[model_name]['n'], token_hdim = params[model_name]['h'],
                ch_hdim = params[model_name]['h'], dr_rates = 0.2, use_lr_scheduler = True)
            elif model_name =='Q-CONV-MIXER':
                current_model = Q_CONV_MIXER(c_in = C_IN, d_model = params[model_name]['d'],
                out_len = OUT_LEN, n_layers = params[model_name]['n'], token_hdim = params[model_name]['h'],
                ch_hdim = params[model_name]['h'], dr_rates = 0.2, use_lr_scheduler = True)
            elif model_name =='Q-CONV-MIXER-V2':
                current_model = Q_CONV_MIXER(c_in = C_IN, d_model = params[model_name]['d'],
                out_len = OUT_LEN, n_layers = params[model_name]['n'], token_hdim = params[model_name]['h'],
                ch_hdim = params[model_name]['h'], dr_rates = 0.2, use_lr_scheduler = True)
            elif model_name == 'SMT-MV-QMIXER-V4':
                current_model = SEP_MTMV_QMLP_MIXER_V4(c_in = C_IN, d_model = params[model_name]['d'],
                out_len = OUT_LEN, n_layers = params[model_name]['n'], token_hdim = params[model_name]['h'],
                ch_hdim = params[model_name]['h'], dr_rates = 0.2, use_lr_scheduler = True, loss_weight=loss_weights)
            elif model_name =='Q-CONV-MIXER-AVGPOOL':
                current_model = Q_CONV_MIXER_AVGPOOL(c_in = C_IN, d_model = params[model_name]['d'],
                out_len = OUT_LEN, n_layers = params[model_name]['n'], token_hdim = params[model_name]['h'],
                ch_hdim = params[model_name]['h'], dr_rates = 0.2, use_lr_scheduler = True)
            elif model_name =='CASC-Q-CONV-MIXER-AVGPOOL':
                current_model = CASCADE_Q_CONV_MIXER_AVGPOOL(c_in = C_IN, d_model = params[model_name]['d'],
                out_len = OUT_LEN, n_layers = params[model_name]['n'], token_hdim = params[model_name]['h'],
                ch_hdim = params[model_name]['h'], dr_rates = 0.2, use_lr_scheduler = True)


            checkpoint_callback = ModelCheckpoint(dirpath='./EXP1/R{}_in{}_out{}_{}_{}_{}'.format(R_SEED,INPUT_SCALE, OUTPUT_SCALE, 'MV', model_name, str(tr_seed)), save_top_k=1, monitor='val_loss')
            logger = pl.loggers.TensorBoardLogger(save_dir='./logs/EXP1/R{}_in{}_out{}_{}_{}_{}'.format(R_SEED,INPUT_SCALE, OUTPUT_SCALE, 'MV', model_name, str(tr_seed)), version=1, name=str(tr_seed))
            trainer = pl.Trainer(gpus=1, max_epochs = EPOCH,  progress_bar_refresh_rate=2, logger=logger, callbacks=[EarlyStopping(monitor="val_loss",patience=20),checkpoint_callback])

            # best_path = glob.glob('./RE/{}_{}_{}/*.ckpt'.format(target_data,model_name, str(seed)))[0]
            #     trainer = pl.Trainer(gpus=1)
            trainer.fit(current_model, loader_train, loader_val)

            if model_name == 'SMT-MV-QMIXER-V2':
                loaded_model = SEP_MTMV_QMLP_MIXER_V2.load_from_checkpoint(checkpoint_path = checkpoint_callback.best_model_path, c_in = C_IN, d_model = params[model_name]['d'],
                out_len = OUT_LEN, n_layers = params[model_name]['n'], token_hdim = params[model_name]['h'],
                ch_hdim = params[model_name]['h'], dr_rates = 0.2, use_lr_scheduler = True)
            elif model_name == 'SMT-MV-QMIXER-V4':
                loaded_model = SEP_MTMV_QMLP_MIXER_V4.load_from_checkpoint(checkpoint_path = checkpoint_callback.best_model_path, c_in = C_IN, d_model = params[model_name]['d'],
                out_len = OUT_LEN, n_layers = params[model_name]['n'], token_hdim = params[model_name]['h'],
                ch_hdim = params[model_name]['h'], dr_rates = 0.2, use_lr_scheduler = True)
            elif model_name =='Q-CONV-MIXER':
                loaded_model = Q_CONV_MIXER.load_from_checkpoint(checkpoint_path = checkpoint_callback.best_model_path, c_in = C_IN, d_model = params[model_name]['d'],
                out_len = OUT_LEN, n_layers = params[model_name]['n'], token_hdim = params[model_name]['h'],
                ch_hdim = params[model_name]['h'], dr_rates = 0.2, use_lr_scheduler = True)

            elif model_name =='Q-CONV-MIXER-V2':
                loaded_model = Q_CONV_MIXER.load_from_checkpoint(checkpoint_path = checkpoint_callback.best_model_path, c_in = C_IN, d_model = params[model_name]['d'],
                out_len = OUT_LEN, n_layers = params[model_name]['n'], token_hdim = params[model_name]['h'],
                ch_hdim = params[model_name]['h'], dr_rates = 0.2, use_lr_scheduler = True)

            elif model_name =='Q-CONV-MIXER-AVGPOOL':
                loaded_model = Q_CONV_MIXER_AVGPOOL.load_from_checkpoint(checkpoint_path = checkpoint_callback.best_model_path, c_in = C_IN, d_model = params[model_name]['d'],
                out_len = OUT_LEN, n_layers = params[model_name]['n'], token_hdim = params[model_name]['h'],
                ch_hdim = params[model_name]['h'], dr_rates = 0.2, use_lr_scheduler = True)
            elif model_name =='CASC-Q-CONV-MIXER-AVGPOOL':
                loaded_model = CASCADE_Q_CONV_MIXER_AVGPOOL.load_from_checkpoint(checkpoint_path = checkpoint_callback.best_model_path, c_in = C_IN, d_model = params[model_name]['d'],
                out_len = OUT_LEN, n_layers = params[model_name]['n'], token_hdim = params[model_name]['h'],
                ch_hdim = params[model_name]['h'], dr_rates = 0.2, use_lr_scheduler = True)

            strt_time = time.time()
            temp_prediction = trainer.predict(loaded_model, loader_test)    
            end_time = time.time()
            exp_time = round(end_time - strt_time,4)
            Y_HAT = np.concatenate([x[0] for x in temp_prediction])
            Y_TRUE = np.concatenate([x[1] for x in temp_prediction])

            for i, variable in enumerate(data_variables):
                a = Y_HAT[...,i].reshape(-1,1)
                DE_Y_HAT = scalers['output'][OUTPUT_SCALE][variable].inverse_transform(a).reshape(-1,OUT_LEN)
                DE_Y_TRUE = scalers['output'][OUTPUT_SCALE][variable].inverse_transform(Y_TRUE[...,i].reshape(-1,1)).reshape(-1,OUT_LEN)

                mse =  ((DE_Y_HAT - DE_Y_TRUE)**2).mean(axis=1)
                mape = abs((DE_Y_HAT - DE_Y_TRUE)/(DE_Y_TRUE+1e-6)).mean(axis=1)*100
                smape = (abs(DE_Y_HAT - DE_Y_TRUE)/((abs(DE_Y_TRUE)+abs(DE_Y_HAT)))).mean(axis=1)*100
                nmae1 = abs((DE_Y_HAT - DE_Y_TRUE)/(DE_Y_TRUE.mean(axis=1)[...,np.newaxis])).mean(axis=1)*100
                nmae2 = abs((DE_Y_HAT - DE_Y_TRUE)/(DE_Y_TRUE.max(axis=1)[...,np.newaxis])).mean(axis=1)*100

                np.save('./EXP1/R{}_in{}_out{}_{}_{}_{}.npy'.format(R_SEED, INPUT_SCALE, OUTPUT_SCALE, variable, model_name, str(tr_seed)), Y_HAT)
                np.save('./EXP1/R{}_in{}_out{}_{}_{}_true.npy'.format(R_SEED, INPUT_SCALE, OUTPUT_SCALE, variable, model_name), Y_TRUE)

                results.append({'data':variable, 'model':model_name, 
                'in_scale':INPUT_SCALE, 'out_scale':OUTPUT_SCALE,
                'mses':mse.mean(), 'mapes':mape.mean(), 'smapes':smape.mean(),'nmae1':nmae1.mean(), 'nmae2':nmae2.mean(),
                'mses_med':np.median(mse), 'mapes_med':np.median(mape), 'smapes_med':np.median(smape),'nmae1_med':np.median(nmae1), 'nmae2_med':np.median(nmae2),
                'time':exp_time,
                'c_in':C_IN,'R_SEED':R_SEED,'T_SEED':tr_seed,
                'd_model':params[model_name]['d'], 'hdim':params[model_name]['h'], 'nlayer':params[model_name]['n'],
                'batch': BATCH, 'lr':1e-3,
                'mapes_oh':np.sum(mape>100),'smapes_oh':np.sum(smape>100), 'nmae1_oh':np.sum(nmae1>100), 'nmae2_oh':np.sum(nmae2>100)
                })

            del(current_model)
            gc.collect()
            torch.cuda.empty_cache()

df_results = pd.DataFrame.from_dict(results)
# df_results.to_csv('./MV_result_0715.csv')
df_results.to_csv('./CASC_CONV_MIXER_AVGPOOL_result_0727.csv') #loss weight 0.5,0.3, 0.1,0.1

#%%
skf = StratifiedKFold(n_splits = 5, random_state=0, shuffle = True)

for R_SEED, (tr_ind, te_ind) in enumerate(skf.split(range(len(scaled_inputs)), np_index[:,1])):
    inp_train = scaled_inputs[tr_ind,...]
    inp_ = scaled_inputs[te_ind,...]
    outp_train = scaled_outputs[tr_ind,...] 
    outp_ = scaled_outputs[te_ind,...] 
    ind_ = np_index[te_ind]
    inp_val, inp_test, outp_val, outp_test = train_test_split(inp_, outp_, test_size = 0.5, stratify = ind_[:,1],random_state=0)

    x_tr = np.concatenate([inp_train]*3,axis=0)
    x_val = np.concatenate([inp_val]*3,axis=0)
    x_te = np.concatenate([inp_test]*3,axis=0)
    y_tr = np.concatenate([outp_train[:,::3,:][:,:-1,:],outp_train[:,1::3,:], outp_train[:,2::3,:]])
    y_val = np.concatenate([outp_val[:,::3,:][:,:-1,:],outp_val[:,1::3,:], outp_val[:,2::3,:]])
    y_te = np.concatenate([outp_test[:,::3,:][:,:-1,:],outp_test[:,1::3,:], outp_test[:,2::3,:]])

    for tr_seed in range(5):
        # for model_name in ['MIXER','SEP_MIXER']:
        # for model_name in ['SMT-MV-QMIXER-V4']:
        # for model_name in ['Q-CONV-MIXER']:
        # for model_name in ['Q-CONV-MIXER-V2']:
        # for model_name in ['Q-CONV-MIXER-AVGPOOL']:
        # for model_name in ['CASC-Q-CONV-MIXER-AVGPOOL']:
        for model_name in ['SMT-MV-QMIXER-V2','SMT-MV-QMIXER-V4','Q-CONV-MIXER','Q-CONV-MIXER-V2','Q-CONV-MIXER-AVGPOOL','CASC-Q-CONV-MIXER-AVGPOOL']:

            utils.seed_everything(tr_seed)

            '''
            Prepare Input data (scenarios)
            '''
            
            # train = TensorDataset(torch.Tensor(x_tr), torch.Tensor(y_tr))
            # val = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
            # test = TensorDataset(torch.Tensor(x_te), torch.Tensor(y_te))

            # loader_train = DataLoader(train, batch_size = BATCH, shuffle =True)
            # loader_val = DataLoader(val, batch_size= BATCH, shuffle = False)
            # loader_test = DataLoader(test, batch_size= BATCH, shuffle = False)


            for i, variable in enumerate(data_variables):
                Y_TRUE = np.load('./EXP1/R{}_in{}_out{}_{}_{}_true.npy'.format(R_SEED, INPUT_SCALE, OUTPUT_SCALE, variable, 'SMT-MV-QMIXER-V2'))
                Y_HAT = np.load('./EXP1/R{}_in{}_out{}_{}_{}_{}.npy'.format(R_SEED, INPUT_SCALE, OUTPUT_SCALE, variable, model_name, str(tr_seed))) 
                a = Y_HAT[...,i].reshape(-1,1)
                DE_Y_HAT = scalers['output'][OUTPUT_SCALE][variable].inverse_transform(a).reshape(-1,OUT_LEN)
                DE_Y_TRUE = scalers['output'][OUTPUT_SCALE][variable].inverse_transform(Y_TRUE[...,i].reshape(-1,1)).reshape(-1,OUT_LEN)



                mse =  ((DE_Y_HAT - DE_Y_TRUE)**2).mean(axis=1)
                mape = abs((DE_Y_HAT - DE_Y_TRUE)/(DE_Y_TRUE+1e-6)).mean(axis=1)*100
                smape = (abs(DE_Y_HAT - DE_Y_TRUE)/((abs(DE_Y_TRUE)+abs(DE_Y_HAT)))).mean(axis=1)*100
                nmae1 = abs((DE_Y_HAT - DE_Y_TRUE)/(DE_Y_TRUE.mean(axis=1)[...,np.newaxis])).mean(axis=1)*100
                nmae2 = abs((DE_Y_HAT - DE_Y_TRUE)/(DE_Y_TRUE.max(axis=1)[...,np.newaxis])).mean(axis=1)*100


                results.append({'data':variable, 'model':model_name, 
                'in_scale':INPUT_SCALE, 'out_scale':OUTPUT_SCALE,
                'mses':mse.mean(), 'mapes':mape.mean(), 'smapes':smape.mean(),'nmae1':nmae1.mean(), 'nmae2':nmae2.mean(),
                'mses_med':np.median(mse), 'mapes_med':np.median(mape), 'smapes_med':np.median(smape),'nmae1_med':np.median(nmae1), 'nmae2_med':np.median(nmae2),
                # 'time':exp_time,
                'c_in':C_IN,'R_SEED':R_SEED,'T_SEED':tr_seed,
                'd_model':params[model_name]['d'], 'hdim':params[model_name]['h'], 'nlayer':params[model_name]['n'],
                'batch': BATCH, 'lr':1e-3,
                'mapes_oh':np.sum(mape>100),'smapes_oh':np.sum(smape>100), 'nmae1_oh':np.sum(nmae1>100), 'nmae2_oh':np.sum(nmae2>100)
                })


df_results = pd.DataFrame.from_dict(results)
# df_results.to_csv('./MV_result_0715.csv')
# df_results.to_csv('./CASC_CONV_MIXER_AVGPOOL_result_0727.csv') #loss weight 0.5,0.3, 0.1,0.1

df_results.to_csv('./nmae.csv') #loss weight 0.5,0.3, 0.1,0.1
