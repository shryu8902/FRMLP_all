#%%
import numpy as np
import pandas as pd
from torch.nn.modules.dropout import Dropout
# pd.options.display.float_format = '{:,.6f}'.format 

#torch related
import torch
from torch import nn
import torch.nn.functional as F
from utils import PinballLoss
#pytorch lightning
import pytorch_lightning as pl

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from Model.embed import DataEmbedding, SeperateDataEmbedding, SeperateDataEmbedding_V2
from Model.common_model import LTS_model
from Model.TCN import TemporalBlock
# %%
class MLP(nn.Module):
    def __init__(self, dim, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

class MixerLayer(nn.Module):
    def __init__(self, token_dim, ch_dim, token_hdim = 256, ch_hdim = 256):
        super().__init__()
        self.token_dim = token_dim
        self.ch_dim = ch_dim
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim

        self.token_mix = nn.Sequential(
            nn.LayerNorm(ch_dim),
            Rearrange('b n d -> b d n'), # (batch, length, depth) -> (batch, depth, length)
            MLP(token_dim, hidden_size=token_hdim),#             output: (batch, depth, length)
            Rearrange('b d n -> b n d')  # (batch, depth, length) -> (batch, length, depth)
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(ch_dim),
            MLP(ch_dim, hidden_size=ch_hdim)    # output : (batch, length, depth)
        )
        
    def forward(self, x): 
        x = x + self.token_mix(x) 
        x = x + self.channel_mix(x)
        
        return x

class Q_CONV_MIXER_all(LTS_model):
    '''
    Separte positional encoding,
    Multitask loss
    Multivariate outputs
    quantiles per variable
    weighted loss function
    '''
    def __init__(self, c_in, d_model, d_emb, out_len = 333, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, loss_weight = [1,1,1,1], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.out_len = out_len
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers = n_layers
        self.dr_rates = dr_rates
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = SeperateDataEmbedding_V2(c_in, d_model, d_emb = d_emb)
        self.arrange1 = Rearrange('b d n -> b n d') # (batch, depth, length) -> (batch, length, depth)
        self.arrange2 = Rearrange('b n d -> b d n') # (batch, length, depth) -> (batch, depth, length)
        self.d_lat = self.input_embedding.d_lat
        self.loss_weight = loss_weight

        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.out_len//3, ch_dim=self.d_lat, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        self.conv_head = nn.ConvTranspose1d(self.d_lat, 7, kernel_size = 3, stride = 3)
        self.conv_head_u = nn.ConvTranspose1d(self.d_lat, 7, kernel_size = 3, stride = 3)
        self.conv_head_l = nn.ConvTranspose1d(self.d_lat, 7, kernel_size = 3, stride = 3)

    def forward(self, x):
        x = x.unsqueeze(-1).repeat(1,1,self.out_len//3) # X :  b x d x n
        x = self.arrange1(x) #X : b x n x d
        x = self.input_embedding(x)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = self.arrange2(x) # x : b x d x n

        y_preds = self.arrange1(self.conv_head(x))
        y_pred_us = self.arrange1(self.conv_head_u(x))
        y_pred_ls = self.arrange1(self.conv_head_l(x))
        return y_preds, (y_pred_ls,y_pred_us)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hats, (y_hat_ls,y_hat_us) = self(x)

        losses = []
        loss_func = PinballLoss()
        
        for i in range(7):
            # recon pinball loss
            losses.append(self.loss_weight[i]*loss_func(y_hat_ls[...,i], y[...,i], 0.1)) 
            losses.append(self.loss_weight[i]*loss_func(y_hats[...,i], y[...,i], 0.5)) 
            losses.append(self.loss_weight[i]*loss_func(y_hat_us[...,i], y[...,i], 0.9)) 

            # differntial loss 
            y_diff_l = torch.diff(y_hat_ls[...,i])
            y_diff_u = torch.diff(y_hat_us[...,i])
            y_diff_m = torch.diff(y_hats[...,i])
            y_diff = torch.diff(y[...,i])
            losses.append(self.loss_weight[i]*loss_func(y_diff_l, y_diff, 0.1))
            losses.append(self.loss_weight[i]*loss_func(y_diff_m, y_diff, 0.5))
            losses.append(self.loss_weight[i]*loss_func(y_diff_u, y_diff, 0.9))

        loss = torch.stack(losses).sum()

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hats, (y_hat_ls,y_hat_us) = self(x)

        losses = []
        loss_func = PinballLoss()
        for i in range(7):
            # recon pinball loss
            losses.append(self.loss_weight[i]*loss_func(y_hat_ls[...,i], y[...,i], 0.1)) 
            losses.append(self.loss_weight[i]*loss_func(y_hats[...,i], y[...,i], 0.5)) 
            losses.append(self.loss_weight[i]*loss_func(y_hat_us[...,i], y[...,i], 0.9)) 

            # differntial loss 
            y_diff_l = torch.diff(y_hat_ls[...,i])
            y_diff_u = torch.diff(y_hat_us[...,i])
            y_diff_m = torch.diff(y_hats[...,i])
            y_diff = torch.diff(y[...,i])
            losses.append(self.loss_weight[i]*loss_func(y_diff_l, y_diff, 0.1))
            losses.append(self.loss_weight[i]*loss_func(y_diff_m, y_diff, 0.5))
            losses.append(self.loss_weight[i]*loss_func(y_diff_u, y_diff, 0.9))


        loss = torch.stack(losses).sum()
        self.log('val_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hats, (y_hat_ls,y_hat_us) = self(x)
        return y_hats, y
#%%
class Q_CONV_MIXER_all_reslink(LTS_model):
    '''
    Separte positional encoding,
    Multitask loss
    Multivariate outputs
    quantiles per variable
    weighted loss function
    '''
    def __init__(self, c_in, d_model, d_emb, out_len = 333, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, loss_weight = [1,1,1,1], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.out_len = out_len
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers = n_layers
        self.dr_rates = dr_rates
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = SeperateDataEmbedding_V2(c_in, d_model, d_emb = d_emb)
        self.arrange1 = Rearrange('b d n -> b n d') # (batch, depth, length) -> (batch, length, depth)
        self.arrange2 = Rearrange('b n d -> b d n') # (batch, length, depth) -> (batch, depth, length)
        self.d_lat = self.input_embedding.d_lat
        self.loss_weight = loss_weight

        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.out_len//3, ch_dim=self.d_lat, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        self.conv_head = nn.ConvTranspose1d(self.d_lat+self.c_in, 7, kernel_size = 3, stride = 3)
        self.conv_head_u = nn.ConvTranspose1d(self.d_lat+self.c_in, 7, kernel_size = 3, stride = 3)
        self.conv_head_l = nn.ConvTranspose1d(self.d_lat+self.c_in, 7, kernel_size = 3, stride = 3)

    def forward(self, x):
        x = x.unsqueeze(-1).repeat(1,1,self.out_len//3) # X :  b x d x n
        raw_seq = self.arrange1(x) #X : b x n x d

        x = self.input_embedding(raw_seq)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = torch.cat((x, raw_seq), dim = -1)
        x = self.arrange2(x) # x : b x d x n
        y_preds = self.arrange1(self.conv_head(x))
        y_pred_us = self.arrange1(self.conv_head_u(x))
        y_pred_ls = self.arrange1(self.conv_head_l(x))
        return y_preds, (y_pred_ls,y_pred_us)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hats, (y_hat_ls,y_hat_us) = self(x)

        losses = []
        loss_func = PinballLoss()
        
        for i in range(7):
            # recon pinball loss
            losses.append(self.loss_weight[i]*loss_func(y_hat_ls[...,i], y[...,i], 0.1)) 
            losses.append(self.loss_weight[i]*loss_func(y_hats[...,i], y[...,i], 0.5)) 
            losses.append(self.loss_weight[i]*loss_func(y_hat_us[...,i], y[...,i], 0.9)) 

            # differntial loss 
            y_diff_l = torch.diff(y_hat_ls[...,i])
            y_diff_u = torch.diff(y_hat_us[...,i])
            y_diff_m = torch.diff(y_hats[...,i])
            y_diff = torch.diff(y[...,i])
            losses.append(self.loss_weight[i]*loss_func(y_diff_l, y_diff, 0.1))
            losses.append(self.loss_weight[i]*loss_func(y_diff_m, y_diff, 0.5))
            losses.append(self.loss_weight[i]*loss_func(y_diff_u, y_diff, 0.9))

        loss = torch.stack(losses).sum()

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hats, (y_hat_ls,y_hat_us) = self(x)

        losses = []
        loss_func = PinballLoss()
        for i in range(7):
            # recon pinball loss
            losses.append(self.loss_weight[i]*loss_func(y_hat_ls[...,i], y[...,i], 0.1)) 
            losses.append(self.loss_weight[i]*loss_func(y_hats[...,i], y[...,i], 0.5)) 
            losses.append(self.loss_weight[i]*loss_func(y_hat_us[...,i], y[...,i], 0.9)) 

            # differntial loss 
            y_diff_l = torch.diff(y_hat_ls[...,i])
            y_diff_u = torch.diff(y_hat_us[...,i])
            y_diff_m = torch.diff(y_hats[...,i])
            y_diff = torch.diff(y[...,i])
            losses.append(self.loss_weight[i]*loss_func(y_diff_l, y_diff, 0.1))
            losses.append(self.loss_weight[i]*loss_func(y_diff_m, y_diff, 0.5))
            losses.append(self.loss_weight[i]*loss_func(y_diff_u, y_diff, 0.9))


        loss = torch.stack(losses).sum()
        self.log('val_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hats, (y_hat_ls,y_hat_us) = self(x)
        return y_hats, y
