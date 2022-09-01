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


# %%
class MLP_MIXER(LTS_model):
    def __init__(self, c_in, d_model, out_len = 333, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.out_len = out_len
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers=n_layers
        self.dr_rates = dr_rates
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = DataEmbedding(c_in, d_model)
        self.arrange1 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)

        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.out_len, ch_dim=self.d_model, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        
        self.mlp_head = nn.Linear(int(d_model), 1)

    def forward(self, x):
        x = x.unsqueeze(-1).repeat(1,1,self.out_len) # X :  b x d x n
        x = self.arrange1(x) #X : b x n x d
        x = self.input_embedding(x)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = self.mlp_head(x)
        x = x.flatten(start_dim = 1)
        y_pred=x
        return y_pred
#%%
class SEP_MLP_MIXER(LTS_model):
    def __init__(self, c_in, d_model, out_len = 333, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.out_len = out_len
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers = n_layers
        self.dr_rates = dr_rates
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = SeperateDataEmbedding(c_in, d_model)
        self.arrange1 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)

        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.out_len, ch_dim=2*self.d_model, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        
        self.mlp_head = nn.Linear(int(2*d_model), 1)

    def forward(self, x):
        x = x.unsqueeze(-1).repeat(1,1,self.out_len) # X :  b x d x n
        x = self.arrange1(x) #X : b x n x d
        x = self.input_embedding(x)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = self.mlp_head(x)
        x = x.flatten(start_dim = 1)
        y_pred=x
        return y_pred
#%%
#%%
class SEP_Multitask_MLP_MIXER(LTS_model):
    def __init__(self, c_in, d_model, out_len = 333, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.out_len = out_len
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers = n_layers
        self.dr_rates = dr_rates
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = SeperateDataEmbedding(c_in, d_model)
        self.arrange1 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)

        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.out_len, ch_dim=2*self.d_model, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        
        self.mlp_head = nn.Linear(int(2*d_model), 1)

    def forward(self, x):
        x = x.unsqueeze(-1).repeat(1,1,self.out_len) # X :  b x d x n
        x = self.arrange1(x) #X : b x n x d
        x = self.input_embedding(x)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = self.mlp_head(x)
        x = x.flatten(start_dim = 1)
        y_pred=x
        return y_pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        recon_loss = F.mse_loss(y_hat, y)
        loc_true_peak = torch.argmax(y, axis=1).float()
        loc_hat_peak=torch.argmax(y_hat, axis=1).float()
        location_loss = F.l1_loss(loc_hat_peak, loc_true_peak)
        y_diff = torch.diff(y)
        y_hat_diff = torch.diff(y_hat)
        diff_loss = F.mse_loss(y_hat_diff, y_diff)
        loss =recon_loss+location_loss+diff_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        recon_loss = F.mse_loss(y_hat, y)
        loc_true_peak = torch.argmax(y, axis=1).float()
        loc_hat_peak=torch.argmax(y_hat, axis=1).float()
        location_loss = F.l1_loss(loc_hat_peak, loc_true_peak)
        y_diff = torch.diff(y)
        y_hat_diff = torch.diff(y_hat)
        diff_loss = F.mse_loss(y_hat_diff, y_diff)
        loss =recon_loss+location_loss+diff_loss
        self.log('val_loss', loss)
        return loss
#%%
class SEP_MT_QMLP_MIXER(LTS_model):
    def __init__(self, c_in, d_model, out_len = 333, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.out_len = out_len
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers = n_layers
        self.dr_rates = dr_rates
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = SeperateDataEmbedding(c_in, d_model)
        self.arrange1 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)

        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.out_len, ch_dim=2*self.d_model, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        
        self.mlp_head = nn.Linear(int(2*d_model), 3)
        # self.mlp_head_u = nn.Linear(int(2*d_model), 1)
        # self.mlp_head_l = nn.Linear(int(2*d_model), 1)

    def forward(self, x):
        x = x.unsqueeze(-1).repeat(1,1,self.out_len) # X :  b x d x n
        x = self.arrange1(x) #X : b x n x d
        x = self.input_embedding(x)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = self.mlp_head(x)
        y_pred = x[...,1]
        y_pred_l = x[...,0]
        y_pred_u = x[...,2]
        return y_pred, (y_pred_l,y_pred_u)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, (y_hat_l,y_hat_u) = self(x)
        loss_func = PinballLoss()
        loss_l = loss_func(y_hat_l,y, 0.1)
        loss_m = loss_func(y_hat,y,0.5)
        loss_u = loss_func(y_hat_u,y,0.9)
        loss = loss_l+loss_m+loss_u

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, (y_hat_l,y_hat_u) = self(x)
        loss_func = PinballLoss()
        loss_l = loss_func(y_hat_l,y, 0.1)
        loss_m = loss_func(y_hat,y,0.5)
        loss_u = loss_func(y_hat_u,y,0.9)
        loss = loss_l+loss_m+loss_u
        self.log('val_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat, (y_hat_l,y_hat_u) = self(x)
        return y_hat, y

#%%
class SEP_MT_QMLP_MIXER_V2(LTS_model):
    def __init__(self, c_in, d_model, out_len = 333, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.out_len = out_len
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers = n_layers
        self.dr_rates = dr_rates
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = SeperateDataEmbedding(c_in, d_model)
        self.arrange1 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)

        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.out_len, ch_dim=2*self.d_model, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        
        self.mlp_head = nn.Linear(int(2*d_model), 3)
        self.mlp_head_u = nn.Linear(int(2*d_model), 1)
        self.mlp_head_l = nn.Linear(int(2*d_model), 1)

    def forward(self, x):
        x = x.unsqueeze(-1).repeat(1,1,self.out_len) # X :  b x d x n
        x = self.arrange1(x) #X : b x n x d
        x = self.input_embedding(x)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = self.mlp_head(x)
        y_pred = x[...,1]
        y_pred_l = x[...,0]
        y_pred_u = x[...,2]
        return y_pred, (y_pred_l,y_pred_u)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, (y_hat_l,y_hat_u) = self(x)
        loss_func = PinballLoss()
        loss_l = loss_func(y_hat_l,y, 0.1)
        loss_m = loss_func(y_hat,y,0.5)
        loss_u = loss_func(y_hat_u,y,0.9)
        recon_loss = loss_l+loss_m+loss_u

        y_diff_l = torch.diff(y_hat_l)
        y_diff_u = torch.diff(y_hat_u)
        y_diff_m = torch.diff(y_hat)
        y_diff = torch.diff(y)
        diff_loss_l = loss_func(y_diff_l, y_diff, 0.1)
        diff_loss_m = loss_func(y_diff_m, y_diff, 0.5)
        diff_loss_u = loss_func(y_diff_u, y_diff, 0.9)
        diff_loss = diff_loss_l+diff_loss_m+diff_loss_u

        loc_true_peak = torch.argmax(y,axis=1).float()
        loc_peak_l = torch.argmax(y_hat_l, axis=1).float()
        loc_peak_m = torch.argmax(y_hat, axis=1).float()
        loc_peak_u = torch.argmax(y_hat_u, axis=1).float()
        peak_loss_l = loss_func(loc_peak_l, loc_true_peak, 0.1)
        peak_loss_m = loss_func(loc_peak_m, loc_true_peak, 0.5)
        peak_loss_u = loss_func(loc_peak_u, loc_true_peak, 0.9)        
        peak_loss = peak_loss_l + peak_loss_m + peak_loss_u

        loss =recon_loss+diff_loss+peak_loss

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, (y_hat_l,y_hat_u) = self(x)
        loss_func = PinballLoss()
        loss_l = loss_func(y_hat_l,y, 0.1)
        loss_m = loss_func(y_hat,y,0.5)
        loss_u = loss_func(y_hat_u,y,0.9)
        recon_loss = loss_l+loss_m+loss_u

        y_diff_l = torch.diff(y_hat_l)
        y_diff_u = torch.diff(y_hat_u)
        y_diff_m = torch.diff(y_hat)
        y_diff = torch.diff(y)
        diff_loss_l = loss_func(y_diff_l, y_diff, 0.1)
        diff_loss_m = loss_func(y_diff_m, y_diff, 0.5)
        diff_loss_u = loss_func(y_diff_u, y_diff, 0.9)
        diff_loss = diff_loss_l+diff_loss_m+diff_loss_u

        loc_true_peak = torch.argmax(y,axis=1).float()
        loc_peak_l = torch.argmax(y_hat_l, axis=1).float()
        loc_peak_m = torch.argmax(y_hat, axis=1).float()
        loc_peak_u = torch.argmax(y_hat_u, axis=1).float()
        peak_loss_l = loss_func(loc_peak_l, loc_true_peak, 0.1)
        peak_loss_m = loss_func(loc_peak_m, loc_true_peak, 0.5)
        peak_loss_u = loss_func(loc_peak_u, loc_true_peak, 0.9)        
        peak_loss = peak_loss_l + peak_loss_m + peak_loss_u

        loss =recon_loss+diff_loss+peak_loss
        self.log('val_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat, (y_hat_l,y_hat_u) = self(x)
        return y_hat, y

#%%
class SEP_MTMV_QMLP_MIXER(LTS_model):
    '''
    Separte positional encoding,
    Multitask loss
    Multivariate outputs
    quantiles per variable
    '''
    def __init__(self, c_in, d_model, out_len = 333, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.out_len = out_len
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers = n_layers
        self.dr_rates = dr_rates
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = SeperateDataEmbedding(c_in, d_model)
        self.arrange1 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)

        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.out_len, ch_dim=2*self.d_model, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        
        self.mlp_head = nn.Linear(int(2*d_model), 4)
        self.mlp_head_u = nn.Linear(int(2*d_model), 4)
        self.mlp_head_l = nn.Linear(int(2*d_model), 4)

    def forward(self, x):
        x = x.unsqueeze(-1).repeat(1,1,self.out_len) # X :  b x d x n
        x = self.arrange1(x) #X : b x n x d
        x = self.input_embedding(x)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        y_preds =self.mlp_head(x)
        y_pred_us = self.mlp_head_u(x)
        y_pred_ls = self.mlp_head_l(x)
        return y_preds, (y_pred_ls,y_pred_us)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hats, (y_hat_ls,y_hat_us) = self(x)

        losses = []
        loss_func = PinballLoss()
        for i in range(4):
            # recon pinball loss
            losses.append(loss_func(y_hat_ls[...,i], y[...,i], 0.1)) 
            losses.append(loss_func(y_hats[...,i], y[...,i], 0.5)) 
            losses.append(loss_func(y_hat_us[...,i], y[...,i], 0.9)) 

            # differntial loss 
            y_diff_l = torch.diff(y_hat_ls[...,i])
            y_diff_u = torch.diff(y_hat_us[...,i])
            y_diff_m = torch.diff(y_hats[...,i])
            y_diff = torch.diff(y[...,i])
            losses.append(loss_func(y_diff_l, y_diff, 0.1))
            losses.append(loss_func(y_diff_m, y_diff, 0.5))
            losses.append(loss_func(y_diff_u, y_diff, 0.9))

            # diff of diff loss
            y_diff_diff_l = torch.diff(y_diff_l)
            y_diff_diff_u = torch.diff(y_diff_u)
            y_diff_diff_m = torch.diff(y_diff_m)
            y_diff_diff = torch.diff(y_diff)
            losses.append(loss_func(y_diff_diff_l, y_diff_diff, 0.1))
            losses.append(loss_func(y_diff_diff_m, y_diff_diff, 0.5))
            losses.append(loss_func(y_diff_diff_u, y_diff_diff, 0.9))

        loss = torch.stack(losses).sum()

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hats, (y_hat_ls,y_hat_us) = self(x)

        losses = []
        loss_func = PinballLoss()
        for i in range(4):
            # recon pinball loss
            losses.append(loss_func(y_hat_ls[...,i], y[...,i], 0.1)) 
            losses.append(loss_func(y_hats[...,i], y[...,i], 0.5)) 
            losses.append(loss_func(y_hat_us[...,i], y[...,i], 0.9)) 

            # differntial loss 
            y_diff_l = torch.diff(y_hat_ls[...,i])
            y_diff_u = torch.diff(y_hat_us[...,i])
            y_diff_m = torch.diff(y_hats[...,i])
            y_diff = torch.diff(y[...,i])
            losses.append(loss_func(y_diff_l, y_diff, 0.1))
            losses.append(loss_func(y_diff_m, y_diff, 0.5))
            losses.append(loss_func(y_diff_u, y_diff, 0.9))

            # diff of diff loss
            y_diff_diff_l = torch.diff(y_diff_l)
            y_diff_diff_u = torch.diff(y_diff_u)
            y_diff_diff_m = torch.diff(y_diff_m)
            y_diff_diff = torch.diff(y_diff)
            losses.append(loss_func(y_diff_diff_l, y_diff_diff, 0.1))
            losses.append(loss_func(y_diff_diff_m, y_diff_diff, 0.5))
            losses.append(loss_func(y_diff_diff_u, y_diff_diff, 0.9))

        loss = torch.stack(losses).sum()
        self.log('val_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hats, (y_hat_ls,y_hat_us) = self(x)
        return y_hats, y
#%%
class SEP_MTMV_QMLP_MIXER_V2(LTS_model):
    '''
    Separte positional encoding,
    Multitask loss
    Multivariate outputs
    quantiles per variable
    '''
    def __init__(self, c_in, d_model, out_len = 333, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.out_len = out_len
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers = n_layers
        self.dr_rates = dr_rates
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = SeperateDataEmbedding(c_in, d_model)
        self.arrange1 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)

        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.out_len, ch_dim=2*self.d_model, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        
        self.mlp_head = nn.Linear(int(2*d_model), 4)
        self.mlp_head_u = nn.Linear(int(2*d_model), 4)
        self.mlp_head_l = nn.Linear(int(2*d_model), 4)

    def forward(self, x):
        x = x.unsqueeze(-1).repeat(1,1,self.out_len) # X :  b x d x n
        x = self.arrange1(x) #X : b x n x d
        x = self.input_embedding(x)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        y_preds =self.mlp_head(x)
        y_pred_us = self.mlp_head_u(x)
        y_pred_ls = self.mlp_head_l(x)
        return y_preds, (y_pred_ls,y_pred_us)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hats, (y_hat_ls,y_hat_us) = self(x)

        losses = []
        loss_func = PinballLoss()
        for i in range(4):
            # recon pinball loss
            losses.append(loss_func(y_hat_ls[...,i], y[...,i], 0.1)) 
            losses.append(loss_func(y_hats[...,i], y[...,i], 0.5)) 
            losses.append(loss_func(y_hat_us[...,i], y[...,i], 0.9)) 

            # differntial loss 
            # y_diff_l = torch.diff(y_hat_ls[...,i])
            # y_diff_u = torch.diff(y_hat_us[...,i])
            # y_diff_m = torch.diff(y_hats[...,i])
            # y_diff = torch.diff(y[...,i])
            # losses.append(loss_func(y_diff_l, y_diff, 0.1))
            # losses.append(loss_func(y_diff_m, y_diff, 0.5))
            # losses.append(loss_func(y_diff_u, y_diff, 0.9))

            # # diff of diff loss
            # y_diff_diff_l = torch.diff(y_diff_l)
            # y_diff_diff_u = torch.diff(y_diff_u)
            # y_diff_diff_m = torch.diff(y_diff_m)
            # y_diff_diff = torch.diff(y_diff)
            # losses.append(loss_func(y_diff_diff_l, y_diff_diff, 0.1))
            # losses.append(loss_func(y_diff_diff_m, y_diff_diff, 0.5))
            # losses.append(loss_func(y_diff_diff_u, y_diff_diff, 0.9))

        loss = torch.stack(losses).sum()

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hats, (y_hat_ls,y_hat_us) = self(x)

        losses = []
        loss_func = PinballLoss()
        for i in range(4):
            # recon pinball loss
            losses.append(loss_func(y_hat_ls[...,i], y[...,i], 0.1)) 
            losses.append(loss_func(y_hats[...,i], y[...,i], 0.5)) 
            losses.append(loss_func(y_hat_us[...,i], y[...,i], 0.9)) 

        loss = torch.stack(losses).sum()
        self.log('val_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hats, (y_hat_ls,y_hat_us) = self(x)
        return y_hats, y
#%%
class SEP_MTMV_QMLP_MIXER_V3(LTS_model):
    '''
    Separte positional encoding,
    Multitask loss
    Multivariate outputs
    quantiles per variable
    '''
    def __init__(self, c_in, d_model, out_len = 333, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.out_len = out_len
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers = n_layers
        self.dr_rates = dr_rates
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = SeperateDataEmbedding(c_in, d_model)
        self.arrange1 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)

        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.out_len, ch_dim=2*self.d_model, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        
        self.mlp_head = nn.Linear(int(2*d_model), 4)
        self.mlp_head_u = nn.Linear(int(2*d_model), 4)
        self.mlp_head_l = nn.Linear(int(2*d_model), 4)

    def forward(self, x):
        x = x.unsqueeze(-1).repeat(1,1,self.out_len) # X :  b x d x n
        x = self.arrange1(x) #X : b x n x d
        x = self.input_embedding(x)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        y_preds =self.mlp_head(x)
        y_pred_us = self.mlp_head_u(x)
        y_pred_ls = self.mlp_head_l(x)
        return y_preds, (y_pred_ls,y_pred_us)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hats, (y_hat_ls,y_hat_us) = self(x)

        losses = []
        loss_func = PinballLoss()
        for i in range(4):
            # recon pinball loss
            losses.append(loss_func(y_hat_ls[...,i], y[...,i], 0.1)) 
            losses.append(loss_func(y_hats[...,i], y[...,i], 0.5)) 
            losses.append(loss_func(y_hat_us[...,i], y[...,i], 0.9)) 

            # differntial loss 
            y_diff_l = torch.diff(y_hat_ls[...,i])
            y_diff_u = torch.diff(y_hat_us[...,i])
            y_diff_m = torch.diff(y_hats[...,i])
            y_diff = torch.diff(y[...,i])
            losses.append(loss_func(y_diff_l, y_diff, 0.1))
            losses.append(loss_func(y_diff_m, y_diff, 0.5))
            losses.append(loss_func(y_diff_u, y_diff, 0.9))

            # # diff of diff loss
            # y_diff_diff_l = torch.diff(y_diff_l)
            # y_diff_diff_u = torch.diff(y_diff_u)
            # y_diff_diff_m = torch.diff(y_diff_m)
            # y_diff_diff = torch.diff(y_diff)
            # losses.append(loss_func(y_diff_diff_l, y_diff_diff, 0.1))
            # losses.append(loss_func(y_diff_diff_m, y_diff_diff, 0.5))
            # losses.append(loss_func(y_diff_diff_u, y_diff_diff, 0.9))

        loss = torch.stack(losses).sum()

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hats, (y_hat_ls,y_hat_us) = self(x)

        losses = []
        loss_func = PinballLoss()
        for i in range(4):
            # recon pinball loss
            losses.append(loss_func(y_hat_ls[...,i], y[...,i], 0.1)) 
            losses.append(loss_func(y_hats[...,i], y[...,i], 0.5)) 
            losses.append(loss_func(y_hat_us[...,i], y[...,i], 0.9)) 

            # differntial loss 
            y_diff_l = torch.diff(y_hat_ls[...,i])
            y_diff_u = torch.diff(y_hat_us[...,i])
            y_diff_m = torch.diff(y_hats[...,i])
            y_diff = torch.diff(y[...,i])
            losses.append(loss_func(y_diff_l, y_diff, 0.1))
            losses.append(loss_func(y_diff_m, y_diff, 0.5))
            losses.append(loss_func(y_diff_u, y_diff, 0.9))


        loss = torch.stack(losses).sum()
        self.log('val_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hats, (y_hat_ls,y_hat_us) = self(x)
        return y_hats, y

#%%
class SEP_MTMV_QMLP_MIXER_V4(LTS_model):
    '''
    Separte positional encoding,
    Multitask loss
    Multivariate outputs
    quantiles per variable
    weighted loss function
    '''
    def __init__(self, c_in, d_model, out_len = 333, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, loss_weight = [1,1,1,1], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.out_len = out_len
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers = n_layers
        self.dr_rates = dr_rates
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = SeperateDataEmbedding(c_in, d_model)
        self.arrange1 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)

        self.loss_weight = loss_weight

        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.out_len, ch_dim=2*self.d_model, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        
        self.mlp_head = nn.Linear(int(2*d_model), 4)
        self.mlp_head_u = nn.Linear(int(2*d_model), 4)
        self.mlp_head_l = nn.Linear(int(2*d_model), 4)

    def forward(self, x):
        x = x.unsqueeze(-1).repeat(1,1,self.out_len) # X :  b x d x n
        x = self.arrange1(x) #X : b x n x d
        x = self.input_embedding(x)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        y_preds =self.mlp_head(x)
        y_pred_us = self.mlp_head_u(x)
        y_pred_ls = self.mlp_head_l(x)
        return y_preds, (y_pred_ls,y_pred_us)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hats, (y_hat_ls,y_hat_us) = self(x)

        losses = []
        loss_func = PinballLoss()
        
        for i in range(4):
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
        for i in range(4):
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
class Q_CONV_MIXER(LTS_model):
    '''
    Separte positional encoding,
    Multitask loss
    Multivariate outputs
    quantiles per variable
    weighted loss function
    '''
    def __init__(self, c_in, d_model, out_len = 333, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, loss_weight = [1,1,1,1], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.out_len = out_len
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers = n_layers
        self.dr_rates = dr_rates
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = SeperateDataEmbedding_V2(c_in, d_model,d_emb = d_model//2)
        self.arrange1 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)
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
        self.conv_head = nn.ConvTranspose1d(self.d_lat, 4, kernel_size = 3, stride = 3)
        self.conv_head_u = nn.ConvTranspose1d(self.d_lat, 4, kernel_size = 3, stride = 3)
        self.conv_head_l = nn.ConvTranspose1d(self.d_lat, 4, kernel_size = 3, stride = 3)

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
        
        for i in range(4):
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
        for i in range(4):
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
class Q_CONV_MIXER_AVGPOOL(LTS_model):
    '''
    Separte positional encoding,
    Multitask loss
    Multivariate outputs
    quantiles per variable
    weighted loss function
    '''
    def __init__(self, c_in, d_model, out_len = 333, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, loss_weight = [1,1,1,1], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.out_len = out_len
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers = n_layers
        self.dr_rates = dr_rates
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = SeperateDataEmbedding_V2(c_in, d_model,d_emb = d_model//2)
        self.arrange1 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)
        self.arrange2 = Rearrange('b n d -> b d n') # (batch, length, depth) -> (batch, depth, length)
        self.ma_layer = nn.AvgPool1d(3,stride = 1, padding = 1)
        self.d_lat = self.input_embedding.d_lat
        self.loss_weight = loss_weight

        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.out_len//3, ch_dim=self.d_lat, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        self.conv_head = nn.ConvTranspose1d(self.d_lat, 4, kernel_size = 3, stride = 3)
        self.conv_head_u = nn.ConvTranspose1d(self.d_lat, 4, kernel_size = 3, stride = 3)
        self.conv_head_l = nn.ConvTranspose1d(self.d_lat, 4, kernel_size = 3, stride = 3)

    def forward(self, x):
        x = x.unsqueeze(-1).repeat(1,1,self.out_len//3) # X :  b x d x n
        x = self.arrange1(x) #X : b x n x d
        x = self.input_embedding(x)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = self.arrange2(x) # x : b x d x n

        y_preds = self.arrange1(self.ma_layer(self.conv_head(x)))
        y_pred_us = self.arrange1(self.ma_layer(self.conv_head_u(x)))
        y_pred_ls = self.arrange1(self.ma_layer(self.conv_head_l(x)))
        return y_preds, (y_pred_ls,y_pred_us)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hats, (y_hat_ls,y_hat_us) = self(x)

        losses = []
        loss_func = PinballLoss()
        
        for i in range(4):
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
        for i in range(4):
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
class CASCADE_Q_CONV_MIXER_AVGPOOL(LTS_model):
    '''
    Separte positional encoding,
    Multitask loss
    Multivariate outputs
    quantiles per variable
    weighted loss function
    '''
    def __init__(self, c_in, d_model, out_len = 333, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, loss_weight = [1,1,1,1], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.out_len = out_len
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers = n_layers
        self.dr_rates = dr_rates
        self.mixer_layers = nn.ModuleList([])
        self.mixer2_layers = nn.ModuleList([])

        self.input_embedding = SeperateDataEmbedding_V2(c_in, d_model,d_emb = d_model//2)
        self.arrange1 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)
        self.arrange2 = Rearrange('b n d -> b d n') # (batch, length, depth) -> (batch, depth, length)
        self.ma_layer = nn.AvgPool1d(3,stride = 1, padding = 1)
        self.ma_layer2 = nn.AvgPool1d(3,stride = 3)
        self.d_lat = self.input_embedding.d_lat
        self.loss_weight = loss_weight

        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.out_len//3, ch_dim=self.d_lat, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        for _ in range(n_layers):
            self.mixer2_layers.append(
                MixerLayer(token_dim  = self.out_len//3, ch_dim=self.d_lat+2, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer2_layers.append(
            nn.Dropout(p=self.dr_rates)
            )

        self.conv_head = nn.ConvTranspose1d(self.d_lat, 2, kernel_size = 3, stride = 3) 
        self.conv_head_u = nn.ConvTranspose1d(self.d_lat, 2, kernel_size = 3, stride = 3) 
        self.conv_head_l = nn.ConvTranspose1d(self.d_lat, 2, kernel_size = 3, stride = 3) 

        self.conv2_head = nn.ConvTranspose1d(self.d_lat+2, 2, kernel_size = 3, stride = 3) 
        self.conv2_head_u = nn.ConvTranspose1d(self.d_lat+2, 2, kernel_size = 3, stride = 3) 
        self.conv2_head_l = nn.ConvTranspose1d(self.d_lat+2, 2, kernel_size = 3, stride = 3) 

        # self.mlp_head = nn.Linear(self.d_lat,2)
        

    def forward(self, x):

        x = x.unsqueeze(-1).repeat(1,1,self.out_len//3) # X :  b x d x n
        x = self.arrange1(x) #X : b x n x d
        # x = self.input_embedding(x)
        x_embed = self.input_embedding(x)
        x = x_embed
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = self.arrange2(x) # x : b x d x n

        pps_base = self.ma_layer(self.conv_head(x))
        pps_preds = self.arrange1(pps_base)
        pps_preds_u = self.arrange1(self.ma_layer(self.conv_head_u(x)))
        pps_preds_l = self.arrange1(self.ma_layer(self.conv_head_l(x)))

        pps_sampled = self.arrange1(self.ma_layer2(pps_base))
        cas_input = torch.concat([x_embed,pps_sampled],axis=-1)

        y = cas_input
        for mixer_layer in self.mixer2_layers:
            y = mixer_layer(y)
        y = self.arrange2(y) # y : b x d x n
        frel_base = self.ma_layer(self.conv2_head(y))
        frel_preds = self.arrange1(frel_base)
        frel_preds_u = self.arrange1(self.ma_layer(self.conv2_head_u(y)))
        frel_preds_l = self.arrange1(self.ma_layer(self.conv2_head_l(y)))

        y_preds = torch.concat([pps_preds,frel_preds],axis=-1)
        y_pred_us = torch.concat([pps_preds_u,frel_preds_u],axis=-1)
        y_pred_ls = torch.concat([pps_preds_l,frel_preds_l],axis=-1)
        return y_preds, (y_pred_ls,y_pred_us)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hats, (y_hat_ls,y_hat_us) = self(x)

        losses = []
        loss_func = PinballLoss()
        
        for i in range(4):
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
        for i in range(4):
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
class FREL_QMLP_MIXER(LTS_model):
    '''
    Separte positional encoding,
    Multitask loss
    Multivariate outputs
    quantiles per variable
    '''
    def __init__(self, c_in, d_model, out_len = 333, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.out_len = out_len
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers = n_layers
        self.dr_rates = dr_rates
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = SeperateDataEmbedding(c_in, d_model)
        self.arrange1 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)

        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.out_len, ch_dim=2*self.d_model, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        self.mlp_head = nn.Sequential(nn.Linear(int(2*d_model), 3),
                                        nn.ReLU())

    def forward(self, x):
        x = x.unsqueeze(-1).repeat(1,1,self.out_len) # X :  b x d x n
        x = self.arrange1(x) #X : b x n x d
        x = self.input_embedding(x)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        y_pred = x[...,1]
        y_pred_l = x[...,0]
        y_pred_u = x[...,2]
        return y_pred, (y_pred_l,y_pred_u)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, (y_hat_l,y_hat_u) = self(x)
        losses = []
        loss_func = PinballLoss()

        losses.append(loss_func(y_hat_l, y, 0.1)) 
        losses.append(loss_func(y_hat, y, 0.5)) 
        losses.append(loss_func(y_hat_u, y, 0.9)) 
        loss = torch.stack(losses).sum()

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hats, (y_hat_ls,y_hat_us) = self(x)

        y_hat, (y_hat_l,y_hat_u) = self(x)
        losses = []
        loss_func = PinballLoss()

        losses.append(loss_func(y_hat_l, y, 0.1)) 
        losses.append(loss_func(y_hat, y, 0.5)) 
        losses.append(loss_func(y_hat_u, y, 0.9)) 

        loss = torch.stack(losses).sum()
        self.log('val_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat, (y_hat_l,y_hat_u) = self(x)
        return y_hat, y
#%%
class TCMLP_MIXER(LTS_model):
    def __init__(self, c_in, d_model, window = 168, out_len = 24, latent_window= 7, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, use_time_feature=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.window = window
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers=n_layers
        self.dr_rates = dr_rates
        self.use_time_feature = use_time_feature
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = DataEmbedding(c_in, d_model, use_time_feature=self.use_time_feature)
        self.arrange1 = Rearrange('b n d -> b d n') # (batch, length, depth) -> (batch, depth, length)
        self.arrange2 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)

        self.TC_layers = nn.ModuleList([])
        kernel_size = 2
        dilation_list = [1,2]
        self.TC_layers.append(TemporalBlock(n_inputs = d_model, n_outputs = 2*d_model, kernel_size = kernel_size,
                                            stride=1, dilation=dilation_list[0], padding = (kernel_size-1)*dilation_list[0],
                                            dropout = dr_rates))
        self.TC_layers.append(TemporalBlock(n_inputs = 2*d_model, n_outputs = 2*d_model, kernel_size = kernel_size,
                                            stride=1, dilation=dilation_list[1], padding = (kernel_size-1)*dilation_list[1],
                                            dropout = dr_rates))

        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.window//4, ch_dim=2*self.d_model, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        self.mixer_layers.append(
                Rearrange('b n d -> b d n')
            )
        self.mixer_layers.append(
            nn.AdaptiveAvgPool1d(latent_window)
        )
        self.mlp_head = nn.Linear(int(2*d_model*latent_window), out_len)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.arrange1(x)
        for tc_layer in self.TC_layers:
            x = tc_layer(x)
        x = self.arrange2(x)
        x = x[:,3::4,:]
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = x.flatten(start_dim=1)
        out = self.mlp_head(x)
        y_pred=out
        return y_pred
        
class DTCMLP_MIXER(LTS_model):
    def __init__(self, c_in, d_model, window = 168, out_len = 24, latent_window= 7, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, use_time_feature=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.window = window
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers=n_layers
        self.dr_rates = dr_rates
        self.use_time_feature = use_time_feature
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = DataEmbedding(c_in, d_model, use_time_feature=self.use_time_feature)
        self.arrange1 = Rearrange('b n d -> b d n') # (batch, length, depth) -> (batch, depth, length)
        self.arrange2 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)

        self.inc_TC_layers = nn.ModuleList([])
        self.dec_TC_layers = nn.ModuleList([])

        kernel_size = 2
        inc_dilation = [1,2]
        dec_dilation = [4,2]
        self.inc_TC_layers.append(TemporalBlock(n_inputs = d_model, n_outputs = d_model, kernel_size = kernel_size,
                                                stride = 1, dilation = inc_dilation[0], padding = (kernel_size-1)*inc_dilation[0],
                                                dropout = dr_rates))
        self.inc_TC_layers.append(TemporalBlock(n_inputs = 2*d_model, n_outputs = d_model, kernel_size = kernel_size,
                                                stride = 1, dilation = inc_dilation[1], padding = (kernel_size-1)*inc_dilation[1],
                                                dropout = dr_rates))
        self.dec_TC_layers.append(TemporalBlock(n_inputs = d_model, n_outputs = d_model, kernel_size = kernel_size,
                                                stride = 1, dilation = dec_dilation[0], padding = (kernel_size-1)*dec_dilation[0],
                                                dropout = dr_rates))
        self.dec_TC_layers.append(TemporalBlock(n_inputs = 2*d_model, n_outputs = d_model, kernel_size = kernel_size,
                                                stride = 1, dilation = dec_dilation[1], padding = (kernel_size-1)*dec_dilation[1],
                                                dropout = dr_rates))
        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.window//4, ch_dim=2*self.d_model, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        self.mixer_layers.append(
                Rearrange('b n d -> b d n')
            )
        self.mixer_layers.append(
            nn.AdaptiveAvgPool1d(latent_window)
        )
        self.mlp_head = nn.Linear(int(2*d_model*latent_window), out_len)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.arrange1(x)
        for i in range(2):
            x_dense = self.inc_TC_layers[i](x)
            x_sparse = self.dec_TC_layers[i](x)
            x = torch.cat((x_dense,x_sparse),axis=1)
        x = self.arrange2(x)
        x = x[:,3::4,:]
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = x.flatten(start_dim=1)
        out = self.mlp_head(x)
        y_pred=out
        return y_pred
#%%
class CMLP_MIXER(LTS_model):
    def __init__(self, c_in, d_model, window = 168, out_len = 24, latent_window= 7, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, use_time_feature=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.window = window
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers=n_layers
        self.dr_rates = dr_rates
        self.use_time_feature = use_time_feature
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = DataEmbedding(c_in, d_model, use_time_feature=self.use_time_feature)
        self.arrange1 = Rearrange('b n d -> b d n') # (batch, length, depth) -> (batch, depth, length)
        self.arrange2 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)
        self.TC_layers = nn.ModuleList([])
        self.TC_layers.append(nn.Conv1d(self.d_model, 2*self.d_model, kernel_size = 2, stride = 2, padding = 'valid'))
        self.TC_layers.append(nn.LeakyReLU())
        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.window//2, ch_dim=2*self.d_model, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        self.mixer_layers.append(
                Rearrange('b n d -> b d n')
            )
        self.mixer_layers.append(
            nn.AdaptiveAvgPool1d(latent_window)
        )
        self.mlp_head = nn.Linear(int(2*d_model*latent_window), out_len)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.arrange1(x)
        for tc_layer in self.TC_layers:
            x = tc_layer(x)
        x = self.arrange2(x)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = x.flatten(start_dim=1)
        out = self.mlp_head(x)
        y_pred=out
        return y_pred
#%%
class CMLP_MIXER_V2(LTS_model):
    def __init__(self, c_in, d_model, window = 168, out_len = 24, latent_window= 7, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, use_time_feature=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.window = window
        self.hdim = token_hdim
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers=n_layers
        self.dr_rates = dr_rates
        self.use_time_feature = use_time_feature
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = DataEmbedding(c_in, d_model, use_time_feature=self.use_time_feature)

        self.arrange1 = Rearrange('b n d -> b d n') # (batch, length, depth) -> (batch, depth, length)
        self.arrange2 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)
        self.conv_layers = nn.ModuleList([])
        for i in range(self.n_layers):
            if i==0:
                self.conv_layers.append(
                    nn.Conv1d(self.d_model, 2*self.d_model, kernel_size = 3, stride = 1, padding = 'same')
                )            
            else:
                self.conv_layers.append(
                    nn.Conv1d(2*self.d_model, 2*self.d_model, kernel_size = 3, stride = 1, padding = 'same')
                )            
            self.conv_layers.append(nn.MaxPool1d(2))
            self.conv_layers.append(nn.LeakyReLU())
        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.window//(2*self.n_layers), ch_dim=2*self.d_model, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        self.mixer_layers.append(
                Rearrange('b n d -> b d n')
            )
        self.mixer_layers.append(
            nn.AdaptiveAvgPool1d(latent_window)
        )
        self.mlp_head = nn.Linear(int(2*self.d_model*latent_window), out_len)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.arrange1(x)
        for layer in self.conv_layers:
            x = layer(x)
        x = self.arrange2(x)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = x.flatten(start_dim=1)
        out = self.mlp_head(x)
        y_pred=out
        return y_pred
#%%
class CMLP_MIXER_V3(LTS_model):
    def __init__(self, c_in, d_model, window = 168, out_len = 24, latent_window= 7, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, use_time_feature=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.window = window
        self.hdim = token_hdim
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers=n_layers
        self.dr_rates = dr_rates
        self.use_time_feature = use_time_feature
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = DataEmbedding(c_in, d_model, use_time_feature=self.use_time_feature)

        self.arrange1 = Rearrange('b n d -> b d n') # (batch, length, depth) -> (batch, depth, length)
        self.arrange2 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)
        self.conv_layers = nn.ModuleList([])
        self.conv_layers.append(
            nn.Conv1d(self.d_model, 2*self.d_model, kernel_size = 3, stride = 1, padding = 'same')
        )            
        self.conv_layers.append(nn.MaxPool1d(2))
        # self.conv_layers.append(nn.LeakyReLU())
        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.window//2, ch_dim=2*self.d_model, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        self.mixer_layers.append(
                Rearrange('b n d -> b d n')
            )
        self.mixer_layers.append(
            nn.AdaptiveAvgPool1d(latent_window)
        )
        self.mlp_head = nn.Linear(int(2*self.d_model*latent_window), out_len)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.arrange1(x)
        for layer in self.conv_layers:
            x = layer(x)
        x = self.arrange2(x)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = x.flatten(start_dim=1)
        out = self.mlp_head(x)
        y_pred=out
        return y_pred
#%%
class Q_MLP_MIXER(LTS_model):
    def __init__(self, c_in, d_model, window = 168,  out_len = 24, latent_window= 7, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, num_stoch = 10, use_time_feature=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.window = window
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers = n_layers
        self.dr_rates = dr_rates
        self.num_stoch = num_stoch
        self.use_time_feature = use_time_feature
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = DataEmbedding(c_in, d_model, use_time_feature=self.use_time_feature)

        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.window, ch_dim=self.d_model+1, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        self.mixer_layers.append(
                Rearrange('b n d -> b d n')
            )
        self.mixer_layers.append(
            nn.AdaptiveAvgPool1d(latent_window)
        )
        self.mlp_head = nn.Linear(int((d_model+1)*latent_window), out_len)

    def forward(self, x, q):
        x = self.input_embedding(x)
        q_input = q*torch.ones([x.shape[0],x.shape[1],1])
        x = torch.concat([x, q_input.to(x.device)], axis=2)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = x.flatten(start_dim=1)
        out = self.mlp_head(x)
        y_pred=out
        return y_pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss_func = PinballLoss()
        qs = torch.rand(self.num_stoch)
        qs = torch.cat([qs,torch.tensor([0.5])])
        loss_list = []
        for q in qs:
            y_hat = self(x,q)                     
            loss_list.append(loss_func(y_hat,y,q))
        loss = torch.stack(loss_list).mean()
        self.log('train_loss', loss)
        return loss

    def validation_step(self,batch,batch_idx):
        x, y = batch
        loss_func = PinballLoss()
        qs = torch.tensor(np.arange(0.1,1,0.1))
        loss_list = []
        for q in qs:
            y_hat = self(x,q)                     
            loss_list.append(loss_func(y_hat,y,q))
        loss = torch.stack(loss_list).mean()
        self.log('val_loss', loss)
        return loss

    def test_step(self,batch,batch_idx):
        x, y = batch
        loss_func = PinballLoss()
        qs = torch.tensor(np.arange(0.1,1,0.1))
        loss_list = []
        for q in qs:
            y_hat = self(x,q)                     
            loss_list.append(loss_func(y_hat,y,q))
        loss = torch.stack(loss_list).mean()
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x,0.5)
        return y, y_hat

#%%
class Gaussian_MIXER(LTS_model):
    def __init__(self, c_in, d_model, window = 168,  out_len = 24, latent_window= 7, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, num_stoch = 10, use_time_feature=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.window = window
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers = n_layers
        self.dr_rates = dr_rates
        self.num_stoch = num_stoch
        self.use_time_feature = use_time_feature
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = DataEmbedding(c_in, d_model, use_time_feature=self.use_time_feature)

        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.window, ch_dim=self.d_model, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        self.mixer_layers.append(
                Rearrange('b n d -> b d n')
            )
        self.mixer_layers.append(
            nn.AdaptiveAvgPool1d(latent_window)
        )
        self.mlp_head_mu = nn.Linear(int((d_model)*latent_window), out_len)
        self.mlp_head_var = nn.Sequential(nn.Linear(int((d_model)*latent_window), out_len),
                                    nn.ReLU())

    def forward(self, x):
        x = self.input_embedding(x)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = x.flatten(start_dim=1)
        mu = self.mlp_head_mu(x)
        var = self.mlp_head_var(x)
        return mu, var

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss_func = nn.GaussianNLLLoss()
        y_hat_mu, y_hat_var =self(x)
        loss = loss_func(input = y_hat_mu, target=y, var = y_hat_var)
        self.log('train_loss', loss)
        return loss

    def validation_step(self,batch,batch_idx):
        x, y = batch
        loss_func = nn.GaussianNLLLoss()
        y_hat_mu, y_hat_var =self(x)
        loss = loss_func(input = y_hat_mu, target=y, var = y_hat_var)
        self.log('val_loss', loss)
        return loss

    def test_step(self,batch,batch_idx):
        x, y = batch
        loss_func = nn.GaussianNLLLoss()
        y_hat_mu, y_hat_var =self(x)
        loss = loss_func(input = y_hat_mu, target=y, var = y_hat_var)
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat_mu, y_hat_var = self(x)
        return y, y_hat_mu

class FIXED_Q_MIXER(LTS_model):
    def __init__(self, c_in, d_model, target_q = [0.3, 0.5, 0.7], window = 168,  out_len = 24, latent_window= 7, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, num_stoch = 10, use_time_feature=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.window = window
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers = n_layers
        self.dr_rates = dr_rates
        self.num_stoch = num_stoch
        self.use_time_feature = use_time_feature
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = DataEmbedding(c_in, d_model, use_time_feature=self.use_time_feature)
        self.target_q=target_q

        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.window, ch_dim=self.d_model, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        self.mixer_layers.append(
                Rearrange('b n d -> b d n')
            )
        self.mixer_layers.append(
            nn.AdaptiveAvgPool1d(latent_window)
        )
        self.mlp_head_l = nn.Linear(int((d_model)*latent_window), out_len)
        self.mlp_head_m = nn.Linear(int((d_model)*latent_window), out_len)
        self.mlp_head_u = nn.Linear(int((d_model)*latent_window), out_len)

    def forward(self, x):
        x = self.input_embedding(x)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = x.flatten(start_dim=1)
        low = self.mlp_head_l(x)
        mid = self.mlp_head_m(x)
        up = self.mlp_head_u(x)
        return (low,mid,up)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss_func = PinballLoss()
        y_hat_low, y_hat_mid, y_hat_up = self(x)
        loss_l = loss_func(y_hat_low,y,self.target_q[0])
        loss_m = loss_func(y_hat_mid,y,self.target_q[1])
        loss_u = loss_func(y_hat_up,y,self.target_q[2])
        loss = loss_l+loss_m+loss_u
        self.log('train_loss', loss)
        return loss

    def validation_step(self,batch,batch_idx):
        x, y = batch
        loss_func = PinballLoss()
        y_hat_low, y_hat_mid, y_hat_up = self(x)
        loss_l = loss_func(y_hat_low,y,self.target_q[0])
        loss_m = loss_func(y_hat_mid,y,self.target_q[1])
        loss_u = loss_func(y_hat_up,y,self.target_q[2])
        loss = loss_l+loss_m+loss_u
        self.log('val_loss', loss)
        return loss

    def test_step(self,batch,batch_idx):
        x, y = batch
        loss_func = PinballLoss()
        y_hat_low, y_hat_mid, y_hat_up = self(x)
        loss_l = loss_func(y_hat_low,y,self.target_q[0])
        loss_m = loss_func(y_hat_mid,y,self.target_q[1])
        loss_u = loss_func(y_hat_up,y,self.target_q[2])
        loss = loss_l+loss_m+loss_u
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        l,m,u = self(x)
        return y, m
