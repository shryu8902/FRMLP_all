#%%
from json import decoder
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

# %%

# %%
class AR_TRANSFMR(LTS_model):
    def __init__(self, c_in, d_model, d_emb, n_out, out_len = 333, n_layers = 5, n_head = 8, dr_rates = 0.2, loss_weight = [1,1,1,1], *args, **kwargs ):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in
        self.out_len = out_len #maximum output length
        self.n_layers = n_layers
        self.dr_rates = dr_rates
        self.n_head = n_head
        self.d_model = d_model
        self.d_emb = d_emb
        self.n_out = n_out
        self.input_embedding = SeperateDataEmbedding_V2(c_in, d_model, d_emb = d_emb)
        self.d_lat = self.input_embedding.d_lat
        self.loss_weight = loss_weight

        assert self.d_lat%8 ==0, 'd_lat cannot be divided by num_heads'

        self.main_layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.main_layers.append(nn.TransformerDecoderLayer(d_model = self.d_lat, nhead = self.n_head, batch_first = True,dropout=self.dr_rates))
        self.arrange1 = Rearrange('b d n -> b n d')
        self.mlp_head = nn.Linear(self.d_lat, self.n_out)
        self.mlp_head_u = nn.Linear(self.d_lat, self.n_out)
        self.mlp_head_l = nn.Linear(self.d_lat, self.n_out)

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        #x  # [b, n, d]
        # print(x.shape[1])
        lookahead = self._generate_square_subsequent_mask(x.shape[1])
        x = self.input_embedding(x)
        lookahead = lookahead.to(x.device)
        for decoder_layer in self.main_layers:
            x = decoder_layer(tgt = x, memory = x, tgt_mask = lookahead, memory_mask = lookahead)
        y_preds = self.mlp_head(x)
        y_pred_us = self.mlp_head_u(x)
        y_pred_ls = self.mlp_head_l(x)

        return y_preds, (y_pred_ls,y_pred_us)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hats, (y_hat_ls,y_hat_us) = self(x)

        losses = []
        loss_func = PinballLoss()
        for i in range(self.n_out):
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

class TRANSFORMER(LTS_model):
    def __init__(self, c_in, d_model, n_layers = 3, window = 168, out_len = 24, latent_window= 7,  hdim = 128, n_head=8, dr_rates = 0.2, use_time_feature=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.window = window
        self.hdim = hdim
        self.n_layer = n_layers
        self.dr_rates = dr_rates
        self.use_time_feature = use_time_feature
        self.n_head = n_head
        self.input_embedding = DataEmbedding(c_in, d_model, use_time_feature=self.use_time_feature)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=self.n_head, batch_first=True, dim_feedforward=hdim, dropout = self.dr_rates) # batch, sequence, feature
        self.main_layers = nn.ModuleList([])
        self.main_layers.append(nn.TransformerEncoder(enc_layer, num_layers = n_layers))
        self.main_layers.append(Rearrange('b n d -> b d n'))
        self.main_layers.append(nn.AdaptiveAvgPool1d(latent_window))
        self.mlp_head = nn.Linear(int(d_model*latent_window), out_len)

    def forward(self, x):
        x = self.input_embedding(x)
        for main_layer in self.main_layers:
            x = main_layer(x)
        x = x.flatten(start_dim=1)
        out = self.mlp_head(x)
        y_pred=out
        return y_pred

