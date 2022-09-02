#%%
import numpy as np
import pandas as pd
from torch.nn.modules.dropout import Dropout
from torch.nn.utils import weight_norm

#torch related
import torch
from torch import nn
import torch.nn.functional as F

#pytorch lightning
import pytorch_lightning as pl

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from Model.embed import DataEmbedding
from Model.common_model import LTS_model

# %%

class Chomp1D(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:,:,:-self.chomp_size].contiguous() # conv1d : N x C x L not N x L xC

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout = 0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride = stride, padding = padding, dilation =dilation))
        self.chomp1 = Chomp1D(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding = padding, dilation = dilation))
        self.chomp2 = Chomp1D(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0,0.01)
        self.conv2.weight.data.normal_(0,0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0,0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
#%%
class TemporalConvNet(LTS_model):
    def __init__(self, c_in, d_model, window= 168, out_len= 24, hdim = 128, dilation_list = [1,2,3,4,8,8,16] , kernel_size = 3, dr_rates = 0.2, use_time_feature = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_in = c_in
        self.use_time_feature = use_time_feature
        self.input_embedding = DataEmbedding(c_in, d_model, use_time_feature=self.use_time_feature)
        self.arrange1 = Rearrange('b n d -> b d n') # (batch, length, depth) -> (batch, depth, length)
        self.arrange2 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)
        self.tcn_stack = nn.ModuleList([])
        self.dilation_list = dilation_list

        for ii, dilation_size in enumerate(dilation_list):
            if ii ==0:
                self.tcn_stack.append(TemporalBlock(n_inputs = d_model, n_outputs= hdim,
                                                    kernel_size = kernel_size, stride = 1, 
                                                    dilation = dilation_size, padding = (kernel_size-1)*dilation_size,
                                                    dropout = dr_rates))
            else:
                self.tcn_stack.append(TemporalBlock(n_inputs = hdim, n_outputs= hdim,
                                                    kernel_size = kernel_size, stride = 1, 
                                                    dilation = dilation_size, padding = (kernel_size-1)*dilation_size,
                                                    dropout = dr_rates))
        self.mlp_head = nn.Linear(hdim, out_len)
        self.kernel_size = kernel_size
        self.len_receptive = self.receptive_field_length()
        # assert self.len_receptive > window,  'receptive field does not cover input window'

    def receptive_field_length(self):
        return 1+2*(self.kernel_size-1)*np.sum(self.dilation_list)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.arrange1(x)
        for temporal_conv_layer in self.tcn_stack:
            x = temporal_conv_layer(x)
        x = self.arrange2(x)
        y_pred = self.mlp_head(x[:,-1,:])
        return y_pred


class DualTemporalConvNet(LTS_model):
    '''
    DualTCN : in general dilation decreases as stacking more layer. in DualTCN, there exists two dilation factors. inc_dilation increase and dec_dilation decrease.
    Feature map from both dilation directions are concatenated thus the model leverage both recent time step and past time step.
    '''
    def __init__(self, c_in, d_model, window= 168, out_len = 24, hdim=128, inc_dilation = [1,2,3,4,8,8,16], dec_dilation = [24,8,6,4,3,2,1] , kernel_size = 3, dr_rates = 0.2, use_time_feature = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_in = c_in
        self.use_time_feature = use_time_feature
        self.input_embedding = DataEmbedding(c_in, d_model, use_time_feature=self.use_time_feature)
        self.arrange1 = Rearrange('b n d -> b d n') # (batch, length, depth) -> (batch, depth, length)
        self.arrange2 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)
        self.tcn_inc_stack = nn.ModuleList([])
        self.tcn_dec_stack = nn.ModuleList([])
        self.inc_dilation = inc_dilation
        self.dec_dilation = dec_dilation
        assert len(inc_dilation)==len(dec_dilation)
        self.n_stack = len(inc_dilation)

        for i in range(self.n_stack):
            if i ==0:
                self.tcn_inc_stack.append(TemporalBlock(n_inputs = d_model, n_outputs= hdim//2,
                                                    kernel_size = kernel_size, stride = 1, 
                                                    dilation = self.inc_dilation[i], padding = (kernel_size-1)*self.inc_dilation[i],
                                                    dropout = dr_rates))
                self.tcn_dec_stack.append(TemporalBlock(n_inputs = d_model, n_outputs= hdim//2,
                                                    kernel_size = kernel_size, stride = 1, 
                                                    dilation = self.dec_dilation[i], padding = (kernel_size-1)*self.dec_dilation[i],
                                                    dropout = dr_rates))
            else:
                self.tcn_inc_stack.append(TemporalBlock(n_inputs = hdim, n_outputs= hdim//2,
                                                    kernel_size = kernel_size, stride = 1, 
                                                    dilation = self.inc_dilation[i], padding = (kernel_size-1)*self.inc_dilation[i],
                                                    dropout = dr_rates))
                self.tcn_dec_stack.append(TemporalBlock(n_inputs = hdim, n_outputs= hdim//2,
                                                    kernel_size = kernel_size, stride = 1, 
                                                    dilation = self.dec_dilation[i], padding = (kernel_size-1)*self.dec_dilation[i],
                                                    dropout = dr_rates))

        self.mlp_head = nn.Linear(hdim, out_len)
        self.kernel_size = kernel_size
        self.len_receptive = self.receptive_field_length()
        # assert self.len_receptive['inc_recep'] > window,  'receptive field does not cover input window'
        # assert self.len_receptive['dec_recep'] > window,  'receptive field does not cover input window'

    def receptive_field_length(self):
        inc_recep = 1+2*(self.kernel_size-1)*np.sum(self.inc_dilation)
        dec_recep = 1+2*(self.kernel_size-1)*np.sum(self.dec_dilation)
        return {'inc_recep':inc_recep, 'dec_recep':dec_recep}

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.arrange1(x)
        for i in range(self.n_stack):
            x_dense = self.tcn_inc_stack[i](x)
            x_sparse = self.tcn_dec_stack[i](x)
            x = torch.cat((x_dense,x_sparse),axis=1)
        x = self.arrange2(x)
        y_pred = self.mlp_head(x[:,-1,:])
        return y_pred
#%%
class TemporalConvNet_V2(LTS_model):
    def __init__(self, c_in, d_model, window= 168, out_len= 24, latent_window=7, hdim = 128, dilation_list = [1,2,3,4,8,8,16] , kernel_size = 3, dr_rates = 0.2, use_time_feature = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_in = c_in
        self.use_time_feature = use_time_feature
        self.input_embedding = DataEmbedding(c_in, d_model, use_time_feature=self.use_time_feature)
        self.arrange1 = Rearrange('b n d -> b d n') # (batch, length, depth) -> (batch, depth, length)
        self.arrange2 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)
        self.tcn_stack = nn.ModuleList([])
        self.dilation_list = dilation_list

        for ii, dilation_size in enumerate(dilation_list):
            if ii ==0:
                self.tcn_stack.append(TemporalBlock(n_inputs = d_model, n_outputs= hdim,
                                                    kernel_size = kernel_size, stride = 1, 
                                                    dilation = dilation_size, padding = (kernel_size-1)*dilation_size,
                                                    dropout = dr_rates))
            else:
                self.tcn_stack.append(TemporalBlock(n_inputs = hdim, n_outputs= hdim,
                                                    kernel_size = kernel_size, stride = 1, 
                                                    dilation = dilation_size, padding = (kernel_size-1)*dilation_size,
                                                    dropout = dr_rates))
        self.mlp_head = nn.Linear(hdim*latent_window, out_len)
        self.kernel_size = kernel_size
        self.len_receptive = self.receptive_field_length()
        interval=window//latent_window
        self.lat_ind = [-1-i*interval for i in range(latent_window)]

        # assert self.len_receptive > window,  'receptive field does not cover input window'

    def receptive_field_length(self):
        return 1+2*(self.kernel_size-1)*np.sum(self.dilation_list)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.arrange1(x)
        for temporal_conv_layer in self.tcn_stack:
            x = temporal_conv_layer(x)
        x = self.arrange2(x)
        y_pred = self.mlp_head(x[:,self.lat_ind,:].flatten())
        return y_pred


class DualTemporalConvNet_V2(LTS_model):
    '''
    DualTCN : in general dilation decreases as stacking more layer. in DualTCN, there exists two dilation factors. inc_dilation increase and dec_dilation decrease.
    Feature map from both dilation directions are concatenated thus the model leverage both recent time step and past time step.
    '''
    def __init__(self, c_in, d_model, window= 168, out_len = 24, latent_window=7, hdim=128, inc_dilation = [1,2,3,4,8,8,16], dec_dilation = [24,8,6,4,3,2,1] , kernel_size = 3, dr_rates = 0.2, use_time_feature = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_in = c_in
        self.use_time_feature = use_time_feature
        self.input_embedding = DataEmbedding(c_in, d_model, use_time_feature=self.use_time_feature)
        self.arrange1 = Rearrange('b n d -> b d n') # (batch, length, depth) -> (batch, depth, length)
        self.arrange2 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)
        self.tcn_inc_stack = nn.ModuleList([])
        self.tcn_dec_stack = nn.ModuleList([])
        self.inc_dilation = inc_dilation
        self.dec_dilation = dec_dilation
        assert len(inc_dilation)==len(dec_dilation)
        self.n_stack = len(inc_dilation)
        interval=window//latent_window
        self.lat_ind = [-1-i*interval for i in range(latent_window)]

        for i in range(self.n_stack):
            if i ==0:
                self.tcn_inc_stack.append(TemporalBlock(n_inputs = d_model, n_outputs= hdim//2,
                                                    kernel_size = kernel_size, stride = 1, 
                                                    dilation = self.inc_dilation[i], padding = (kernel_size-1)*self.inc_dilation[i],
                                                    dropout = dr_rates))
                self.tcn_dec_stack.append(TemporalBlock(n_inputs = d_model, n_outputs= hdim//2,
                                                    kernel_size = kernel_size, stride = 1, 
                                                    dilation = self.dec_dilation[i], padding = (kernel_size-1)*self.dec_dilation[i],
                                                    dropout = dr_rates))
            else:
                self.tcn_inc_stack.append(TemporalBlock(n_inputs = hdim, n_outputs= hdim//2,
                                                    kernel_size = kernel_size, stride = 1, 
                                                    dilation = self.inc_dilation[i], padding = (kernel_size-1)*self.inc_dilation[i],
                                                    dropout = dr_rates))
                self.tcn_dec_stack.append(TemporalBlock(n_inputs = hdim, n_outputs= hdim//2,
                                                    kernel_size = kernel_size, stride = 1, 
                                                    dilation = self.dec_dilation[i], padding = (kernel_size-1)*self.dec_dilation[i],
                                                    dropout = dr_rates))

        self.mlp_head = nn.Linear(hdim*latent_window, out_len)
        self.kernel_size = kernel_size
        self.len_receptive = self.receptive_field_length()
        # assert self.len_receptive['inc_recep'] > window,  'receptive field does not cover input window'
        # assert self.len_receptive['dec_recep'] > window,  'receptive field does not cover input window'

    def receptive_field_length(self):
        inc_recep = 1+2*(self.kernel_size-1)*np.sum(self.inc_dilation)
        dec_recep = 1+2*(self.kernel_size-1)*np.sum(self.dec_dilation)
        return {'inc_recep':inc_recep, 'dec_recep':dec_recep}

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.arrange1(x)
        for i in range(self.n_stack):
            x_dense = self.tcn_inc_stack[i](x)
            x_sparse = self.tcn_dec_stack[i](x)
            x = torch.cat((x_dense,x_sparse),axis=1)
        x = self.arrange2(x)
        y_pred = self.mlp_head(x[:,self.lat_ind,:].flatten())
        # y_pred = self.mlp_head(x[:,-1,:])
        return y_pred
#%%
class TemporalConvNet_V3(LTS_model):
    def __init__(self, c_in, d_model, window= 168, out_len= 24, hdim = 128, n_layers = 3, dilation_list = [1,2,3,4,8,8,16] , kernel_size = 3, dr_rates = 0.2, use_time_feature = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_in = c_in
        self.use_time_feature = use_time_feature
        self.input_embedding = DataEmbedding(c_in, d_model, use_time_feature=self.use_time_feature)
        self.arrange1 = Rearrange('b n d -> b d n') # (batch, length, depth) -> (batch, depth, length)
        self.arrange2 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)
        self.tcn_stack = nn.ModuleList([])
        self.dilation_list = dilation_list

        for i in range(n_layers):
            for ii, dilation_size in enumerate(dilation_list):
                if ii ==0 and i==0:
                    self.tcn_stack.append(TemporalBlock(n_inputs = d_model, n_outputs= hdim,
                                                        kernel_size = kernel_size, stride = 1, 
                                                        dilation = dilation_size, padding = (kernel_size-1)*dilation_size,
                                                        dropout = dr_rates))
                else:
                    self.tcn_stack.append(TemporalBlock(n_inputs = hdim, n_outputs= hdim,
                                                        kernel_size = kernel_size, stride = 1, 
                                                        dilation = dilation_size, padding = (kernel_size-1)*dilation_size,
                                                        dropout = dr_rates))
        self.mlp_head = nn.Linear(hdim, out_len)
        self.kernel_size = kernel_size
        self.len_receptive = self.receptive_field_length()

    def receptive_field_length(self):
        return 1+2*(self.kernel_size-1)*np.sum(self.dilation_list)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.arrange1(x)
        for temporal_conv_layer in self.tcn_stack:
            x = temporal_conv_layer(x)
        x = self.arrange2(x)
        y_pred = self.mlp_head(x[:,-1,:])
        return y_pred


class DualTemporalConvNet_V3(LTS_model):
    '''
    DualTCN : in general dilation decreases as stacking more layer. in DualTCN, there exists two dilation factors. inc_dilation increase and dec_dilation decrease.
    Feature map from both dilation directions are concatenated thus the model leverage both recent time step and past time step.
    '''
    def __init__(self, c_in, d_model, window= 168, out_len = 24, hdim=128, n_layers=3, inc_dilation = [1,2,3,4,8,8,16], dec_dilation = [24,8,6,4,3,2,1] , kernel_size = 3, dr_rates = 0.2, use_time_feature = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_in = c_in
        self.use_time_feature = use_time_feature
        self.input_embedding = DataEmbedding(c_in, d_model, use_time_feature=self.use_time_feature)
        self.arrange1 = Rearrange('b n d -> b d n') # (batch, length, depth) -> (batch, depth, length)
        self.arrange2 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)
        self.tcn_inc_stack = nn.ModuleList([])
        self.tcn_dec_stack = nn.ModuleList([])
        self.inc_dilation = inc_dilation
        self.dec_dilation = dec_dilation
        assert len(inc_dilation)==len(dec_dilation)
        self.n_stack = len(inc_dilation)

        for ii in range(n_layers):
            for i in range(self.n_stack):
                if i ==0 and ii==0:
                    self.tcn_inc_stack.append(TemporalBlock(n_inputs = d_model, n_outputs= hdim//2,
                                                        kernel_size = kernel_size, stride = 1, 
                                                        dilation = self.inc_dilation[i], padding = (kernel_size-1)*self.inc_dilation[i],
                                                        dropout = dr_rates))
                    self.tcn_dec_stack.append(TemporalBlock(n_inputs = d_model, n_outputs= hdim//2,
                                                        kernel_size = kernel_size, stride = 1, 
                                                        dilation = self.dec_dilation[i], padding = (kernel_size-1)*self.dec_dilation[i],
                                                        dropout = dr_rates))
                else:
                    self.tcn_inc_stack.append(TemporalBlock(n_inputs = hdim, n_outputs= hdim//2,
                                                        kernel_size = kernel_size, stride = 1, 
                                                        dilation = self.inc_dilation[i], padding = (kernel_size-1)*self.inc_dilation[i],
                                                        dropout = dr_rates))
                    self.tcn_dec_stack.append(TemporalBlock(n_inputs = hdim, n_outputs= hdim//2,
                                                        kernel_size = kernel_size, stride = 1, 
                                                        dilation = self.dec_dilation[i], padding = (kernel_size-1)*self.dec_dilation[i],
                                                        dropout = dr_rates))

        self.mlp_head = nn.Linear(hdim, out_len)
        self.kernel_size = kernel_size
        self.len_receptive = self.receptive_field_length()
        # assert self.len_receptive['inc_recep'] > window,  'receptive field does not cover input window'
        # assert self.len_receptive['dec_recep'] > window,  'receptive field does not cover input window'

    def receptive_field_length(self):
        inc_recep = 1+2*(self.kernel_size-1)*np.sum(self.inc_dilation)
        dec_recep = 1+2*(self.kernel_size-1)*np.sum(self.dec_dilation)
        return {'inc_recep':inc_recep, 'dec_recep':dec_recep}

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.arrange1(x)
        for i in range(self.n_stack):
            x_dense = self.tcn_inc_stack[i](x)
            x_sparse = self.tcn_dec_stack[i](x)
            x = torch.cat((x_dense,x_sparse),axis=1)
        x = self.arrange2(x)
        y_pred = self.mlp_head(x[:,-1,:])
        return y_pred

#%%
class TemporalConvNet_V4(LTS_model):
    def __init__(self, c_in, d_model, window= 168, out_len= 24, n_layers = 5, latent_window=7, hdim = 128, dilation_list = [1,2,3,4,8,8,16] , kernel_size = 3, dr_rates = 0.2, use_time_feature = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_in = c_in
        self.use_time_feature = use_time_feature
        self.input_embedding = DataEmbedding(c_in, d_model, use_time_feature=self.use_time_feature)
        self.arrange1 = Rearrange('b n d -> b d n') # (batch, length, depth) -> (batch, depth, length)
        self.arrange2 = Rearrange('b d n -> b n d') # (batch, length, depth) -> (batch, depth, length)
        self.tcn_stack = nn.ModuleList([])
        self.dilation_list = dilation_list

        for i in range(n_layers):
            for ii, dilation_size in enumerate(dilation_list):
                if ii ==0 and i==0:
                    self.tcn_stack.append(TemporalBlock(n_inputs = d_model, n_outputs= hdim,
                                                        kernel_size = kernel_size, stride = 1, 
                                                        dilation = dilation_size, padding = (kernel_size-1)*dilation_size,
                                                        dropout = dr_rates))
                else:
                    self.tcn_stack.append(TemporalBlock(n_inputs = hdim, n_outputs= hdim,
                                                        kernel_size = kernel_size, stride = 1, 
                                                        dilation = dilation_size, padding = (kernel_size-1)*dilation_size,
                                                        dropout = dr_rates))
        self.pool_head = nn.AdaptiveAvgPool1d(latent_window)

        self.mlp_head = nn.Linear(hdim*latent_window, out_len)
        self.kernel_size = kernel_size
        self.len_receptive = self.receptive_field_length()
        # interval=window//latent_window
        # self.lat_ind = [-1-i*interval for i in range(latent_window)]

        # assert self.len_receptive > window,  'receptive field does not cover input window'

    def receptive_field_length(self):
        return 1+2*(self.kernel_size-1)*np.sum(self.dilation_list)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.arrange1(x)
        for temporal_conv_layer in self.tcn_stack:
            x = temporal_conv_layer(x)
        x = self.pool_head(x)
        # x = self.arrange2(x)
        y_pred = self.mlp_head(x.flatten(start_dim=1))
        return y_pred

