
import numpy as np
import pandas as pd
import torch
import random
import os


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def ensure_dir(file_path):  # if there is no directory then create one
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
def MAPE(y_true,y_pred):    # MAPE with mean along channel i.e., N x W x d >> N x W
    ape = abs(y_true-y_pred)/y_true*100
    mape = np.mean(ape,axis=-1)
    return mape
def MSE(y_true,y_pred):    # MSE with mean along channel i.e., N x W x d >> N x W
    se = (y_true-y_pred)**2
    mse = np.mean(se,axis=-1)
    return mse
def nanMAPE(y_true,y_pred): # optional
    ape = abs(y_true-y_pred)/y_true*100
    mape = np.nanmean(ape,axis=-1)
    return mape
def nanMSE(y_true,y_pred):  # optional
    se = (y_true-y_pred)**2
    mse = np.nanmean(se,axis=-1)
    return mse

def time_encode(time_data):
    time_series = pd.Series(time_data)   #time_series = pd.Series(df_target.index)
    hour = time_series.dt.hour
    month = time_series.dt.month
    weekday = time_series.dt.weekday #mon~sun 1~7
    week = time_series.dt.week #i-th week 0~52
    quarter = time_series.dt.quarter #season 1~4
    day = time_series.dt.dayofyear # 0~365

def seed_everything(seed: int = 0):
    '''
    fix random seed for reproduction
    '''
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    # torch.backends.cudnn.deterministic = True  # type: ignore
    # torch.backends.cudnn.benchmark = True  # type: ignore

def ErrorCalculator(in_hat, in_real, out_hat, out_real):
    mape_in = np.round(np.mean(MAPE(y_true=in_real, y_pred=in_hat)),3)
    mape_out = np.round(np.mean(MAPE(y_true=out_real,y_pred=out_hat)),3)
    mse_in = np.round(np.mean(MSE(y_true=in_real,y_pred=in_hat)),3)
    mse_out = np.round(np.mean(MSE(y_true=out_real,y_pred=out_hat)),3)
    return mape_in, mape_out, mse_in, mse_out

def ErrorCalculator2(in_hat, in_real, out_hat, out_real):
    mape_in = np.round(np.mean(MAPE(y_true=in_real, y_pred=in_hat)),3)
    mape_in_std = np.round(np.std(MAPE(y_true=in_real, y_pred=in_hat)),3)
    mape_out = np.round(np.mean(MAPE(y_true=out_real,y_pred=out_hat)),3)
    mape_out_std = np.round(np.std(MAPE(y_true=out_real,y_pred=out_hat)),3)
    mse_in = np.round(np.mean(MSE(y_true=in_real,y_pred=in_hat)),3)
    mse_in_std = np.round(np.std(MSE(y_true=in_real,y_pred=in_hat)),3)
    mse_out = np.round(np.mean(MSE(y_true=out_real,y_pred=out_hat)),3)
    mse_out_std = np.round(np.std(MSE(y_true=out_real,y_pred=out_hat)),3)

    return mape_in,mape_in_std, mape_out,mape_out_std, mse_in, mse_in_std, mse_out, mse_out_std

class PinballLoss(torch.nn.Module):
    def __init__(self,reduction='mean'):
        super().__init__()
        self.reduction = reduction
    def forward(self, y_preds, y_true, quantile):
        errors = y_preds - y_true
        # losses = torch.max(errors*(quantile-1), errors*quantile) >> this was wrong
        losses = torch.max((1-quantile)*errors, -quantile*errors)
        ret = torch.sum(losses, dim=-1)
        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
        return ret


def ErrorCalculator_SG(in_hat, in_real, out_hat, out_real):
    # in_real2=in_real.copy()
    # in_real2[in_real<=1.0332]=np.nan
    # out_real2=out_real.copy()
    # out_real2[out_real<=1.0332]=np.nan
    in_index=np.sum(in_real,axis=-1)>5000
    out_index=np.sum(out_real,axis=-1)>5000
    in_real=in_real[in_index]
    out_real=out_real[out_index]
    in_hat=in_hat[in_index]
    out_hat = out_hat[out_index]
    mape_in = np.round(np.mean(MAPE(y_true=in_real, y_pred=in_hat)),3)
    mape_in_std = np.round(np.std(MAPE(y_true=in_real, y_pred=in_hat)),3)
    mape_out = np.round(np.mean(MAPE(y_true=out_real,y_pred=out_hat)),3)
    mape_out_std = np.round(np.std(MAPE(y_true=out_real,y_pred=out_hat)),3)
    mse_in = np.round(np.mean(MSE(y_true=in_real,y_pred=in_hat)),3)
    mse_in_std = np.round(np.std(MSE(y_true=in_real,y_pred=in_hat)),3)
    mse_out = np.round(np.mean(MSE(y_true=out_real,y_pred=out_hat)),3)
    mse_out_std = np.round(np.std(MSE(y_true=out_real,y_pred=out_hat)),3)

    # mape_in = np.round(np.mean(nanMAPE(y_true=in_real2, y_pred=in_hat)),3)
    # mape_in_std = np.round(np.std(nanMAPE(y_true=in_real2, y_pred=in_hat)),3)
    # mape_out = np.round(np.mean(nanMAPE(y_true=out_real2,y_pred=out_hat)),3)
    # mape_out_std = np.round(np.std(nanMAPE(y_true=out_real2,y_pred=out_hat)),3)
    # mse_in = np.round(np.mean(nanMSE(y_true=in_real2,y_pred=in_hat)),3)
    # mse_in_std = np.round(np.std(nanMSE(y_true=in_real2,y_pred=in_hat)),3)
    # mse_out = np.round(np.mean(nanMSE(y_true=out_real2,y_pred=out_hat)),3)
    # mse_out_std = np.round(np.std(nanMSE(y_true=out_real2,y_pred=out_hat)),3)

    return mape_in,mape_in_std, mape_out,mape_out_std, mse_in, mse_in_std, mse_out, mse_out_std

def ErrorCalculator_SG2(in_hat, in_real, out_hat, out_real):
    in_hat2=in_hat.copy()
    out_hat2=out_hat.copy()
    in_hat2[in_real<=1.0332]=in_real[in_real<=1.0332]
    out_hat2[out_real<=1.0332]=out_real[out_real<=1.0332]

    mape_in = np.round(np.mean(MAPE(y_true=in_real, y_pred=in_hat2)),3)
    mape_in_std = np.round(np.std(MAPE(y_true=in_real, y_pred=in_hat2)),3)
    mape_out = np.round(np.mean(MAPE(y_true=out_real,y_pred=out_hat2)),3)
    mape_out_std = np.round(np.std(MAPE(y_true=out_real,y_pred=out_hat2)),3)
    mse_in = np.round(np.mean(MSE(y_true=in_real,y_pred=in_hat2)),3)
    mse_in_std = np.round(np.std(MSE(y_true=in_real,y_pred=in_hat2)),3)
    mse_out = np.round(np.mean(MSE(y_true=out_real,y_pred=out_hat2)),3)
    mse_out_std = np.round(np.std(MSE(y_true=out_real,y_pred=out_hat2)),3)

    return mape_in,mape_in_std, mape_out,mape_out_std, mse_in, mse_in_std, mse_out, mse_out_std
