#-*- coding:utf-8 -*-

import os
import sys
import time
import random
import math
import pickle

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import scipy.stats as st
from scipy.stats import norm
from scipy.optimize import curve_fit

from datetime import datetime, timedelta
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from sklearn.preprocessing import StandardScaler

from rpy2.robjects.packages import importr
from rpy2.robjects import robject

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.ipython.html
# rpy2.ipython.html.init_printing()

rTE = importr('RTransferEntropy')
rTE.set_quiet(True)

from utils import *

# setup seed
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True

SEED = 12345
seed_everything(SEED)


if not os.path.exists('results'):
    os.mkdir('results')
if not os.path.exists('out_{}'.format(int(sys.argv[3]))):
    os.mkdir('out_{}'.format(int(sys.argv[3])))

raw_results = []
scale_results = []
center_results = []
norm_results = []

scale_p_results = []
center_p_results = []
norm_p_results = []

raw_bounds = []
scale_bounds = []
center_bounds = []
norm_bounds = []

scale_p_bounds = []
center_p_bounds = []
norm_p_bounds = []

for rep in tqdm(range(100)):
    result_mae = []
    result_rmse = []
    
    n = 120
    lag = int(sys.argv[1])
    noise = int(sys.argv[2])

    
    # entire case
    x_scaling = scaling(x)
    x_centering = centering(x)
    # x_norm = normalization(x)
    x_norm_scaled = normalization(x, scale=True)

    y_scaling = scaling(y)
    y_centering = centering(y)
    #y_norm = normalization(y)
    y_norm_scaled = normalization(y, scale=True)

    TE_lag, ETE_value, TE_boot_lag, TE_boot_count, TE_boot_mean, curve = get_boot(x, y, plot=False, curve=True)
    c = list(curve[2])
    c[1] = c[1]*c[1]
    out = [*c, get_mae(TE_boot_count, lag)]
    raw_results.append(out)
    raw_bounds.append(list(get_bound(TE_boot_count, confidence=0.95)))
    result_mae.append(get_mae(TE_boot_count, lag))
    result_rmse.append(get_rmse(TE_boot_count, lag))

    TE_lag, ETE_value, TE_boot_lag, TE_boot_count, TE_boot_mean, curve = get_boot(x_scaling,y_scaling, plot=False, curve=True)
    c = list(curve[2])
    c[1] = c[1]*c[1]
    out = [*c, get_mae(TE_boot_count, lag)]
    scale_results.append(out)
    scale_bounds.append(list(get_bound(TE_boot_count, confidence=0.95)))
    result_mae.append(get_mae(TE_boot_count, lag))
    result_rmse.append(get_rmse(TE_boot_count, lag))

    TE_lag, ETE_value, TE_boot_lag, TE_boot_count, TE_boot_mean, curve = get_boot(x_centering,y_centering, plot=False, curve=True)
    c = list(curve[2])
    c[1] = c[1]*c[1]
    out = [*c, get_mae(TE_boot_count, lag)]
    center_results.append(out)
    center_bounds.append(list(get_bound(TE_boot_count, confidence=0.95)))
    result_mae.append(get_mae(TE_boot_count, lag))
    result_rmse.append(get_rmse(TE_boot_count, lag))

    TE_lag, ETE_value, TE_boot_lag, TE_boot_count, TE_boot_mean, curve = get_boot(x_norm_scaled,y_norm_scaled, plot=False, curve=True)
    c = list(curve[2])
    c[1] = c[1]*c[1]
    out = [*c, get_mae(TE_boot_count, lag)]
    norm_results.append(out)
    norm_bounds.append(list(get_bound(TE_boot_count, confidence=0.95)))
    result_mae.append(get_mae(TE_boot_count, lag))
    result_rmse.append(get_rmse(TE_boot_count, lag))

    # period case
    period = int(sys.argv[3])
    x_pre = [100] * period + np.random.normal(0, 1, period)*noise
    x = pd.Series(np.append(x_pre, x.to_numpy()))
    y_pre = [70] * period + np.random.normal(0, 1, period)*noise
    y = pd.Series(np.append(y_pre, y.to_numpy()))
    
    x_scaling = scaling(x, period=period)[-n:]
    x_centering = centering(x, period=period)[-n:]
    x_norm_scaled = normalization(x, scale=True, period=period)[-n:]
    
    y_scaling = scaling(y, period=period)[-n:]
    y_centering = centering(y, period=period)[-n:]
    y_norm_scaled = normalization(y, scale=True, period=period)[-n:]
    
    TE_lag, ETE_value, TE_boot_lag, TE_boot_count, TE_boot_mean, curve = get_boot(x_scaling,y_scaling, plot=False, curve=True)
    c = list(curve[2])
    c[1] = c[1]*c[1]
    out = [*c, get_mae(TE_boot_count, lag)]
    scale_p_results.append(out)
    scale_p_bounds.append(list(get_bound(TE_boot_count, confidence=0.95)))
    result_mae.append(get_mae(TE_boot_count, lag))
    result_rmse.append(get_rmse(TE_boot_count, lag))

    TE_lag, ETE_value, TE_boot_lag, TE_boot_count, TE_boot_mean, curve = get_boot(x_centering,y_centering, plot=False, curve=True)
    c = list(curve[2])
    c[1] = c[1]*c[1]
    out = [*c, get_mae(TE_boot_count, lag)]
    center_p_results.append(out)
    center_p_bounds.append(list(get_bound(TE_boot_count, confidence=0.95)))
    result_mae.append(get_mae(TE_boot_count, lag))
    result_rmse.append(get_rmse(TE_boot_count, lag))

    TE_lag, ETE_value, TE_boot_lag, TE_boot_count, TE_boot_mean, curve = get_boot(x_norm_scaled,y_norm_scaled, plot=False, curve=True)
    c = list(curve[2])
    c[1] = c[1]*c[1]
    out = [*c, get_mae(TE_boot_count, lag)]
    norm_p_results.append(out)
    norm_p_bounds.append(list(get_bound(TE_boot_count, confidence=0.95)))
    result_mae.append(get_mae(TE_boot_count, lag))
    result_rmse.append(get_rmse(TE_boot_count, lag))

    print(result_mae)
    print(result_rmse)
    
    if rep == 0:
        TE_boot_count_all = TE_boot_count
    else:
        TE_boot_count_all = TE_boot_count_all+TE_boot_count

TE_boot_count_all = TE_boot_count_all.fillna(0)
TE_boot_count_all.to_csv('tmp/boot_{}_{}_{}'.format(period, lag, noise))


results_all = [
                np.array(raw_results).mean(axis=0).tolist() + np.array(raw_results).std(axis=0).tolist(),
                np.array(scale_results).mean(axis=0).tolist() + np.array(scale_results).std(axis=0).tolist(),
                np.array(center_results).mean(axis=0).tolist() + np.array(center_results).std(axis=0).tolist(),
                np.array(norm_results).mean(axis=0).tolist() + np.array(norm_results).std(axis=0).tolist(),
                np.array(scale_p_results).mean(axis=0).tolist() + np.array(scale_p_results).std(axis=0).tolist(),
                np.array(center_p_results).mean(axis=0).tolist() + np.array(center_p_results).std(axis=0).tolist(),
                np.array(norm_p_results).mean(axis=0).tolist() + np.array(norm_p_results).std(axis=0).tolist(),
    
              ]

results_all = pd.DataFrame(results_all, 
                           index=['raw', 'scale', 'center', 'norm', 'scale_p', 'center_p', 'norm_p'], 
                           columns=['mu(mean)', 'std(mean)', 'MAE(mean)', 'mu(std)', 'std(std)', 'MAE(std)'])

results_all.to_csv('results/out_{}/results_{}_{}.csv'.format(period, lag, noise))


