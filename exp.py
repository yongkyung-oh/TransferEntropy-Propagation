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

from core.functions import *
from core.preprocess import *

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
if not os.path.exists('results/out_{}'.format(int(sys.argv[3]))):
    os.mkdir('results/out_{}'.format(int(sys.argv[3])))

raw_results = []
min_max_results = []
z_score_results = []
nonlinear_results = []

min_max_p_results = []
z_score_p_results = []
nonlinear_p_results = []


for rep in tqdm(range(2)):
    result_mae = []
    result_rmse = []
    
    n = 120
    lag = int(sys.argv[1])
    noise = int(sys.argv[2])
    x,y = generate_samples(n=n, lag=lag, noise=noise)
    
    # entire case
    x_min_max = min_max(x)
    x_z_score = z_score(x)
    # x_nonlinear = nonlinear(x)
    x_nonlinear_scaled = nonlinear(x, scale=True)

    y_min_max = min_max(y)
    y_z_score = z_score(y)
    #y_nonlinear = nonlinear(y)
    y_nonlinear_scaled = nonlinear(y, scale=True)

    TE_lag, ETE_value, TE_boot_lag, TE_boot_count, TE_boot_mean, curve = get_boot(x, y, plot=False, curve=True)
    c = list(curve[2])
    c[1] = c[1]*c[1]
    out = [*c, get_mae(TE_boot_count, lag)]
    raw_results.append(out)
    result_mae.append(get_mae(TE_boot_count, lag))
    result_rmse.append(get_rmse(TE_boot_count, lag))

    TE_lag, ETE_value, TE_boot_lag, TE_boot_count, TE_boot_mean, curve = get_boot(x_min_max,y_min_max, plot=False, curve=True)
    c = list(curve[2])
    c[1] = c[1]*c[1]
    out = [*c, get_mae(TE_boot_count, lag)]
    min_max_results.append(out)
    result_mae.append(get_mae(TE_boot_count, lag))
    result_rmse.append(get_rmse(TE_boot_count, lag))

    TE_lag, ETE_value, TE_boot_lag, TE_boot_count, TE_boot_mean, curve = get_boot(x_z_score,y_z_score, plot=False, curve=True)
    c = list(curve[2])
    c[1] = c[1]*c[1]
    out = [*c, get_mae(TE_boot_count, lag)]
    z_score_results.append(out)
    result_mae.append(get_mae(TE_boot_count, lag))
    result_rmse.append(get_rmse(TE_boot_count, lag))

    TE_lag, ETE_value, TE_boot_lag, TE_boot_count, TE_boot_mean, curve = get_boot(x_nonlinear_scaled,y_nonlinear_scaled, plot=False, curve=True)
    c = list(curve[2])
    c[1] = c[1]*c[1]
    out = [*c, get_mae(TE_boot_count, lag)]
    nonlinear_results.append(out)
    result_mae.append(get_mae(TE_boot_count, lag))
    result_rmse.append(get_rmse(TE_boot_count, lag))

    # period case
    period = int(sys.argv[3])
    x_pre = [100] * period + np.random.normal(0, 1, period)*noise
    x = pd.Series(np.append(x_pre, x.to_numpy()))
    y_pre = [70] * period + np.random.normal(0, 1, period)*noise
    y = pd.Series(np.append(y_pre, y.to_numpy()))
    
    x_min_max = min_max(x, period=period)[-n:]
    x_z_score = z_score(x, period=period)[-n:]
    x_nonlinear_scaled = nonlinear(x, scale=True, period=period)[-n:]
    
    y_min_max = min_max(y, period=period)[-n:]
    y_z_score = z_score(y, period=period)[-n:]
    y_nonlinear_scaled = nonlinear(y, scale=True, period=period)[-n:]
    
    TE_lag, ETE_value, TE_boot_lag, TE_boot_count, TE_boot_mean, curve = get_boot(x_min_max,y_min_max, plot=False, curve=True)
    c = list(curve[2])
    c[1] = c[1]*c[1]
    out = [*c, get_mae(TE_boot_count, lag)]
    min_max_p_results.append(out)
    result_mae.append(get_mae(TE_boot_count, lag))
    result_rmse.append(get_rmse(TE_boot_count, lag))

    TE_lag, ETE_value, TE_boot_lag, TE_boot_count, TE_boot_mean, curve = get_boot(x_z_score,y_z_score, plot=False, curve=True)
    c = list(curve[2])
    c[1] = c[1]*c[1]
    out = [*c, get_mae(TE_boot_count, lag)]
    z_score_p_results.append(out)
    result_mae.append(get_mae(TE_boot_count, lag))
    result_rmse.append(get_rmse(TE_boot_count, lag))

    TE_lag, ETE_value, TE_boot_lag, TE_boot_count, TE_boot_mean, curve = get_boot(x_nonlinear_scaled,y_nonlinear_scaled, plot=False, curve=True)
    c = list(curve[2])
    c[1] = c[1]*c[1]
    out = [*c, get_mae(TE_boot_count, lag)]
    nonlinear_p_results.append(out)
    result_mae.append(get_mae(TE_boot_count, lag))
    result_rmse.append(get_rmse(TE_boot_count, lag))

    print(result_mae)
    print(result_rmse)
    
    if rep == 0:
        TE_boot_count_all = TE_boot_count
    else:
        TE_boot_count_all = TE_boot_count_all+TE_boot_count


results_all = [
                np.array(raw_results).mean(axis=0).tolist() + np.array(raw_results).std(axis=0).tolist(),
                np.array(min_max_results).mean(axis=0).tolist() + np.array(min_max_results).std(axis=0).tolist(),
                np.array(z_score_results).mean(axis=0).tolist() + np.array(z_score_results).std(axis=0).tolist(),
                np.array(nonlinear_results).mean(axis=0).tolist() + np.array(nonlinear_results).std(axis=0).tolist(),
                np.array(min_max_p_results).mean(axis=0).tolist() + np.array(min_max_p_results).std(axis=0).tolist(),
                np.array(z_score_p_results).mean(axis=0).tolist() + np.array(z_score_p_results).std(axis=0).tolist(),
                np.array(nonlinear_p_results).mean(axis=0).tolist() + np.array(nonlinear_p_results).std(axis=0).tolist(),
    
              ]

results_all = pd.DataFrame(results_all, 
                           index=['raw', 'min-max', 'z-score', 'nonlinear', 'min_max_p', 'z-score_p', 'nonlinear_p'], 
                           columns=['mu(mean)', 'std(mean)', 'MAE(mean)', 'mu(std)', 'std(std)', 'MAE(std)'])

# results_all.to_csv('results/out_{}/results_{}_{}.csv'.format(period, lag, noise))
results_all.to_csv('results/out_{}/s_{}_{}.csv'.format(period, lag, noise))


