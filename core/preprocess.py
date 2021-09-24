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


def min_max(x, period=None, max_val=None):
    '''
    (min-max) road restriction min-max: 0~1 
    '''
    x = x.fillna(method='ffill')
    if period == None:
        if max_val == None:
            max_val = x.max()
        return x/max_val
    else: 
        return (x/x.rolling(period).max()).dropna().reset_index(drop=True)[1:]
    
    
def z_score(x, period=None, scale=True):
    '''
    (z-score) standard score with scaling: 0~1
    '''
    x = x.fillna(method='ffill')
    if period == None:
        scaler = StandardScaler()
        x = scaler.fit_transform(pd.DataFrame(x))
        if scale:
            x = (x - x.min()) / (x.max() - x.min())
        return pd.Series(x.reshape(-1)) 
    else:
        x_t = []
        for g in x.rolling(period):
            if len(g) != period:
                continue
            scaler = StandardScaler()
            g = scaler.fit_transform(pd.DataFrame(g))
            if scale:
                g = (g - g.min()) / (g.max() - g.min())    
            x_t.append(g[-1].item())
        return pd.Series(x_t)[1:]
    
    
def nonlinear(x, period=None, scale=True):
    '''
    (nonlinear) periodic normalization with/without scaling: 0~1
    '''
    x = x.fillna(method='ffill')
    if period == None:
        f_25 = x.quantile(0.25)
        f_50 = x.quantile(0.50)
        f_75 = x.quantile(0.75)

        v = 0.5*(x-f_50)/(f_75-f_25)
        x = pd.DataFrame(v).applymap(lambda s: st.norm.cdf(s)).iloc[:,0]

        if scale:
            x = (x-x.min())/(x.max()-x.min())
        return x
    else:
        f_25 = x.rolling(period).quantile(0.25)
        f_50 = x.rolling(period).quantile(0.50)
        f_75 = x.rolling(period).quantile(0.75)

        v = 0.5*(x-f_50)/(f_75-f_25)
        v = v.dropna().reset_index(drop=True)[1:]
        x = pd.DataFrame(v).applymap(lambda s: st.norm.cdf(s)).iloc[:,0]
        
        if scale:
            x = (x-x.min())/(x.max()-x.min())
        return x


