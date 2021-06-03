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


def generate_samples(n=120, lag=10, noise=1):
    while True:
        x = [100] * n + np.random.normal(0, 1, n)*noise

        for i in np.arange(10,n):
            if i < 100:
                x[i] = 0.95 * x[i-1] + np.random.normal(0,1)*noise
            else: 
                x[i] = 1.10 * x[i-1] + np.random.normal(0,1)*noise
        x = pd.Series(x)

        if np.min(x) > 0:
            break

    k = 0
    while True:
        random.seed(k)
        np.random.seed(k)

        y = [70] * n + np.random.normal(0, 1, n)*noise
        for i in range(lag, n):
            y[i] = 0.5 * x[i-lag] + 20 + np.random.normal(0,1)*noise
        y = pd.Series(y)

        out = []
        for lag_test in np.arange(1,25,1):
            x_copy = x.to_numpy().copy()
            y_copy = y.to_numpy().copy()

            ETE_value = rTE.calc_ete(x = x_copy[1:(len(x_copy)-lag_test)+1], y = y_copy[(lag_test):len(y_copy)],lx=1,ly=1)
            out.append([lag_test, np.asarray(ETE_value).item()])

        #check TE
        if lag == (np.argmax(np.array(out)[:,1])+1):
            break

        k += 1   

    return x, y


def to_categorical(x):
    '''
    Categorical Encoding
    '''
    x = (x-np.min(x))/(np.max(x)-np.min(x))
    
    x_t = []
    for item in x:
        if item >= 0.0 and item < 0.2:
            x_t.append(0)
        elif item >= 0.2 and item < 0.4:
            x_t.append(1)
        elif item >= 0.4 and item < 0.6:
            x_t.append(2)
        elif item >= 0.6 and item < 0.8:
            x_t.append(3)
        elif item >= 0.8 and item <= 1.0:
            x_t.append(4)
    return x_t


def transition_matrix(transitions):
    '''
    Generate transition matrix
    '''
    n = 1 + max(transitions) #number of states

    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return np.array(M)


def generate_boot_samples(y, n_boot=100, window=2):
    '''
    Generate Bootstrap samples 
    '''
    y = y.fillna(method='ffill')
    y_fit = y.rolling(window).mean()
    residual = y - y_fit
    
    y_fit = y_fit[window:].reset_index(drop=True)
    residual = residual[window:].reset_index(drop=True)

    boot_sample_list = []
    for i in range(n_boot):
        np.random.seed(i)
        x_t = to_categorical(residual)
        M = transition_matrix(x_t)

        x_b = []
        x_b.append(x_t[0])

        while len(x_t) != len(x_b):
            try:
                x_b.append(np.random.choice([0,1,2,3,4], p=M[x_b[-1]]))
            except:
                x_b.append(np.random.choice([0,1,2,3,4], p=[0.2, 0.2, 0.2, 0.2, 0.2]))

        residual_b = []
        for item in x_b: 
            random = np.random.random()*0.2
            residual_b.append(random+item*0.2)

        residual_b = np.array(residual_b) * (np.max(residual)-np.min(residual)) + np.min(residual)
        
        y_boot = y_fit.to_numpy().reshape(-1,1)+residual_b.reshape(-1,1)
        boot_sample_list.append(y_boot)

    return boot_sample_list


def get_lag_by_ETE(sample):
    '''
    Get lag information with ETE values 
    '''
    out = []
    for lag_test in np.arange(1,25,1):
        x_copy = sample[0]
        y_copy = sample[1]
        ETE_value = rTE.calc_ete(x = x_copy[1:(len(x_copy)-lag_test+1)], y = y_copy[(lag_test):len(y_copy)],lx=1,ly=1)
        out.append([lag_test, np.asarray(ETE_value).item()])
    return out


def get_boot(x, y, lag=None, n_boot=100, plot=True, title=None, raw=None, save=None, curve=None):
    '''
    Generate Bootstrap estimation of two time series 
    '''
    # one-sample
    out = []
    for lag_test in np.arange(1,25,1):
        x_copy = x.to_numpy().copy()
        y_copy = y.to_numpy().copy()

        ETE_value = rTE.calc_ete(x = x_copy[1:(len(x_copy)-lag_test)+1], y = y_copy[(lag_test):len(y_copy)],lx=1,ly=1)
        out.append([lag_test, np.asarray(ETE_value).item()])
    TE_lag = out[np.argmax(np.array(out)[:,1], axis=0)][0]

    # Bootstrap
    x_boot_samples = generate_boot_samples(x, n_boot=n_boot, window=2)
    y_boot_samples = generate_boot_samples(y, n_boot=n_boot, window=2)

    boot_sample = np.stack([x_boot_samples, y_boot_samples])
    boot_sample = np.transpose(boot_sample, (1,0,2,3))

    TE_boot_value = process_map(get_lag_by_ETE, boot_sample, max_workers=32)
    TE_boot_out = [out[np.argmax(np.array(out)[:,1], axis=0)] for out in TE_boot_value]
    TE_boot_count = pd.Series(np.array(TE_boot_out)[:,0]).value_counts()
    TE_boot_lag = np.bincount(np.array(TE_boot_out)[:,0].astype(int)).argmax()

    TE_boot_mean = (TE_boot_count.keys().tolist() * TE_boot_count.values).sum()/n_boot
    
    # Curve-fitting
    TE_boot = [[val] * TE_boot_count[val]  for val in TE_boot_count.keys()]
    TE_boot = np.array([x for y in TE_boot for x in y])
        
    popt = norm.fit(TE_boot)
    x_interval_for_fit = np.linspace(0, 25, 10000)
    
    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(24,6))

        ax[0].plot(x.reset_index(drop=True), color='k', lw=2)
        ax[0].plot(y.reset_index(drop=True), color='r', ls='dashed', lw=2)
        # ax[0].legend(['X', 'Y'], loc=1, fontsize=24)
        ax[0].legend(['  X','  Y'], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=20)
        ax[0].axvline(10, color='k', ls = ':', alpha=0.5)
        ax[0].set_xlabel('Time (min)', fontsize=20)
        ax[0].set_ylabel('Speed value', fontsize=20)

        if raw:
            ax[1].plot(np.array(raw[0])[:,0], np.array(raw[0])[:,1], marker='o')
        ax[1].plot(np.array(out)[:,0], np.array(out)[:,1], marker='o')
        ax[1].set_xticks(np.arange(1,25,1))
        ax[1].set_xlabel('Time lag (min)', fontsize=20)
        ax[1].set_ylabel('ETE values', fontsize=20)
        if raw:
            ax[1].legend(['  Raw','  Normalized'], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=20)
        else:
            ax[1].legend(['  Raw'], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=20)
        #ax[1].axvline(TE_lag, color='k', ls=':', lw=2)
        if lag:
            ax[1].axvline(lag, color='k', ls=':', lw=2)
        ax[1].set_ylim(0,0.205)

        if raw:
            ax[2].bar(np.array(raw[1].keys())-0.2, raw[1].values/raw[1].sum(), width=0.4, alpha=0.8)
            ax[2].bar(np.array(TE_boot_count.keys())+0.2, TE_boot_count.values/TE_boot_count.sum(), width=0.4, alpha=0.8)
        else:
            ax[2].bar(TE_boot_count.keys(), TE_boot_count.values/TE_boot_count.sum(), width=0.6, alpha=0.8)
        ax[2].set_xticks(np.arange(1,25,1))
        ax[2].set_ylim(0,0.52)
        ax[2].set_yticks(0.1*np.arange(0,5.5,0.5))
        ax[2].set_yticklabels(10*np.arange(0,5.5,0.5))

        ax[2].set_xlabel('Time lag (min)', fontsize=20)
        ax[2].set_ylabel('Estimated time lag frequency (%)', fontsize=20)
        if raw:
            ax[2].legend(['  Raw','  Normalized'], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=20)
        else:
            ax[2].legend(['  Raw'], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=20)
        #ax[2].axvline(TE_boot_lag, color='k', ls=':', lw=2)
        if lag:
            ax[2].axvline(lag, color='k', ls=':', lw=2)
        if raw:
            ax[2].plot(np.linspace(0, 25, 10000), raw[2][1], label='fit', color='b', lw=2)
            ax[2].plot(np.linspace(0, 25, 10000), norm.pdf(x_interval_for_fit, *popt), label='fit', color='r', lw=2)
        else:
            ax[2].plot(np.linspace(0, 25, 10000), norm.pdf(x_interval_for_fit, *popt), label='fit', color='b', lw=2)
                
        plt.suptitle('{} | MAE: {:3.3f} | RMSE: {:3.3f}'.format(title, get_mae(TE_boot_count, lag), get_rmse(TE_boot_count, lag)), fontsize=24)
        if save:
            plt.savefig('out/img/{}.png'.format(save), bbox_inches='tight')
        plt.show()
        
    if curve:
        return TE_lag, np.array(out), TE_boot_lag, TE_boot_count, TE_boot_mean, [x_interval_for_fit, norm.pdf(x_interval_for_fit, *popt), popt]
    else:
        return TE_lag, np.array(out), TE_boot_lag, TE_boot_count, TE_boot_mean


def get_mae(TE_boot_count, lag):
    TE_boot = [[val] * TE_boot_count[val]  for val in TE_boot_count.keys()]
    TE_boot = np.array([x for y in TE_boot for x in y])
    return np.mean(abs(TE_boot-lag))

def get_rmse(TE_boot_count, lag):
    return np.sqrt((((np.array(TE_boot_count.keys().tolist())-lag)**2) * TE_boot_count.values).sum() / TE_boot_count.values.sum())

def get_bound(TE_boot_count, confidence=0.95):
    TE_boot = [[val] * TE_boot_count[val]  for val in TE_boot_count.keys()]
    TE_boot = np.array([x for y in TE_boot for x in y])
    return st.t.interval(confidence, len(TE_boot)-1, loc=np.mean(TE_boot), scale=st.sem(TE_boot))

