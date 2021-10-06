#-*- coding:utf-8 -*-

import os
import sys
import time
import random
import math
import pickle
import unicodedata

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import networkx as nx
import scipy.stats as st
from scipy.stats import norm
from scipy.optimize import curve_fit

def gaussian(x, amplitude, mean, standard_deviation):
    return amplitude * np.exp( - (x - mean)**2 / (2*standard_deviation ** 2))

import pmdarima as pm
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

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

rTE = importr('RTransferEntropy', lib_loc="/home/yongkyung/R/x86_64-pc-linux-gnu-library/3.6")
rTE.set_quiet(True)


def to_categorical(x):
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


def generate_boot_samples(y, n_boot=100, window=2, decompose=True):
    y = y.fillna(method='ffill')
    y_fit = y.rolling(window).mean()
    # y_fit = y_fit.shift(1)
    residual = y - y_fit
    
#     arima = ARIMA(y, order=(1, 0, 0), enforce_stationarity=False, enforce_invertibility=True)
#     arima_res = arima.fit()
#     y_fit = arima_res.fittedvalues
#     residual = y - y_fit

#     arima = pm.auto_arima(y, seasonal=False, stationary=False)
#     y_fit = pd.Series(arima.arima_res_.fittedvalues, index=y.index)
#     residual = y - y_fit
    
    y_fit = y_fit[window:].reset_index(drop=True)
    residual = residual[window:].reset_index(drop=True)

    if not decompose:
        residual = y[window:]
    
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
        if not decompose:
            y_boot = residual_b.reshape(-1,1)
        boot_sample_list.append(y_boot)

    return boot_sample_list



def get_lag_by_ETE(sample):
    out = []
    for lag_test in np.arange(1,25,1):
        x_copy = sample[0]
        y_copy = sample[1]
        ETE_value = rTE.calc_ete(x = x_copy[1:(len(x_copy)-lag_test+1)], y = y_copy[(lag_test):len(y_copy)],lx=1,ly=1)
        out.append([lag_test, np.asarray(ETE_value).item()])
    return out


def get_boot(x, y, lag=None, n_boot=100, decompose=True, plot=True, title=None, raw=None, save=None, curve=None):
    # one-sample
    out = []
    for lag_test in np.arange(1,25,1):
        x_copy = x.to_numpy().copy()
        y_copy = y.to_numpy().copy()

        ETE_value = rTE.calc_ete(x = x_copy[1:(len(x_copy)-lag_test)+1], y = y_copy[(lag_test):len(y_copy)],lx=1,ly=1)
        out.append([lag_test, np.asarray(ETE_value).item()])
    TE_lag = out[np.argmax(np.array(out)[:,1], axis=0)][0]

    # bootstrap
    x_boot_samples = generate_boot_samples(x, n_boot=n_boot, window=2, decompose=decompose)
    y_boot_samples = generate_boot_samples(y, n_boot=n_boot, window=2, decompose=decompose)

    boot_sample = np.stack([x_boot_samples, y_boot_samples])
    boot_sample = np.transpose(boot_sample, (1,0,2,3))

    TE_boot_value = process_map(get_lag_by_ETE, boot_sample, max_workers=32)
    TE_boot_out = [out[np.argmax(np.array(out)[:,1], axis=0)] for out in TE_boot_value]
    TE_boot_count = pd.Series(np.array(TE_boot_out)[:,0]).value_counts()
    TE_boot_lag = np.bincount(np.array(TE_boot_out)[:,0].astype(int)).argmax()

    TE_boot_mean = (TE_boot_count.keys().tolist() * TE_boot_count.values).sum()/n_boot
    
    # Curve-fitting
    # bin_centers = TE_boot_count.reindex(range(25)).fillna(0).keys().to_numpy() + 0.5
    # bin_heights = TE_boot_count.reindex(range(25)).fillna(0).values
    bin_centers = TE_boot_count.keys().to_numpy() + 0.5
    bin_heights = TE_boot_count.values / TE_boot_count.sum()
    
    if n_boot > 1:
        try:
            popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[0.1, np.mean(TE_boot_out), np.std(TE_boot_out)])
        except RuntimeError:
            popt = [0.1, np.mean(TE_boot_out), np.std(TE_boot_out)]

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
            plt.savefig('img/{}.png'.format(save), bbox_inches='tight')
        # plt.show()
        
        # save fig 1
        fig, ax = plt.subplots(1, 1, figsize=(8,8))

        ax.plot(x.reset_index(drop=True), color='k', lw=2)
        ax.plot(y.reset_index(drop=True), color='r', ls='dashed', lw=2)
        ax.legend(['  X','  Y'], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=20)
        ax.axvline(10, color='k', ls = ':', alpha=0.5)
        ax.set_xticks(np.arange(0,140,20))
        ax.set_xticklabels(np.arange(0,140,20), fontsize=16)
        if raw:
            ax.set_yticks(0.1*np.arange(0,12,2))
            ax.set_yticklabels(0.1*np.arange(0,12,2), fontsize=16)
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
        else:
            ax.set_yticks(np.arange(0,120,20))
            ax.set_yticklabels(np.arange(0,120,20), fontsize=16)
        ax.set_xlabel('Time (min)', fontsize=20)
        if raw:
            ax.set_ylabel('Normalized speed value', fontsize=20)
        else:
            ax.set_ylabel('Speed value', fontsize=20)           
        plt.savefig('img/{}_1.png'.format(save), bbox_inches='tight')
        
        # save fig 2
        fig, ax = plt.subplots(1, 1, figsize=(8,8))

        if raw:
            ax.bar(np.array(raw[1].keys())-0.2, raw[1].values/raw[1].sum(), width=0.4, alpha=0.8)
            ax.bar(np.array(TE_boot_count.keys())+0.2, TE_boot_count.values/TE_boot_count.sum(), width=0.4, alpha=0.8)
        else:
            ax.bar(TE_boot_count.keys(), TE_boot_count.values/TE_boot_count.sum(), width=0.6, alpha=0.8)
        ax.set_xticks(np.arange(0,25,2))
        ax.set_xticklabels(np.arange(0,25,2), fontsize=16)
        ax.set_ylim(0,0.52)
        ax.set_yticks(0.1*np.arange(0,5.5,0.5))
        ax.set_yticklabels(10*np.arange(0,5.5,0.5), fontsize=16)

        ax.set_xlabel('Time lag (min)', fontsize=20)
        ax.set_ylabel('Estimated time lag frequency (%)', fontsize=20)
        if raw:
            ax.legend(['  Raw','  Normalized'], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=20)
        else:
            ax.legend(['  Raw'], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=20)
        if lag:
            ax.axvline(lag, color='k', ls=':', lw=2)
        if raw:
            ax.plot(np.linspace(0, 25, 10000), raw[2][1], label='fit', color='b', lw=2)
            ax.plot(np.linspace(0, 25, 10000), norm.pdf(x_interval_for_fit, *popt), label='fit', color='r', lw=2)
        else:
            ax.plot(np.linspace(0, 25, 10000), norm.pdf(x_interval_for_fit, *popt), label='fit', color='b', lw=2)
        plt.savefig('img/{}_2.png'.format(save), bbox_inches='tight')        
        
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

# apply different normalization 
# 1. road restriction min-max: 0~1 
# 2. standard scale + scale: 0~1
# 3. periodic normalization: 0~1

def scaling(x, max_val=None, period=None):
    x = x.fillna(method='ffill')
    if period == None:
        if max_val == None:
            max_val = x.max()
        return x/max_val
    else: 
        return (x/x.rolling(period).max()).dropna().reset_index(drop=True)[1:]
    
    
def centering(x, period=None):
    x = x.fillna(method='ffill')
    if period == None:
        scaler = StandardScaler()
        x = scaler.fit_transform(pd.DataFrame(x))
        x = (x - x.min()) / (x.max() - x.min())
        return pd.Series(x.reshape(-1)) 
    else:
        x_t = []
        for g in x.rolling(period):
            if len(g) != period:
                continue
            scaler = StandardScaler()
            g = scaler.fit_transform(pd.DataFrame(g))
            g = (g - g.min()) / (g.max() - g.min())    
            x_t.append(g[-1].item())
        return pd.Series(x_t)[1:]
    
    
def normalization(x, scale=False, period=None):
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

















