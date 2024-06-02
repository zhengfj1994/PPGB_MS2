import os
import math
import glob
import copy
import random
import shutil
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from scipy.ndimage import convolve1d
from scipy.signal.windows import triang
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def pickle_load(pkl_file):
    save_file = open(pkl_file,'rb')
    try:
        return(pickle.load(save_file))
    except:
        return []

# For LDS (https://github.com/YyzHarry/imbalanced-regression)
def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


# For LDS (https://github.com/YyzHarry/imbalanced-regression)
def get_bin_idx(label,bin_size,contain_zero=True):
    bin1 = int(label/bin_size)
    bin2 = math.ceil(label/bin_size)
    if contain_zero:
        if bin1 == bin2: return bin1
        else: return bin1 + 1
    else:
        if bin1 == bin2: return bin1-1
        else: return bin1

def LDS_weight(data_list,bin_size, weighted_power):
    # preds, labels: [Ns,], "Ns" is the number of total samples
    labels = [_['y'].numpy().tolist() for _ in data_list]
    # assign each label to its corresponding bin (start from 0)
    # with your defined get_bin_idx(), return bin_index_per_label: [Ns,] 
    bin_index_per_label = [get_bin_idx(label,bin_size=bin_size,contain_zero=True) for label in labels]

    # calculate empirical (original) label distribution: [Nb,]
    # "Nb" is the number of bins
    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

    # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
    lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)
    # calculate effective label distribution: [Nb,]
    eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')

    # Use re-weighting based on effective label distribution, sample-wise weights: [Ns,]
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    weights = [np.float32(len(data_list) / x) for x in eff_num_per_label]
    
    for i,j in zip(data_list,weights): i['weight'] = j ** weighted_power # calculate the weight of y in loop

    return data_list


def raw_LDS_weight(data_list, bin_size, weighted_power):
    # preds, labels: [Ns,], "Ns" is the number of total samples
    labels = [_[3]['intensity'] for _ in data_list]
    CEs = [int(_[4]['CE']) for _ in data_list]
    # assign each label to its corresponding bin (start from 0)
    # with your defined get_bin_idx(), return bin_index_per_label: [Ns,] 
    bin_index_per_label = [get_bin_idx(label,bin_size=bin_size,contain_zero=True) for label in labels]

    # calculate empirical (original) label distribution: [Nb,]
    # "Nb" is the number of bins
    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

    # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
    lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)
    # calculate effective label distribution: [Nb,]
    eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')

    # Use re-weighting based on effective label distribution, sample-wise weights: [Ns,]
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    # weights_int = [np.float32(len(data_list) / x) for x in eff_num_per_label]
    # weights_CE = [CE ** 1 for CE in CEs]
    # weights = [x * y for x, y in zip(weights_int, weights_CE)]
    weights = [np.float32(len(data_list) / x) for x in eff_num_per_label]
    
    for i,j in zip(data_list,weights): i[3]['weight'] = j ** weighted_power # calculate the weight of y in loop

    return data_list