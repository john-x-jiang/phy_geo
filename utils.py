import sys
import json
import logging
import copy
import numpy as np
import scipy.stats as stats
import torch
from skimage.filters import threshold_otsu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calc_dc(x, y):
    """Dice coefficient between x and y; threshold based on ostu method
    
    Args: 
        x: a matrix of input tissue properties
        y: a matrix of reconstructed tissue properties
    
    Output:
        mean dice coefficient
    """
    thresh_gt = 0.16  # tissue property threshold
    dc = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        idx_gt = np.where(x[i, :] >= thresh_gt)[0]
        if y[i, :].min() == y[i, :].max():
            continue
        thresh_c = threshold_otsu(y[i, :])
        idx_c = np.where(y[i, :] >= thresh_c)[0]
        dc[i] = 2 * len(np.intersect1d(idx_gt, idx_c)) / (len(idx_gt) + len(idx_c))
    return np.mean(dc)


def calc_msse(x, y):
    """Calculates the mean of the sum of squared error 
    between matrices X and Y
    """
    # rmse = np.zeros(x.shape[0])
    # sse = ((x - y) ** 2).sum(axis=1)
    sse = np.sum(np.square(x - y)) / np.prod(x.shape)
    # mae = (np.absolute(x - y)).mean(axis=1)
    # return np.mean(sse)
    return sse


def calc_AT(u, M):
    u = u.transpose(2, 0, 1)
    M = M.transpose(2, 0, 1)
    w, m, n = u.shape
    u_new = np.roll(u, -1, axis=0)
    u_slope = (u_new - u)[:-1, :, :]
    u_AT = np.argmax(u_slope, axis=0)

    M_new = np.roll(M, -1, axis=0)
    M_slope = (M_new - M)[:-1, :, :]
    M_AT = np.argmax(M_slope, axis=0)

    corr_coeff = 0
    #print('Size of AT is :', u_AT.shape)
    u_apd = np.sum((u > 0.7), axis=0)
    u_scar = u_apd < (0.25 * w)
    x_apd = np.sum((M > 0.7), axis=0)
    x_scar = x_apd < (0.25 * w)

    u_AT[u_scar] = 200
    M_AT[x_scar] = 200

    #m,n=u_AT.shape
    count = 0
    for i in range(m):
        true_AT = u_AT[i, :]
        x_AT = M_AT[i, :]
        if (x_AT == x_AT[0]).all() and (true_AT == true_AT[0]).all():
            corr_coeff = corr_coeff + 1
        elif (x_AT == x_AT[0]).all() or (true_AT == true_AT[0]).all():
            count += 1
            continue
        else:
            corr_coeff = corr_coeff + stats.pearsonr(true_AT, x_AT)[0]

    return corr_coeff / m


def calc_DC(u,x):
    u = u.transpose(2, 0, 1)
    x = x.transpose(2, 0, 1)
    w, m, n = u.shape
    u_apd = np.sum((u > 0.7), axis=0)
    u_scar = u_apd > 0.25 * w
    x_apd = np.sum((x > 0.7), axis=0)
    x_scar = x_apd > 0.25 * w
    dice_coeff = 0

    for i in range(m):
        u_row = u_scar[i, :]
        x_row = x_scar[i, :]
        u_scar_index = np.where(u_row == 0)[0]
        x_scar_index = np.where(x_row == 0)[0]

        intersect = set(u_scar_index) & set(x_scar_index)

        dice_coeff = dice_coeff + len(intersect) / float(len(set(u_scar_index)) + len(set(x_scar_index)))

    return 2 * dice_coeff / m, u_scar, x_scar


def calc_corr_Pot(u, x):
    m, n, w = u.shape
    correlation_sum = 0
    count = 0
    for i in range(m):
        for j in range(n):
            a = u[i, j, :]
            b = x[i, j, :]
            if (a == a[0]).all() or (b == b[0]).all():
                count += 1
                continue
            correlation_sum = correlation_sum + stats.pearsonr(a, b)[0]
    return (correlation_sum / (m * n - count))


def calc_corr_Pot_spatial(u, x):
    m, n, w = u.shape
    correlation_sum = 0
    count = 0
    for i in range(m):
        for j in range(w):
            a = u[i, :, j]
            b = x[i, :, j]
            if (a == a[0]).all() or (b == b[0]).all():
                count += 1
                continue
            correlation_sum = correlation_sum + stats.pearsonr(a, b)[0]
    return (correlation_sum / (m * w - count))


def calc_dc_fixedthres(x, y):
    """Dice coefficient with fixed threshold for both healthy and scar regions
    """
    thresh_gt = 0.16
    dch = np.zeros(x.shape[0])
    dcs = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        idx_gth = np.where(x[i, :] <= 0.16)[0]
        idx_gts = np.where(x[i, :] > 0.4)[0]
        idx_ch = np.where(y[i, :] <= 0.16)[0]
        idx_cs = np.where(y[i, :] > 0.4)[0]
        dch[i] = 2 * len(np.intersect1d(idx_gth, idx_ch)) / (len(idx_gth) + len(idx_ch))
        dcs[i] = 2 * len(np.intersect1d(idx_gts, idx_cs)) / (len(idx_gts) + len(idx_cs))
    return np.mean(dch), np.mean(dcs)


def norm_signal(signals):
    features = copy.deepcopy(signals.numpy())
    features = np.squeeze(features)
    min_x, max_x = np.min(features, axis=1), np.max(features, axis=1)
    features = features.transpose()
    features = (features - min_x) / (max_x - min_x)
    features = features.transpose()
    return torch.from_numpy(features)


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

def inline_print(s):
    sys.stdout.write(s + '\r')
    sys.stdout.flush()
    

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
