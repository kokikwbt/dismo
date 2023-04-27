import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorly.base import fold, unfold
from sklearn import preprocessing


class MinMaxScaler(preprocessing.MinMaxScaler):
    """ Extension of sklearn.preprocessing.MinMaxScaler
        for tensor data
    """
    def fit(self, data, mode=0):
        hoge = unfold(data, mode)
        return super().fit(hoge)  # return self

    def fit_transform(self, data, mode=0):
        hoge = unfold(data, mode)
        hoge = super().fit_transform(hoge)
        return fold(hoge, mode, data.shape)

    def inverse_transform(self, data, mode=0):
        hoge = unfold(data, mode)
        hoge = super().inverse_transform(hoge)
        return fold(hoge, mode, data.shape)

    def transform(self, data, mode=0):
        hoge = unfold(data, mode)
        hoge = super().transform(hoge)
        return fold(hoge, mode, data.shape)


class StandardScaler(preprocessing.StandardScaler):
    """ Extension of sklearn.preprocessing.StandardScaler
        for tensor data
    """
    def fit(self, data, mode=0):
        hoge = unfold(data, mode)
        return super().fit(hoge)  # return self

    def fit_transform(self, data, mode=0):
        hoge = unfold(data, mode)
        hoge = super().fit_transform(hoge)
        return fold(hoge, mode, data.shape)

    def inverse_transform(self, data, mode=0):
        hoge = unfold(data, mode)
        hoge = super().inverse_transform(hoge)
        return fold(hoge, mode, data.shape)

    def transform(self, data, mode=0):
        hoge = unfold(data, mode)
        hoge = super().transform(hoge)
        return fold(hoge, mode, data.shape)


def sma(X, width, axis=0):
    """ Simple Moving Average (SMA) """

    unfolded = unfold(X, axis)
    n_sample = unfolded.shape[0]
    smoothed = np.zeros(unfolded.shape)

    for t in range(width):
        if t == 0:
            smoothed[t] = unfolded[t]
        else:
            smoothed[t] = unfolded[:t + 1].mean(axis=0)

    for t in range(width, n_sample):
        st = t - width + 1
        ed = t + 1
        smoothed[t] = unfolded[st:ed].mean(axis=0)

    smoothed = fold(smoothed, axis, X.shape)

    return smoothed


def wma(X, mask, axis=0):
    """ Weighted Moving Average (WMA)"""

    unfolded = unfold(X, axis)
    n_sample = unfolded.shape[0]
    smoothed = np.zeros(unfolded.shape)
    width = mask.shape[0]

    for t in range(width):
        if t == 0:
            smoothed[t] = unfolded[t]
        else:
            smoothed[t] = unfolded[:t + 1].mean(axis=0)

    for t in range(width, n_sample):
        st = t - width + 1
        ed = t + 1
        smoothed[t] = unfolded[st:ed].mean(axis=0)

    smoothed = fold(smoothed, axis, X.shape)

    return smoothed


def ewma(X, mask=None, axis=0):
    """ Exponentially Weighted Moving Avarage (EWMA) """

    unfolded = unfold(X, axis)
    n_sample = unfolded.shape[0]
    smoothed = np.zeros(unfolded.shape)
    width = len(mask)

    for t in range(n_sample):
        smoothed[t] = None

    smoothed = fold(smoothed, axis, X.shape)

    return smoothed


def cyclic_fold(tensor, period, mode=0):
    unfolded = unfold(tensor, mode)  # mode: time axis
    # Get dimensionality
    duration, n_dim = unfolded.shape
    n_season = duration // period
    valid_duration = n_season * period
    shape = (n_season, period, n_dim)
    # Fold
    folded = fold(unfolded[-valid_duration:], 0, shape)
    return np.moveaxis(folded, 0, 1)  # 


def cyclic_unfold(tensor, dims):
    # valid_duration = n_seasons * period
    # valid_duration = np.prod(tensor.shape[:2])
    Z = np.moveaxis(tensor, 0, -1)  # (dims, period, n_seasons)
    # Z = np.moveaxis(tensor, 0, 1)  # (n_seasons, period, dims)
    Z = unfold(Z, 0)  # (dims, valid_duration)
    print(Z.shape)
    return fold(Z.T, 0, (*dims, Z.shape[-1]))


def plot_factors(factors, transpose=False, center=0, linecolor='white', linewidth=0):

    n_modes = len(factors)
    fig, ax = plt.subplots(1, n_modes, figsize=(4 * n_modes, 4))

    for axi, A in zip(ax, factors):
        if transpose: A = A.T
        sns.heatmap(A, ax=axi, center=center, linecolor=linecolor, linewidths=linewidth)

    return fig, ax


def plot_grid_search_result(dismo_obj, scores, filename=None, title=None):

    c_set = np.arange(dismo_obj.maxc - dismo_obj.minc + 1)
    s_set = np.arange(dismo_obj.maxs - dismo_obj.mins + 1)

    X1, X2 = np.meshgrid(c_set, s_set)

    Y = np.reshape(scores, X1.shape)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Grid Search Result")
    ax.set_xlabel("Number of interaction")
    ax.set_ylabel("Number of seasonality")

    surf = ax.plot_surface(X1, X2, Y, cmap='bwr', linewidth=0)
    fig.colorbar(surf)
    fig.tight_layout()
    
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
    
    return fig, ax


def pred2df(tensor, temporal_mode=0, timestamps=None, window_index=None):

    pred = unfold(tensor, temporal_mode)

    df = pd.DataFrame(pred)

    if timestamps is None:
        timestamps = np.arange(pred.shape[0])
        if window_index is not None:
            timestamps += window_index
        df['timestamp'] = timestamps

    if window_index is not None:
        df['window_id'] = window_index

    return df

def seq2df(arr, timestamps=None, window_index=None):
    """
        arr: (#samples, #dims)
    """
    assert arr.ndim == 2
    if window_index is not None:
        assert type(window_index) == int

    df = pd.DataFrame(arr)

    if timestamps is None:
        timestamps = np.arange(arr.shape[0])
        if window_index is not None:
            timestamps += window_index
    
    df['timestamp'] = timestamps

    if window_index is not None:
        df['window_id'] = window_index

    return df

def saveas_json(filename, data, indent=2):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=indent)

def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)