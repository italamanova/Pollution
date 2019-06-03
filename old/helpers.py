import sys
import warnings

warnings.filterwarnings('ignore')
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from scipy.optimize import minimize

import matplotlib.pyplot as plt


def plotMovingAverage(series, n):
    """
    series - dataframe with timeseries
    n - rolling window size

    """

    rolling_mean = series.rolling(window=n).mean()

    # При желании, можно строить и доверительные интервалы для сглаженных значений
    # rolling_std =  series.rolling(window=n).std()
    # upper_bond = rolling_mean+1.96*rolling_std
    # lower_bond = rolling_mean-1.96*rolling_std

    plt.figure(figsize=(15, 5))
    plt.title("Moving average\n window size = {}".format(n))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    # plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
    # plt.plot(lower_bond, "r--")
    # plt.plot(series[n:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()


def detect_outlier(data):
    data_1 = data['PM10']
    outliers = []
    threshold = 3
    mean_1 = np.mean(data_1)
    std_1 = np.std(data_1)

    for y in data_1:
        z_score = (y - mean_1) / std_1
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers
