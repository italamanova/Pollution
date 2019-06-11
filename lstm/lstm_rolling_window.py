import pandas as pd
import numpy
from numpy import array
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from arima.arima_pollution import get_data
from lstm.lstm import lstm_predict, show_plot


def rolling_window(df, window, shift):
    dataframes = []
    start = 0
    end = window
    while end < len(df):
        cut_df = df.iloc[start:end].copy()
        # cut_df.index = pd.to_datetime(cut_df.index)
        dataframes.append(cut_df)
        start += shift
        end += shift
    return dataframes


def exp_lstm(file):
    df = pd.read_csv(file, index_col=0)
    df.index = pd.to_datetime(df.index)
    dataframes = rolling_window(df, 24 * 30, 24 * 7)
    look_back = 1
    for i, dataframe in enumerate(dataframes):
        dataframe = dataframe.values
        trainPredict, testPredict = lstm_predict(dataframe, look_back)
        show_plot(dataframe, look_back, trainPredict, testPredict)
