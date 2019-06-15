import matplotlib.pyplot as plt
from numpy import polyfit
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns

from helpers.preparator import get_data


def plot_rolling(df):
    fig, ax = plt.subplots(3, figsize=(12, 9))
    ax[0].plot(df.index, df.data, label='raw data')
    ax[0].plot(df.data.rolling(window=12).mean(), label="rolling mean");
    ax[0].plot(df.data.rolling(window=12).std(), label="rolling std (x10)");
    ax[0].legend()

    ax[1].plot(df.index, df.z_data, label="de-trended data")
    ax[1].plot(df.z_data.rolling(window=12).mean(), label="rolling mean");
    ax[1].plot(df.z_data.rolling(window=12).std(), label="rolling std (x10)");
    ax[1].legend()

    ax[2].plot(df.index, df.zp_data, label="12 lag differenced de-trended data")
    ax[2].plot(df.zp_data.rolling(window=12).mean(), label="rolling mean");
    ax[2].plot(df.zp_data.rolling(window=12).std(), label="rolling std (x10)");
    ax[2].legend()

    plt.tight_layout()
    fig.autofmt_xdate()

    plt.show()

    fig, ax = plt.subplots(2, figsize=(12, 6))
    ax[0] = plot_acf(df.z_data.dropna(), ax=ax[0], lags=20)
    ax[1] = plot_pacf(df.z_data.dropna(), ax=ax[1], lags=20)

    plt.show()


def analyze_rolling(file, start=None, end=None):
    df = get_data(file, start, end)

    # train = df.iloc[:-10, :]
    # test = df.iloc[-10:, :]
    # pred = test.copy()
    # df.plot(figsize=(12, 3))

    df['z_data'] = (df['data'] - df.data.rolling(window=12).mean()) / df.data.rolling(window=12).std()
    df['zp_data'] = df['z_data'] - df['z_data'].shift(12)
    plot_rolling(df)


def box_plot(file):
    df = get_data(file)
    col_name = df.columns[0]

    sns.boxplot(data=df, x='month', y=col_name)
    plt.show()


def check_seasonal_decomposition(df):
    col_name = df.columns[0]

    result = seasonal_decompose(df, model='additive', freq=24)
    result.plot()
    plt.show()


def check_polyfit(df):
    col_name = df.columns[0]

    X = [i % 365 for i in range(0, len(df))]
    y = df[col_name].values
    degree = 4
    coef = polyfit(X, y, degree)
    print('Coefficients: %s' % coef)
    # create curve
    curve = list()
    for i in range(len(X)):
        value = coef[-1]
        for d in range(degree):
            value += X[i] ** (degree - d) * coef[d]
        curve.append(value)
    # plot curve over original data
    plt.plot(df[col_name].values)
    plt.plot(curve, color='red', linewidth=3)
    plt.show()
