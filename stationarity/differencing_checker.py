import datetime
from pathlib import Path
import numpy as np
from matplotlib import pyplot

from scipy.fftpack import fft, fftfreq, ifft
from statsmodels.tsa.seasonal import seasonal_decompose

from analysis.analyzer import get_resampled
from analysis.trend_seasonality_checker import check_seasonal_decomposition
from helpers.preparator import get_data
from helpers.visualizer import simple_plot


def fourier(series, date):
    temp_fft = fft(series)
    temp_psd = np.abs(temp_fft) ** 2
    temp_fftfreq = fftfreq(len(temp_psd), 1. / 12)
    i = temp_fftfreq > 0
    fig, ax = pyplot.subplots(1, 1, figsize=(8, 4))
    ax.plot(temp_fftfreq[i], 10 * np.log10(temp_psd[i]))
    # ax.set_xlim(0, 5)
    ax.set_xlabel('Frequency (1/year)')
    ax.set_ylabel('PSD (dB)')
    pyplot.show()

    temp_fft_bis = temp_fft.copy()
    temp_fft_bis[np.abs(temp_fftfreq) > 1.1] = 0

    temp_slow = np.real(ifft(temp_fft_bis))
    fig, ax = pyplot.subplots(1, 1, figsize=(10, 6))
    ax.plot(series, lw=.5)
    ax.plot_date(date, temp_slow, '-', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('PM25')
    pyplot.show()


def main(file):
    start = '2017-01-01 00:00:00'
    end = '2018-01-01 00:00:00'
    df = get_data(file)
    # df = df.loc[start:end]
    # df = get_resampled(df, 'D')
    # simple_plot(df)
    series = df[df.columns[0]]
    fourier(series, df.index)
    # result = seasonal_decompose(df, model='additive', extrapolate_trend='freq')
    # simple_plot(result)


path_prepared = '%s/pollution_data/centar' % Path(__file__).parents[1]
path_to_file_prepared = '%s/Centar_PM25_prepared.csv' % path_prepared
main(path_to_file_prepared)
