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


def fourier(series, date, period):
    temp_fft = fft(series)
    temp_psd = np.abs(temp_fft) ** 2
    temp_fftfreq = fftfreq(len(temp_psd), 1. / period)
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


