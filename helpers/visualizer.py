import matplotlib.pyplot as plt
import pandas as pd
import datetime
from pathlib import Path


def plot(dataset, xlabel='DateTime', ylabel='Value', title='Plot'):
    dataset.plot()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.show()
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_path = Path(__file__).parents[1]
    plt.savefig('%s/plots/plot_%s.png' % (full_path, now))


def scatter_plot(dataset, xlabel='DateTime', ylabel='Value', title='Plot',
                 point_size=10):
    plt.scatter(dataset.index, dataset, s=point_size)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
