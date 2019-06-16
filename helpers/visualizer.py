import matplotlib.pyplot as plt
import pandas as pd
import datetime
from pathlib import Path

parent_dir_path = Path(__file__).parents[1]


def simple_plot(dataset, xlabel='DateTime', ylabel='Value'):
    dataset.plot()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot(dataset, xlabel='DateTime', ylabel='Value', title='Plot'):
    dataset.plot()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.show()
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig('%s/plots/plot_%s.png' % (parent_dir_path, now))


def scatter_plot(dataset, xlabel='DateTime', ylabel='Value', title='Plot',
                 point_size=10):
    plt.scatter(dataset.index, dataset, s=point_size)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()



