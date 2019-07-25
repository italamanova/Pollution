from pathlib import Path

from astropy.table import Table
from numpy.random import seed

from numpy.matlib import randn
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot

from helpers.preparator import get_data
from helpers.visualizer import simple_plot

sns.set(style="darkgrid")


def normtesttab(df):
    nm_value, nm_p = stats.normaltest(df)
    jb_value, jb_p = stats.jarque_bera(df)
    alpha = 1e-3
    nm_norm = nm_p > alpha
    jb_norm = jb_p > alpha

    data_rows = [('D’Agostino-Pearson', nm_value, nm_p, nm_norm),
                 ('Jarque-Bera', jb_value, jb_p, jb_norm)]
    t = Table(rows=data_rows, names=('Test name', 'Statistic', 'p-value', 'Is normal?'),
              meta={'name': 'normal test table'},
              dtype=('S25', 'f8', 'f8', 'b'))
    print(t)


def my_box_cox(series, alpha=0.05):
    xt, maxlog, interval = stats.boxcox(series, alpha=alpha)
    plt.plot(xt)
    plt.show()

    print('Lambda', maxlog)
    return xt


def plot_my_box_cox(series):
    sns.kdeplot(series, shade=True)
    qqplot(series, line='s')
    plt.show()
    normtesttab(series)

    xt, maxlog, interval = stats.boxcox(series, alpha=0.05)
    sns.kdeplot(xt, shade=True)
    qqplot(xt, line='s')
    plt.show()
    print("lambda = {:g}".format(maxlog))

    normtesttab(xt)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    prob = stats.boxcox_normplot(series, -19, 19, plot=ax)
    plt.plot(prob[0], prob[1], color='b')
    ax.axvline(maxlog, color='r')
    ax.axvline(interval[1], color='g', ls='--')
    ax.axvline(interval[0], color='g', ls='--')
    plt.show()


def my_diff(series):
    pass


def main(file):
    start = '2017-01-01 00:00:00'
    end = '2017-01-10 00:00:00'
    df = get_data(file)
    # df = df.loc[start:end]
    series = df[df.columns[0]].values
    # df = stats.loggamma.rvs(5, size=500) + 5
    simple_plot(df)

    xt = my_box_cox(series)


# path_prepared = '%s/data/centar' % Path(__file__).parents[1]
# path_to_file_prepared = '%s/Centar_PM25_prepared.csv' % path_prepared
path = '%s/data/cut_data' % Path(__file__).parents[1]
file = '%s/Centar_PM25_prepared_270H.csv' % path
main(file)
