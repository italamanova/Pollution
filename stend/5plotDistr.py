import pandas as pd 
from pandas import Series
from matplotlib import pyplot



def plot_distribution(data, col_name):
    pyplot.figure(1)
    pyplot.subplot(211)
    data[col_name].hist()
    pyplot.subplot(212)
    data[col_name].plot(kind='kde')
    pyplot.show()
#series = Series.from_csv('d2018_2.csv', header=0)
df = pd.read_csv('o.csv', index_col=0)#, parse_dates=True)
plot_distribution(df,'NO2')
