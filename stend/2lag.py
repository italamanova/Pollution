from pandas import Series
from matplotlib import pyplot
from pandas.plotting import lag_plot
series = Series.from_csv('o.csv', header=0)
lag_plot(series)
pyplot.show()
