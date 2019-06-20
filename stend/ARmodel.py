# from https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
import numpy
from time import time
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
import pandas as pd

# series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
'''
series = Series.from_csv('d2018_2.csv', header=0)
# split dataset
X = series.values

X = numpy.delete(X, (0), axis=0)
train, test = X[1:len(X)-7], X[len(X)-7:]
'''
data = pd.read_csv('5e.csv', index_col=0)
tstart = '2017-02-01 00:00:00'
tend = '2017-02-05 23:00:00'
fstart = '2017-02-06 00:00:00'
fend = '2017-02-06 23:00:00'
train = data.loc[tstart:tend]
test = data.loc[fstart:fend]

# train autoregression
t0 = time()
model = AR(train.CO)
t1 = time()
print('build model %f' % (t1 - t0))
model_fit = model.fit(maxlag=None, ic='aic')
t2 = time()
print('fit model %f' % (t2 - t1))
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
t3 = time()
print('predict model %f' % (t3 - t2))
'''
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
'''
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
p = pd.DataFrame(predictions, index=test.index, columns=['CO'])
pyplot.plot(test.CO)
pyplot.plot(p.CO, color='red')
pyplot.show()
