# from https://www.analyticsvidhya.com/blog/2018/02/time-series-forecasting-methods/
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
data = pd.read_csv('5e.csv', index_col=0)
tstart = '2017-02-01 00:00:00'
tend = '2017-03-31 23:00:00'
fstart = '2017-04-01 00:00:00'
fend = '2017-04-01 23:00:00'
train = data.loc[tstart:tend]
test = data.loc[fstart:fend]

y_hat_avg = test.copy()
fit1 = ExponentialSmoothing(np.asarray(train['CO']) ,seasonal_periods=24, trend='add', seasonal='add',).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot( train['CO'], label='Train')
plt.plot(test['CO'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()
