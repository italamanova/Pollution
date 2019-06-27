import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    #return pd.DataFrame(np.convolve(interval, window, 'some'),columns=['MA'])
    return np.convolve(interval, window, 'some')
data = pd.read_csv('o.csv', index_col='time', parse_dates=True)
window_size = 360

dd = data[0:window_size]
ind = dd.index
values = dd.NO2
lvalues = np.log(values)
#plt.plot(values)
#plt.plot(movingaverage(values, 24))
mva = movingaverage(lvalues, 24)
#print('len=%d' % len(mva))
#print(mva)
#diff = data/movingaverage(values, 1)
#print(diff)
d = lvalues / mva
df = pd.DataFrame(d, index=ind, columns=['divide'])

# Multiplicative Decomposition 
result_mul = seasonal_decompose(d, model='multiplicative', extrapolate_trend='freq')

# Additive Decomposition
result_add = seasonal_decompose(d, model='additive', extrapolate_trend='freq')

# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)

result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()
'''
q = d * mva
plt.plot(lvalues, label='lvalues')
plt.plot(mva, label = 'mva')
plt.plot(d, label = 'divide')
#plt.plot(q, label = 'return')
plt.legend()
plt.show()
'''