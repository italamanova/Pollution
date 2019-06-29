from pandas import Series
import statsmodels.stats.stattools as st
#import statsmodels.iolib.table.SimpleTable
import numpy as np
series = Series.from_csv('o.csv', header=0)
itog = series.describe()
print(itog)
print( 'coefficiet variativnosty(V) = %f' % (itog['std']/itog['mean']))
#=======
#Calculates the Jarque-Bera test for normality
row =  [u'JB', u'p-value', u'skew', u'kurtosis']
jb_test = st.jarque_bera(series)
a = np.vstack([jb_test])
#r = SimpleTable(a, row)
print (a)

