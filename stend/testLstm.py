#from https://www.aiproblog.com/index.php/2018/11/13/how-to-develop-lstm-models-for-time-series-forecasting/
# univariate multi-step vector-output stacked lstm example
import numpy as np
from numpy import array
from numpy import hstack
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot
from datetime import datetime#, date, time
import time
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
import random

import keras.backend as K


def my_metric(y_true, y_pred):
	return K.mean(abs(y_true - y_pred))

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# split a multivariate sequence into samples
def split2_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def research(areLog, scaler, areScale, test, predict):
	#t = test.transpose()
	t = test.copy()
	pr = predict.squeeze() # for seq2seq
	#p = pr.transpose()
	p = pr.copy()
	rmses = []
	avgo = []
	for i in range(0, len(p)):
# report performance
		ti = t[i].reshape(-1, 1)
		#ti = t[i].copy()
		pi = p[i].reshape(-1, 1)
		#pi = p[i].copy()
		if areScale:
			ti = scaler.inverse_transform(ti)
			pi = scaler.inverse_transform(pi)
		if areLog:
			ti = np.exp(ti)
			pi = np.exp(pi)
		#print('ti=', ti)
		#print('pi=', pi)
		tips = []
		pips = []
		avgs = []
		rmse1 = None
		for j in range(0, len(pi)):
			tips.append(ti[j][0])
			pips.append(pi[j][0])
			#print('tips=', tips)
			#print('pips=', pips)
			rmse = sqrt(mean_squared_error(tips, pips))
			avg = 1 
			if j > 0:
				avg = round(rmse / j / avgs[0], 3)
			else:
				rmse1 = round(rmse, 3)
			avgs.append(avg)	
		print('rmse1=', rmse1, 'avgs=', avgs)
		rmses.append(rmse1)
		av = round(sum(avgs) / len(avgs), 3)
		avgo.append(av)
	rmseall = round(sum(rmses) / len(rmses), 3)
	avgall = round(sum(avgo) / len(avgo), 3)
	print('avg first rmse= %f avg mean diff=%f' % (rmseall, avgall))
	'''
	r = array(rmses)
	rm = sqrt(np.sum(r)/len(p))
	print('Mean RMSE: %3f' % rm)
	'''	
		
def Simple(units, dropout, n_steps_in, n_steps_out, n_features):
	model = Sequential()
#,dropout=0.2,recurrent_dropout=0.2
	model.add(LSTM(units, activation='relu'
		, input_shape=(n_steps_in, n_features)))
	if dropout > 0:
		model.add(Dropout(dropout))
	model.add(Dense(n_steps_out, activation='linear'))
#sigmoid, relu
	#model.add(Dense(n_steps_out, activation='elu'))
	return model

		
def Stacked(units, dropout, n_steps_in, n_steps_out, n_features):
	units1 = units#int(np.ceil(units / 2))
	print('units1', units1)
	model = Sequential()
	model.add(LSTM(units1, activation='relu'
		, return_sequences=True
		, input_shape=(n_steps_in, n_features)))
	if dropout > 0:
		model.add(Dropout(dropout))
	model.add(LSTM(units1, activation='relu'))
	if dropout > 0:
		model.add(Dropout(dropout))
	model.add(Dense(n_steps_out, activation='linear'))
	#model.compile(optimizer='adam', loss='mse')
	return model

def Seq2seq(units, dropout, n_steps_in, n_steps_out, n_features):
	model = Sequential()
	model.add(LSTM(units, activation='relu', input_shape=(n_steps_in, n_features)))
	if dropout > 0:
		model.add(Dropout(dropout))
	model.add(RepeatVector(n_steps_out))
	model.add(LSTM(units, activation='relu', return_sequences=True))
	if dropout > 0:
		model.add(Dropout(dropout))
	model.add(TimeDistributed(Dense(1, activation='linear')))
	#model.add(TimeDistributed(Dense(n_steps_out, activation='linear')))
	return model	

def prepare1(data, n_features
	, test_size#train_size
	, areLog, scaler, areScale):
	if areLog:
		data = np.log(data)
	if areScale:
		data = scaler.fit_transform(data)
	Xa, ya = split_sequence(data.squeeze(), n_steps_in, n_steps_out)
	l = len(Xa)
	train_size = l - test_size
	X, Xt = Xa[0:train_size,:], Xa[train_size:l,:]
	y, yt = ya[0:train_size,:], ya[train_size:l,:]
	# reshape from [samples, timesteps] into [samples, timesteps, features]
	#n_features = 1
	X = X.reshape((X.shape[0], X.shape[1], n_features))
	Xt = Xt.reshape((Xt.shape[0], Xt.shape[1], n_features))

	return X, y, Xt, yt

def hoursSeq(ind, areScale):
	seql = list()
	for i in range(0, len(ind)):
		#print('dt=%s' % ind[i])
		dt = datetime.strptime(ind[i], '%Y-%m-%d %H:%M:%S')
		h = dt.hour
		if areScale:
			h = h / 24
		#print(h)
		seql.append(h)
	se = array(seql)
	seq = np.reshape(se, (-1, 1))
	return seq

def diff1(data):
	seql = list()
	seql.append(0)
	for i in range(1, len(data)):
		#print('dt=%s' % ind[i])
		delt = data[i] - data[i-1]
		dif = 0
		#if delt != 0:
		#	dif = 1 / delt
		if delt > 0:
			dif = 1
		if delt < 0:
			dif = -1
		seql.append(dif)
	se = array(seql)
	seq = np.reshape(se, (-1, 1))
	return seq

def prepare2(data, index
	, test_size#train_size
	, areLog, scaler, areScale):
	if areLog:
		data = np.log(data)
	if areScale:
		data = scaler.fit_transform(data)
	seq = hoursSeq(index, areScale)
	diff = diff1(data)
	# convert to [rows, columns] structure
	in_seq1 = data.reshape((len(data), 1))
	in_seq2 = seq.reshape((len(seq), 1))
	in_seq3 = diff.reshape((len(diff), 1))
	#dataset = hstack((in_seq1, in_seq2, in_seq3, in_seq1))
	dataset = hstack((in_seq1, in_seq2, in_seq1))
	# covert into input/output
	Xa, ya = split2_sequences(dataset, n_steps_in, n_steps_out)
	l = len(Xa)
	train_size = l - test_size
	#print('len=%d, test=%d, train=%d' % (l, test_size, train_size))
	X, Xt = Xa[0:train_size,:], Xa[train_size:l,:]
	y, yt = ya[0:train_size,:], ya[train_size:l,:]
 
	return X, y, Xt, yt
	
random.seed(7)
areLog = True
areScale = True
#areDiff = True
test_size = 8
start = 1400#24*20
sample_size = 1400#24*15
#train_size = sample_size - test_size
# choose a number of time steps
n_steps_in = 4
n_steps_out = 24
#batch_size = 64#1
# see https://www.researchgate.net/publication/258393467_Review_on_Methods_to_Fix_Number_of_Hidden_Neurons_in_Neural_Networks
#units = int(sqrt(n_steps_in * n_steps_out)) # Shibata and Ikeda method
#units = 12#n_steps_in * n_steps_out#int((n_steps_in + n_steps_out)/2)
dropout = 0.15
#print('units=%d' % units)
epochs = 1000
patience = int(epochs*0.05)
s1 = ''
if areLog:
	s1 += 'Logarithm values '
if areScale:
	s1 += 'Scaled values'
print(s1)
df = read_csv('o.csv', index_col=0)#, parse_dates=True)
dd = df[start:start+sample_size]
data = dd[['NO2']].values.astype(float)
scaler = MinMaxScaler(feature_range=(-1, 1))
areSeq2seq, areMultiVar, areStack = False, False, False
X, y, Xt, yt, model = None, None, None, None, None
#for j in [0, 1, 2, 3]:
'''
n_steps_in = 0
while n_steps_in < 24:
	if n_steps_in == 0:
		n_steps_in = 1
	else:
		n_steps_in *= 2
	if n_steps_in > 24:
		n_steps_in = 24
'''
units = 2
while units < 513:
	units *= 2
	j = 3
	if j == 0:
		areSeq2seq = False
		areMultiVar = False
		areStack = False
		print('SIMPLE LSTM')
	if j == 1:
		areSeq2seq = False
		areMultiVar = False
		areStack = True
		print('STACKED LSTM')
	if j == 2:
		areSeq2seq = True
		areMultiVar = False
		areStack = False
		print('DECODER-ENCODER LSTM')
	if j == 3:
		areSeq2seq = False
		areMultiVar = True
		areStack = False
		print('TIME SERIES WITH HOURS')
	
	if areMultiVar:
		X, y, Xt, yt = prepare2(data, dd.index, test_size, areLog, scaler, areScale)
	else:
		X, y, Xt, yt = prepare1(data, 1, test_size, areLog, scaler, areScale) #n_features = 1
	if areSeq2seq:
		y = y.reshape((y.shape[0], y.shape[1], X.shape[2]))
		model = Seq2seq(units, dropout, n_steps_in, n_steps_out, X.shape[2])
	else:
		if areStack:
			model = Stacked(units, dropout, n_steps_in, n_steps_out, X.shape[2])
		else:
			model = Simple(units, dropout, n_steps_in, n_steps_out, X.shape[2])
	batch_size = len(X)
	print('samples=%d' % len(X))
	print('start=%d, data length=%d, samples=%d, test size=%d' % (start, sample_size, len(X), test_size))
	print('steps in=%d, steps out=%d' % (n_steps_in, n_steps_out)) 
	print('epochs=%d, patience=%d, units=%d, dropout=%3f, batch_size=%d' % (epochs, patience, units, dropout, batch_size))
	model.compile(optimizer='adam'
		, metrics=['mae', my_metric]#['mse', 'mae', 'mape', 'acc']
		#, metrics = 'accuracy'
		#, loss=my_metric
		#, loss='mae'
		, loss='mse'
		)
	now = time.time()
	model.fit(X, y, epochs=epochs
		, validation_split=0.1
		, callbacks=[EarlyStopping(monitor='val_my_metric'
		#'my_metric'
		#'val_mean_absolute_error'
		#'val_acc'#'val_loss'
			,patience=patience
			, mode='min'#'auto'
			, verbose=1
			, min_delta=0.001
			, restore_best_weights=True)]
		, batch_size=batch_size, verbose=0, shuffle=False)
	later = time.time()
	print('model fit %d sec' % int(later - now))
	yhat = model.predict(Xt, verbose=0, batch_size=batch_size)
	research(areLog, scaler, areScale, yt, yhat)
	
	