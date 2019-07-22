# from https://www.aiproblog.com/index.php/2018/11/13/how-to-develop-lstm-models-for-time-series-forecasting/
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
from datetime import datetime  # , date, time
import time
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.callbacks import EarlyStopping

from keras import regularizers

from sklearn.preprocessing import MinMaxScaler
import random

'''
def calcRmse(ti, pi):
	#tips = []
	#pips = []
	avgs = []
	rmse1 = None
	for j in range(0, len(pi)):
		#tips.append(ti[j][0])
		#pips.append(pi[j][0])
		tips = ti[j]
		pips = pi[j]
		#print('tips=', tips)
		#print('pips=', pips)
		rmse = sqrt(mean_squared_error(tips, pips))
		avg = 1 
		if j > 0:
			#avg = round(rmse / j / avgs[0], 3)
			avg = round(rmse / rmse1, 3)
		else:
			rmse1 = round(rmse, 3)
		avgs.append(avg)	
	return rmse1, avgs
'''


def calcMape(ti, pi):
    avgs = []
    for j in range(0, len(pi)):
        tips = ti[j][0]
        pips = pi[j][0]
        # print('tips=', tips)
        # print('pips=', pips)
        mape = round(abs(tips - pips) / tips * 100, 3)
        avgs.append(mape)
    return avgs


def research(areLog, scaler, areScale, test, predict, verbose=2):
    # t = test.transpose()
    t = test.copy()
    pr = predict.squeeze()  # for seq2seq
    # p = pr.transpose()
    p = pr.copy()
    first = None
    avgo = []
    for i in range(0, len(p)):
        ti = t[i].reshape(-1, 1)
        # ti = t[i].copy()
        pi = p[i].reshape(-1, 1)
        # pi = p[i].copy()
        if areScale:
            ti = scaler.inverse_transform(ti)
            pi = scaler.inverse_transform(pi)
        if areLog:
            ti = np.exp(ti)
            pi = np.exp(pi)
        # print('ti=', ti)
        # print('pi=', pi)
        avgs = calcMape(ti, pi)
        if i < 1:
            first = avgs.copy()
        avgall = round(np.sum(avgs) / len(avgs), 3)
        avgo.append(avgall)
        if verbose > 1:
            print('avgall=', avgall, ' mapes=', avgs)
    avgoall = round(np.sum(avgo) / len(avgo), 3)
    if verbose > 0:
        print('averages are=%f, first avg=%f' % (avgoall, avgo[0]), ' first mape=', first)
    else:
        print('averages are=%f' % avgoall)


def Simple(units, areStateful, dropout, recurrent_dropout
           , n_steps_in, n_steps_out, n_features, batch_size):
    model = Sequential()
    model.add(LSTM(units
                   # , activation='relu'
                   # , recurrent_activation='tanh'
                   , input_shape=(n_steps_in, n_features)
                   , batch_size=batch_size
                   , stateful=areStateful
                   # ???,  kernel_regularizer=regularizers.l2(0.01)
                   , dropout=dropout
                   , recurrent_dropout=recurrent_dropout
                   ))
    model.add(Dense(n_steps_out
                    , activation='linear'  # 'selu'#'relu'#'linear'
                    # , kernel_regularizer=regularizers.l2(0.01)
                    ))
    return model


def Bidirect(units, areStateful, dropout, recurrent_dropout
             , n_steps_in, n_steps_out, n_features, batch_size):
    model = Sequential()
    model.add(Bidirectional(LSTM(units
                                 # , activation='relu'
                                 , stateful=areStateful
                                 , dropout=dropout
                                 , recurrent_dropout=recurrent_dropout
                                 )
                            , input_shape=(n_steps_in, n_features)
                            , batch_size=batch_size
                            ))
    model.add(Dense(n_steps_out, activation='linear'))
    return model


def Stacked(units, areStateful, dropout, recurrent_dropout
            , n_steps_in, n_steps_out, n_features, batch_size):
    units1 = int(units / 3)  # int(np.ceil(units / 2))
    print('units1', units1)
    model = Sequential()
    model.add(LSTM(2 * units1
                   # , activation='relu'
                   , return_sequences=True
                   , input_shape=(n_steps_in, n_features)
                   , batch_size=batch_size
                   , stateful=areStateful
                   , dropout=dropout
                   , recurrent_dropout=recurrent_dropout
                   ))
    model.add(LSTM(units1
                   # , activation='relu'
                   , batch_size=batch_size
                   , stateful=areStateful
                   , dropout=dropout
                   , recurrent_dropout=recurrent_dropout
                   ))
    model.add(Dense(n_steps_out, activation='linear'))
    return model


def Seq2seq(units, areStateful, dropout, recurrent_dropout
            , n_steps_in, n_steps_out, n_features, batch_size):
    model = Sequential()
    model.add(LSTM(units
                   # , activation='relu'
                   , input_shape=(n_steps_in, n_features)
                   , batch_size=batch_size
                   , stateful=areStateful
                   , dropout=dropout
                   , recurrent_dropout=recurrent_dropout
                   ))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(units
                   # , activation='relu'
                   , return_sequences=True
                   , batch_size=batch_size
                   , stateful=areStateful
                   , dropout=dropout
                   , recurrent_dropout=recurrent_dropout
                   ))
    model.add(TimeDistributed(Dense(1, activation='linear')))
    # model.add(TimeDistributed(Dense(n_steps_out, activation='linear')))
    return model


def prepare1(data  # , n_features
             # , validation_split
             , validation_size
             , test_size
             , batch_size
             , n_steps_in, n_steps_out):
    n_features = 1
    datas = data.squeeze()
    # split data at [series. features=1] for data and [spet_out] for labels
    Xl, yl = list(), list()
    for i in range(len(datas)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(datas):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = datas[i:end_ix], datas[end_ix:out_end_ix]
        Xl.append(seq_x)
        yl.append(seq_y)
    Xa, ya = array(Xl), array(yl)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    Xa = Xa.reshape((Xa.shape[0], Xa.shape[1], n_features))
    ya = ya.reshape((ya.shape[0], ya.shape[1], n_features))

    # calc length fn train, validation, test sequence
    length_all = len(Xa)
    length_train_val = length_all - test_size
    length_val = validation_size  # int(length_train_val * validation_split)
    length_train = length_train_val - length_val
    # tweak length to match with batch_size
    to_val = length_val - length_val % batch_size
    to_train = length_train - length_train % batch_size
    to_test = test_size - test_size % batch_size
    # split data on train, validation, test sequence
    X = Xa[0:to_train, :]
    Xv = Xa[length_train:length_train + to_val, :]
    Xt = Xa[length_train_val:length_train_val + to_test, :]
    y = ya[0:to_train, :]
    yv = ya[length_train:length_train + to_val, :]
    yt = ya[length_train_val:length_train_val + to_test, :]
    return X, y, Xv, yv, Xt, yt


def hoursSeq(ind):
    seql = list()
    for i in range(0, len(ind)):
        dt = datetime.strptime(ind[i], '%Y-%m-%d %H:%M:%S')
        h = dt.hour
        h = h / 24
        seql.append(h)
    se = array(seql)
    seq = np.reshape(se, (-1, 1))
    return seq


def differ(step, data):
    seql = list()
    for i in range(0, len(data)):
        if i < step:
            seql.append(0)
        else:
            delt = data[i][0] - data[i - step][0]
            if not np.isfinite(delt):
                print('i=%d, delt=%3f, di=%3f, di-step=%3f' % (i, delt, data[i][0], data[i - step][0]))
            # print('di=%3f, di-step=%3f, delt=%3f' % (data[i][0], data[i - step][0], delt))
            seql.append(delt)
    se = array(seql)
    seq = np.reshape(se, (-1, 1))
    return seq  # .asType(float)


def prepare2(datas
             # , validation_split
             , validation_size
             , test_size
             , batch_size
             , n_steps_in, n_steps_out
             ):
    for i in range(0, len(datas)):
        datas[i] = datas[i].reshape((len(datas[i]), 1))
    # resample features and predict
    dataset = hstack(datas)
    # split data at [series. features=1] for data and [spet_out] for labels
    Xl, yl = list(), list()
    for i in range(len(dataset)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(dataset):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = dataset[i:end_ix, :-1], dataset[end_ix - 1:out_end_ix, -1]
        Xl.append(seq_x)
        yl.append(seq_y)
    Xa = array(Xl)
    ya = array(yl)
    length_all = len(Xa)
    length_train_val = length_all - test_size
    length_val = validation_size  # int(length_train_val * validation_split)
    # tweak to match with batch_size
    to_val = length_val - length_val % batch_size
    length_train = length_train_val - length_val
    to_train = length_train - length_train % batch_size
    # tweak to match with batch_size
    to_test = test_size - test_size % batch_size
    X = Xa[0:to_train, :]
    Xv = Xa[length_train:length_train + to_val, :]
    Xt = Xa[length_train_val:length_train_val + to_test, :]
    y = ya[0:to_train, :]
    yv = ya[length_train:length_train + to_val, :]
    yt = ya[length_train_val:length_train_val + to_test, :]
    return X, y, Xv, yv, Xt, yt


'''
def findBatch_size(test_size):
	batches = []
	for i in range(1, test_size + 1):
		if (test_size % i) == 0:
			batches.append(i)
	return batches
'''


def transform(d, areScale, scaler):
    d1 = None
    if areScale:
        d1 = scaler.fit_transform(d)
    else:
        d1 = d.copy()
    return d1


# from https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# invert differencing
# yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)

def buildDiffDatas(data, diff_size):
    datas = []
    datas.append(data)
    diff = data.copy()
    for i in range(1, diff_size + 1):
        diff = differ(i, diff)
        datas.append(diff)
    datas.append(data)
    return datas


####################
random.seed(7)
file_name = 'Centar_PM25_prepared.csv'
column_name = 'PM25'
areStateful = False
areLog = True
areScale = True
test_size = 24
start = 800  # 24*20
data_length = 3000  # 24*15
# validation_split = 0.1#05
validation_size = 24
# batches = findBatch_size(test_size)
s1 = ''
if areStateful:
    s1 += 'Stateful '
else:
    s1 += 'Stateless '
if areLog:
    s1 += 'Logarithm values '
if areScale:
    s1 += 'Scaled values'
df = read_csv(file_name, index_col=0)  # , parse_dates=True)
# for h in range(0, 24):
#	start +=1
# if h > 1:
#	validation_split += 0.1
# start += h
dd = df[start:start + data_length]
# some values
data = dd[[column_name]].values.astype(float)
scaler = MinMaxScaler(feature_range=(-1, 1))
if areLog:
    data = np.log(data)
if areScale:
    data = scaler.fit_transform(data)
# data = transform(data, areScale, scaler)
# hours in day scaled at 24
hours = hoursSeq(dd.index)
# first difference
diff1 = differ(1, data)
# diff1 = transform(diff1, areScale, scaler)
diff2 = differ(1, diff1)
# diff2 = transform(diff2, areScale, scaler)
diff3 = differ(1, diff2)
# day difference
diff24 = differ(24, data)
# diff24 = transform(diff24, areScale, scaler)
# datas = [data, diff1, diff2, diff3, diff24, hours, data]
datas = [data, data]
# datas = buildDiffDatas(data, 24)
print('datas len =', len(datas))

X, y, Xv, yv, Xt, yt, model = None, None, None, None, None, None, None
areSeq2seq, areStack, areBidirect = False, False, False
for j in [0, 3]:
    # while batch_size < data_length / 4:
    #	n_steps_in += 1
    # n_steps_out = n_steps_in
    # for batch_size in batches:
    #	j = 0
    if j == 0:
        areSeq2seq, areStack, areBidirect = False, False, False
        print('SIMPLE LSTM')
    if j == 1:
        areSeq2seq, areStack, areBidirect = False, True, False
        print('STACKED LSTM')
    if j == 2:
        areSeq2seq, areStack, areBidirect = True, False, False
        print('DECODER-ENCODER LSTM')

    if j == 3:
        areSeq2seq, areStack, areBidirect = False, False, True
        print('BIDIRECTION LSTM')

    #############################
    n_steps_in = 24
    n_steps_out = 24
    batch_size = 24  # 12#int(len(X)/1)
    if test_size < batch_size:
        test_size = batch_size
    if validation_size < batch_size:
        validation_size = batch_size
    #############################
    if areSeq2seq:
        X, y, Xv, yv, Xt, yt = prepare1(data
                                        # , validation_split
                                        , validation_size
                                        , test_size
                                        , batch_size
                                        , n_steps_in, n_steps_out)
    else:
        X, y, Xv, yv, Xt, yt = prepare2(datas
                                        # , validation_split
                                        , validation_size
                                        , test_size
                                        , batch_size
                                        , n_steps_in, n_steps_out)
    # print(X.shape)
    # print(y.shape)
    #############################
    # units = int(sqrt(n_steps_in * n_steps_out)) # Shibata and Ikeda method
    # units = 12#n_steps_in * n_steps_out#int((n_steps_in + n_steps_out)/2)
    # units = int((n_steps_in + n_steps_out)/2)#int(len(X)/8)
    units = 2 * int(len(X) / 7 / (n_steps_in + n_steps_out))
    dropout = 0.1
    # if areStateful:
    #	recurrent_dropout = 0.3
    # else:
    #	recurrent_dropout = 0.1
    recurrent_dropout = 0
    epochs = 50
    patience = max(1, int(epochs * 0.2))
    ##############################
    if areSeq2seq:
        model = Seq2seq(units, areStateful, dropout, recurrent_dropout
                        , n_steps_in, n_steps_out, X.shape[2], batch_size)
    else:
        if areStack:
            model = Stacked(units, areStateful, dropout, recurrent_dropout
                            , n_steps_in, n_steps_out, X.shape[2], batch_size)
        elif areBidirect:
            model = Bidirect(units, areStateful, dropout, recurrent_dropout
                             , n_steps_in, n_steps_out, X.shape[2], batch_size)
        else:
            model = Simple(units, areStateful, dropout, recurrent_dropout
                           , n_steps_in, n_steps_out, X.shape[2], batch_size)
    print('%s : start=%d, data length=%d, test size=%d' % (s1, start, data_length, test_size))
    # print('samples=%d, validation_stlit=%3f, steps in=%d, steps out=%d,  batch_size=%d'
    #	 % (len(X), validation_split, n_steps_in, n_steps_out, batch_size))
    print('samples=%d, validation_size=%d, steps in=%d, steps out=%d,  batch_size=%d'
          % (len(X), validation_size, n_steps_in, n_steps_out, batch_size))
    print('epochs=%d, patience=%d, units=%d, dropout=%3f, recurrent_dropout=%3f'
          % (epochs, patience, units, dropout, recurrent_dropout))
    model.compile(optimizer='adam'  # 'Nadam'#'Adadelta'#'Adamax'#'RMSprop'
                  , metrics=['acc']  # ['mean_squared_logarithmic_error']
                  # , metrics=['mean_squared_logarithmic_error']#['mse', 'mae', 'mape', 'acc']
                  , loss='mean_squared_logarithmic_error'  # 'mae''mse'
                  )
    config = model.get_config()

    now = time.time()
    estop = 0
    if areStateful:
        best_loss = None
        best_weights = None
        for b in range(epochs):
            history = model.fit(X, y, epochs=1  # epochs
                                , validation_data=(Xv, yv)
                                , batch_size=batch_size
                                , verbose=0, shuffle=False
                                )
            curr_loss = history.history['val_loss']  # without validate ['loss']
            curr_weights = model.get_weights()
            model.reset_states()
            if b < 1:
                best_loss = curr_loss
                best_weights = curr_weights
            # print('b=%d' % (b), 'loss=', curr_loss)
            elif curr_loss < best_loss:
                best_loss = curr_loss
                best_weights = curr_weights
            elif patience > 0:
                patience -= 1
            else:
                estop = b
                break
        model = Sequential.from_config(config)
        model.set_weights(best_weights)
    else:
        es = EarlyStopping(monitor='val_loss'
                           , patience=patience
                           , mode='min'  # 'auto'
                           , verbose=0
                           # , min_delta=0.002
                           , restore_best_weights=True)

        history = model.fit(X, y, epochs=epochs
                            , validation_data=(Xv, yv)
                            , callbacks=[es]
                            , batch_size=batch_size
                            , verbose=0, shuffle=False
                            )
        estop = es.stopped_epoch
    if estop == 0:
        estop = epochs

    later = time.time()
    print('stoped epoch number=%d, model fit %d sec' % (estop, int(later - now)))

    yhat = model.predict(Xt, verbose=0
                         , batch_size=batch_size  # 1
                         )
    research(areLog, scaler, areScale, yt, yhat, 1)
