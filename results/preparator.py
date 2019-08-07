from numpy import hstack, array


def prepare_data(datas, validation_size, test_size, batch_size, n_steps_in, n_steps_out):
    print('n_steps_in=%d, n_steps_out=%d' % (n_steps_in, n_steps_out))
    for i in range(0, len(datas)):
        datas[i] = datas[i].reshape((len(datas[i]), 1))

    dataset = hstack(datas)
    Xl, yl = list(), list()
    for i in range(len(dataset)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out#!!!!!! - 1
        if out_end_ix > len(dataset):
            break
        #!!!!!!seq_x, seq_y = dataset[i:end_ix, :-1], dataset[end_ix - 1:out_end_ix, -1]
        seq_x, seq_y = dataset[i:end_ix, :-1], dataset[end_ix:out_end_ix, -1]
        Xl.append(seq_x)
        yl.append(seq_y)

    Xa = array(Xl)
    ya = array(yl)
    #print('yl.shape=', yl.shape)
    print('ya.shape=', ya.shape)
    length_all = len(Xa)
    length_train_val = length_all - test_size

    length_val = validation_size
    to_val = length_val - length_val % batch_size

    length_train = length_train_val - length_val
    to_train = length_train - length_train % batch_size
    to_test = test_size - test_size % batch_size

    X = Xa[0:to_train, :]
    Xv = Xa[length_train:length_train + to_val, :]
    Xt = Xa[length_train_val:length_train_val + to_test, :]
    y = ya[0:to_train, :]
    yv = ya[length_train:length_train + to_val, :]
    yt = ya[length_train_val:length_train_val + to_test, :]
    print('yt.shape=', yt.shape)
    return X, y, Xv, yv, Xt, yt


def inverse_scale(scaler, test, pred):
    print('test.shape=', test.shape)
    print('pred.shape=', pred.shape)
    #reshaped_test = test.reshape(test.shape[0], test.shape[1])#(24, 24)
    #print('reshaped_test.shape=', reshaped_test.shape)

    #!!!!!!! inverse_test = scaler.inverse_transform(reshaped_test)
    inverse_test = scaler.inverse_transform(test)
    inverse_pred = scaler.inverse_transform(pred)

    return inverse_test, inverse_pred
