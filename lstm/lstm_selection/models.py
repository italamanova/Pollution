from keras import Sequential
from keras.layers import LSTM, Dense, Bidirectional, RepeatVector, TimeDistributed


def simple(units, is_stateful, dropout, recurrent_dropout, n_steps_in, n_steps_out, n_features, batch_size):
    model = Sequential()
    model.add(LSTM(units,
                   input_shape=(n_steps_in, n_features),
                   batch_size=batch_size,
                   stateful=is_stateful,
                   dropout=dropout,
                   recurrent_dropout=recurrent_dropout
                   ))
    model.add(Dense(n_steps_out, activation='linear'))
    return model


def bidirect(units, is_stateful, dropout, recurrent_dropout, n_steps_in, n_steps_out, n_features, batch_size):
    model = Sequential()
    model.add(Bidirectional(LSTM(units,
                                 stateful=is_stateful,
                                 dropout=dropout,
                                 recurrent_dropout=recurrent_dropout
                                 ),
                            input_shape=(n_steps_in, n_features),
                            batch_size=batch_size
                            ))
    model.add(Dense(n_steps_out, activation='linear'))
    return model


def stacked(units, is_stateful, dropout, recurrent_dropout
            , n_steps_in, n_steps_out, n_features, batch_size):
    units1 = int(units / 3)
    print('units1', units1)
    model = Sequential()
    model.add(LSTM(2 * units1,
                   return_sequences=True,
                   input_shape=(n_steps_in, n_features),
                   batch_size=batch_size,
                   stateful=is_stateful,
                   dropout=dropout,
                   recurrent_dropout=recurrent_dropout
                   ))

    model.add(LSTM(units1,
                   batch_size=batch_size,
                   stateful=is_stateful,
                   dropout=dropout,
                   recurrent_dropout=recurrent_dropout
                   ))

    model.add(Dense(n_steps_out, activation='linear'))
    return model


def seq2seq(units, is_stateful, dropout, recurrent_dropout
            , n_steps_in, n_steps_out, n_features, batch_size):
    model = Sequential()
    model.add(LSTM(units,
                   input_shape=(n_steps_in, n_features),
                   batch_size=batch_size,
                   stateful=is_stateful,
                   dropout=dropout,
                   recurrent_dropout=recurrent_dropout
                   ))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(units,
                   return_sequences=True,
                   batch_size=batch_size,
                   stateful=is_stateful,
                   dropout=dropout,
                   recurrent_dropout=recurrent_dropout
                   ))
    model.add(TimeDistributed(Dense(1, activation='linear')))
    return model
