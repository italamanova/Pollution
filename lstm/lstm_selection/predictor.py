from keras import Sequential
from keras.callbacks import EarlyStopping

from lstm.lstm_selection.models import simple, bidirect, stacked, seq2seq


class Predictor:
    def __init__(self, X, y, Xv, yv, Xt, yt, n_steps_in, n_steps_out, batch_size, is_stateful,
                 units_coef=7,
                 dropout=0.1,
                 recurrent_dropout=0,
                 epochs=50,
                 patience_coef=0.2):
        self.X = X
        self.y = y
        self.Xv = Xv
        self.yv = yv
        self.Xt = Xt
        self.yt = yt
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.batch_size = batch_size
        self.is_stateful = is_stateful
        self.estop = 0

        self.units = 2 * int(len(X) / units_coef / (n_steps_in + n_steps_out))
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.epochs = epochs
        self.patience = max(1, int(epochs * patience_coef))

    def simple_model(self):
        model = simple(self.units, self.is_stateful, self.dropout, self.recurrent_dropout, self.n_steps_in,
                       self.n_steps_out, self.X.shape[2], self.batch_size)
        return model

    def bidirect_model(self):
        model = bidirect(self.units, self.is_stateful, self.dropout, self.recurrent_dropout, self.n_steps_in,
                         self.n_steps_out, self.X.shape[2], self.batch_size)
        return model

    def stacked_model(self):
        model = stacked(self.units, self.is_stateful, self.dropout, self.recurrent_dropout, self.n_steps_in,
                        self.n_steps_out, self.X.shape[2], self.batch_size)
        return model

    def seq2seq_model(self):
        model = seq2seq(self.units, self.is_stateful, self.dropout, self.recurrent_dropout, self.n_steps_in,
                        self.n_steps_out, self.X.shape[2], self.batch_size)
        return model

    def compile_model(self, model, optimizer='adam', metrics=['acc'], loss='mean_squared_logarithmic_error'):
        model.compile(optimizer=optimizer, metrics=metrics, loss=loss)

    def run_model_stateless(self, model):
        es = EarlyStopping(monitor='val_loss',
                           patience=self.patience,
                           mode='min',
                           verbose=0,
                           restore_best_weights=True)

        history = model.fit(self.X, self.y,
                            epochs=self.epochs,
                            validation_data=(self.Xv, self.yv),
                            callbacks=[es],
                            batch_size=self.batch_size,
                            verbose=0,
                            shuffle=False)
        self.estop = es.stopped_epoch

        if self.estop == 0:
            self.estop = self.epochs

        yhat = model.predict(self.Xt, verbose=0, batch_size=self.batch_size)
        return yhat, history

    def run_model_stateful(self, model):
        config = model.get_config()
        best_loss = None
        best_weights = None
        for epoch_num in range(self.epochs):
            history = model.fit(self.X, self.y,
                                epochs=1,
                                validation_data=(self.Xv, self.yv),
                                batch_size=self.batch_size,
                                verbose=0,
                                shuffle=False)
            curr_loss = history.history['val_loss']
            curr_weights = model.get_weights()
            model.reset_states()

            if epoch_num < 1:
                best_loss = curr_loss
                best_weights = curr_weights
            elif curr_loss < best_loss:
                best_loss = curr_loss
                best_weights = curr_weights
            elif self.patience > 0:
                self.patience -= 1
            else:
                self.estop = epoch_num
                break

        model_copy = Sequential.from_config(config)
        model_copy.set_weights(best_weights)

        yhat = model_copy.predict(self.Xt, verbose=0, batch_size=self.batch_size)
        return yhat, history

    def predict(self, model_name='simple'):
        if model_name == 'simple':
            model = self.simple_model()
        if model_name == 'bidirect':
            model = self.bidirect_model()
        if model_name == 'stacked':
            model = self.stacked_model()
        if model_name == 'seq2seq':
            model = self.seq2seq_model()

        self.compile_model(model)

        if is_stateful:
            predictions = self.run_model_stateful(model)
        else:
            predictions = self.run_model_stateless(model)

        return predictions
