import tensorflow as tf

from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import mdn

LEARNING_RATE = 0.001


class MDRNN():
    def __init__(self, in_dim, out_dim, lstm_units, n_mixes):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.list_units = lstm_units
        self.n_mixes = n_mixes

        self.models = self._build()
        self.model = self.models[0]
        self.forward = self.models[1]

    def _build(self):
        
        # for training
        rnn_x = Input(shape = (None, in_dim))
        lstm = LSTM(lstm_units, return_sequences=True, return_state=True) #old?

        lstm_out, _, _ = lstm(rnn_x)
        mdn_layer = mdn.MDN(out_dim, n_mixes)

        mdn_model = mdn(lstm_out)

        model = Model(rnn_x, mdn_model)

        # for prediction
        lstm_in_h = Input(shape=(lstm_units,))
        lstm_in_c = Input(shape=(lstm_units,))

        lstm_out_forward, lstm_out_h, lstm_out_c = lstm(rnn_x, initial_state = [lstm_in_h, lstm_in_c])

        mdn_forward = mdn_layer(lstm_out_forward)
        
        forward = Model([rnn_x] + [lstm_in_h, lstm_in_c], [mdn_forward, lstm_out_h, lstm_out_c])

        def rnn_loss(z_true, z_pred):
            assert z_true.shape[1] = out_dim
            assert z_pred.shape = (2*out_dim + 1)*n_mixes

            z_loss = mdn_layer.get_mixture_loss_func(out_dim, n_mixes)(z_true, z_pred)
            return z_loss

        opti = Adamn(lr=LEARNING_RATE)
        model.compile(loss=rnn_loss, optimizer=opti)

        return (model, forward)

    def train(self, rnn_in, rnn_put):
        self.model.fit(rnn_input, rnn_output,
                shuffle=False,
                epochs=1,
                batch_size=len(rnn_input))

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    def forward_sample(z_in, lstm_in_h, lstm_in_c):
        mdn_out, lstm_out_h, lstm_out_c = forward(z_in, lstm_in_h, lstm_in_c)
        z_sample = get_mixture_sampling_fun(out_dim, n_mixes)(mdn_out)
        return (z_sample, lstm_out_h, lstm_out_c)
