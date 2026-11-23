"""Neural forecasters (LSTM and a small Transformer) for the shared split.

Import this module before TensorFlow or Keras (it sets two environment variables that only work if TensorFlow is not loaded yet).
Everything is trained on windows of the last ``WINDOW`` months of the standardised logarithm of the series; a window never contains
the month it has to predict or anything after it.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")     # keep TensorFlow start-up messages quiet
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")    # avoid tiny run-to-run rounding differences

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from . import forecasting as fc

tf.get_logger().setLevel("ERROR")        # hide the "retracing" notices that Keras prints when we predict with many small models

WINDOW = 12                         # one year of history is the input of every neural model


def set_seed(seed=fc.SEED):
    """Fix every source of randomness we use, so reruns give the same numbers on a CPU."""
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


# ---------------------------------------------------------------- data preparation
class Representation:
    """How the series is shown to the network. Statistics come from the *training* months only.

    "level":  the standardised logarithm of the series; the label is the value of the month to predict.
    "yoy":    the year-over-year change of the logarithm, log(y_t) - log(y_(t-12)), standardised; the label is the change of the month to
              predict, and the forecast is the value of the same month last year times that change. This removes trend and yearly pattern
              before the network sees the data.
    """

    def __init__(self, kind, train_values):
        self.kind = kind
        logged = np.log(np.asarray(train_values, dtype="float64"))
        values = logged if kind == "level" else logged[12:] - logged[:-12]
        self.mean, self.std = float(values.mean()), float(values.std())
        self.first = WINDOW if kind == "level" else 2 * WINDOW        # first month that has a full window of features

    def features(self, values):
        """Scaled feature for every position (NaN where it is not defined yet)."""
        logged = np.log(np.asarray(values, dtype="float64"))
        if self.kind == "level":
            return (logged - self.mean) / self.std
        scaled = np.full_like(logged, np.nan)
        scaled[12:] = (logged[12:] - logged[:-12] - self.mean) / self.std
        return scaled

    def windows(self, values, targets):
        """For every target position t: the input is the features at t-12 ... t-1 (strictly before t), the label is the feature at t."""
        scaled = self.features(values)
        x = np.stack([scaled[t - WINDOW:t] for t in targets])[..., None].astype("float32")
        y = np.array([scaled[t] for t in targets], dtype="float32")
        return x, y

    def decode(self, prediction, values, t):
        """Turn a network output for month t into passengers, using only ``values`` before t."""
        if self.kind == "level":
            return float(np.exp(prediction * self.std + self.mean))
        return float(np.exp(np.log(values[t - 12]) + prediction * self.std + self.mean))


# ---------------------------------------------------------------- models
def build_lstm(units=16, window=WINDOW):
    return keras.Sequential([keras.layers.Input((window, 1)), keras.layers.LSTM(units),
                             keras.layers.Dense(1)], name="lstm")


def build_transformer(blocks=1, d_model=16, heads=2, window=WINDOW):
    """A small Transformer encoder: embed each month, add a learned position, attention + feed-forward blocks, read the last month.

    Returns (model, attention_model); the second returns the attention weights of the first block (for looking inside).
    """
    inputs = keras.Input((window, 1))
    x = keras.layers.Dense(d_model)(inputs)
    positions = keras.layers.Embedding(window, d_model)(keras.ops.arange(window))
    x = x + positions
    scores = None
    for block in range(blocks):
        attention = keras.layers.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads)
        attended, weights = attention(x, x, return_attention_scores=True)
        scores = weights if scores is None else scores
        x = keras.layers.LayerNormalization()(x + attended)
        hidden = keras.layers.Dense(2 * d_model, activation="relu")(x)
        x = keras.layers.LayerNormalization()(x + keras.layers.Dense(d_model)(hidden))
    outputs = keras.layers.Dense(1)(x[:, -1, :])
    return (keras.Model(inputs, outputs, name="transformer"),
            keras.Model(inputs, scores, name="transformer_attention"))


def build(kind, config, window=WINDOW):
    if kind == "lstm":
        return build_lstm(config["units"], window), None
    return build_transformer(config["blocks"], window=window)


# ---------------------------------------------------------------- training
def train(kind, config, series, train_end, val_end, seed, max_epochs=200, patience=20):
    """Train on the windows whose targets lie before ``train_end``; stop when the loss on the windows whose targets
    lie in [train_end, val_end) has not improved for ``patience`` epochs and keep the best weights.

    ``config`` holds the model size and the representation ("level" or "yoy"). Returns (model, best_epoch, representation).
    """
    set_seed(seed)
    rep = Representation(config["rep"], series.iloc[:train_end])
    values = series.to_numpy()
    x_train, y_train = rep.windows(values, range(rep.first, train_end))
    x_val, y_val = rep.windows(values, range(train_end, val_end))
    model, _ = build(kind, config)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=max_epochs, batch_size=16,
                        callbacks=[stop], verbose=0)
    return model, int(np.argmin(history.history["val_loss"])) + 1, rep


def train_final(kind, config, series, upto, seed, epochs):
    """Train on all windows whose targets lie before ``upto`` for a fixed number of epochs (no validation data left).

    Returns (model, attention_model, representation).
    """
    set_seed(seed)
    rep = Representation(config["rep"], series.iloc[:upto])
    x_train, y_train = rep.windows(series.to_numpy(), range(rep.first, upto))
    model, attention = build(kind, config)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(x_train, y_train, epochs=epochs, batch_size=16, verbose=0)
    return model, attention, rep


# ---------------------------------------------------------------- forecasting with a trained model
def one_step(model, rep, series, start, n):
    """Task A: for each of the ``n`` months from position ``start``, predict it from the true twelve months before it."""
    values = series.to_numpy()
    x, _ = rep.windows(values, range(start, start + n))
    output = model.predict(x, verbose=0)[:, 0]
    predicted = [rep.decode(o, values, t) for o, t in zip(output, range(start, start + n))]
    return pd.Series(predicted, index=series.index[start:start + n])


def recursive(model, rep, series, start, n):
    """Task B: predict ``n`` months from what is known before ``start``, feeding each prediction back as if it had been observed."""
    values = list(series.to_numpy()[:start])
    for t in range(start, start + n):
        scaled = rep.features(np.array(values))
        window = np.array(scaled[t - WINDOW:t], dtype="float32")[None, :, None]
        values.append(rep.decode(float(model.predict(window, verbose=0)[0, 0]), values, t))
    return pd.Series(values[start:], index=series.index[start:start + n])


def attention_by_position(attention_model, rep, series, start, n):
    """Average attention (over test windows and heads) that the *last* month of the window pays to each of the 12 months."""
    x, _ = rep.windows(series.to_numpy(), range(start, start + n))
    weights = attention_model.predict(x, verbose=0)          # (windows, heads, query, key)
    return weights[:, :, -1, :].mean(axis=(0, 1))
