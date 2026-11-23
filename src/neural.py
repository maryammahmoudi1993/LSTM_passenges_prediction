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

WINDOW = 12                         # one year of history is the input of every neural model


def set_seed(seed=fc.SEED):
    """Fix every source of randomness we use, so reruns give the same numbers on a CPU."""
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


# ---------------------------------------------------------------- data preparation
class Scaler:
    """Logarithm, then subtract the mean and divide by the standard deviation of the *training* months only."""

    def __init__(self, train_values):
        logged = np.log(np.asarray(train_values, dtype="float64"))
        self.mean, self.std = float(logged.mean()), float(logged.std())

    def transform(self, values):
        return (np.log(np.asarray(values, dtype="float64")) - self.mean) / self.std

    def inverse(self, values):
        return np.exp(np.asarray(values, dtype="float64") * self.std + self.mean)


def make_windows(scaled, targets, window=WINDOW):
    """For every target position t: the input is scaled[t-window:t] (strictly before t) and the label is scaled[t]."""
    x = np.stack([scaled[t - window:t] for t in targets])[..., None].astype("float32")
    y = np.array([scaled[t] for t in targets], dtype="float32")
    return x, y


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
def train(kind, config, series, train_end, val_end, seed, max_epochs=300, patience=30):
    """Train on the windows whose targets lie before ``train_end``; stop when the loss on the windows whose targets
    lie in [train_end, val_end) has not improved for ``patience`` epochs and keep the best weights.

    Returns (model, best_epoch, scaler). The scaler is fitted on the training months only.
    """
    set_seed(seed)
    scaler = Scaler(series.iloc[:train_end])
    scaled = scaler.transform(series.to_numpy())
    x_train, y_train = make_windows(scaled, range(WINDOW, train_end))
    x_val, y_val = make_windows(scaled, range(train_end, val_end))
    model, _ = build(kind, config)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=max_epochs, batch_size=16,
                        callbacks=[stop], verbose=0)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    return model, best_epoch, scaler


def train_final(kind, config, series, upto, seed, epochs):
    """Train on all windows whose targets lie before ``upto`` for a fixed number of epochs (no validation data left).

    Returns (model, attention_model, scaler).
    """
    set_seed(seed)
    scaler = Scaler(series.iloc[:upto])
    scaled = scaler.transform(series.to_numpy())
    x_train, y_train = make_windows(scaled, range(WINDOW, upto))
    model, attention = build(kind, config)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(x_train, y_train, epochs=epochs, batch_size=16, verbose=0)
    return model, attention, scaler


# ---------------------------------------------------------------- forecasting with a trained model
def one_step(model, scaler, series, start, n):
    """Task A: for each of the ``n`` months from position ``start``, predict it from the true twelve months before it."""
    scaled = scaler.transform(series.to_numpy())
    x, _ = make_windows(scaled, range(start, start + n))
    predicted = scaler.inverse(model.predict(x, verbose=0)[:, 0])
    return pd.Series(predicted, index=series.index[start:start + n])


def recursive(model, scaler, series, start, n):
    """Task B: from the twelve months before ``start``, predict ``n`` months by feeding each prediction back as the next input."""
    scaled = list(scaler.transform(series.to_numpy()[:start]))
    out = []
    for _ in range(n):
        window = np.array(scaled[-WINDOW:], dtype="float32")[None, :, None]
        step = float(model.predict(window, verbose=0)[0, 0])
        scaled.append(step)
        out.append(step)
    return pd.Series(scaler.inverse(out), index=series.index[start:start + n])


def attention_by_position(attention_model, scaler, series, start, n):
    """Average attention (over test windows and heads) that the *last* month of the window pays to each of the 12 months."""
    scaled = scaler.transform(series.to_numpy())
    x, _ = make_windows(scaled, range(start, start + n))
    weights = attention_model.predict(x, verbose=0)          # (windows, heads, query, key)
    return weights[:, :, -1, :].mean(axis=(0, 1))
