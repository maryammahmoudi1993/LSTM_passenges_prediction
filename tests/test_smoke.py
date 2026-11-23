"""Quick checks of the helpers; the slow models are only tried on a short series."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import forecasting as fc  # noqa: E402


def test_data_and_split_are_in_time_order_without_overlap():
    series = fc.load_series()
    assert len(series) == 144 and series.index[0] == pd.Timestamp("1949-01-01") and series.iloc[0] == 112
    assert series.index.freqstr == "MS" and not series.isna().any()
    train, validation, test = fc.split(series)
    assert (len(train), len(validation), len(test)) == (96, 24, 24)
    assert train.index[-1] < validation.index[0] and validation.index[-1] < test.index[0]
    assert pd.concat([train, validation, test]).equals(series)


def test_metrics():
    assert fc.mae([1, 2, 3], [2, 2, 5]) == 1.0
    assert abs(fc.rmse([0, 0], [3, 4]) - np.sqrt(12.5)) < 1e-12
    assert abs(fc.mape([100, 200], [110, 180]) - 10.0) < 1e-12


def test_naive_forecasters_and_protocols_never_peek():
    series = fc.load_series()
    history = series.iloc[:100]
    assert list(fc.naive(history, 3)) == [history.iloc[-1]] * 3
    assert list(fc.seasonal_naive(history, 14)) == list(history.iloc[-12:]) + list(history.iloc[-12:][:2])
    seen = []

    def spy(h, horizon):
        seen.append(h.index[-1])
        return np.zeros(horizon)

    fc.walk_forward(spy, series, 120, 3)
    assert seen == [series.index[119], series.index[120], series.index[121]]      # only the past is given
    assert len(fc.multi_step(fc.seasonal_naive, series, 120, 24)) == 24


def test_classical_models_forecast_a_seasonal_series():
    series = fc.load_series().iloc[:72]
    for forecaster in (fc.sarima((0, 1, 1), (0, 1, 1)), fc.holt_winters("add", "mul"), fc.prophet("multiplicative")):
        forecast = forecaster(series, 12)
        assert forecast.shape == (12,) and np.isfinite(forecast).all()
        assert fc.mape(series.iloc[-12:], forecast) < 60          # sane order of magnitude (not a quality claim)


def test_neural_windows_never_contain_the_target_or_the_future():
    from src import neural as nn
    series = fc.load_series()
    values = series.to_numpy()
    level = nn.Representation("level", series.iloc[:96])
    scaled = level.features(values)
    x, y = level.windows(values, [96, 100])
    assert x.shape == (2, nn.WINDOW, 1) and y.shape == (2,)
    assert np.allclose(x[0, :, 0], scaled[84:96]) and np.isclose(y[0], scaled[96])      # window ends right before its target
    assert np.isclose(level.decode(scaled[100], values, 100), values[100])
    yoy = nn.Representation("yoy", series.iloc[:96])
    xy, yy = yoy.windows(values, [96])
    assert np.isfinite(xy).all() and np.isfinite(yy).all() and yoy.first == 24
    assert np.isclose(yoy.decode(yoy.features(values)[100], values, 100), values[100])  # decoding the true change gives the true value


def test_changing_the_future_does_not_change_a_forecast():
    from src import neural as nn
    series = fc.load_series()
    model, _, rep = nn.train("lstm", {"units": 4, "rep": "yoy"}, series, 96, 120, seed=1, max_epochs=2, patience=2)
    changed = series.copy()
    changed.iloc[120:] = changed.iloc[120:] * 3                       # a different future
    first = nn.recursive(model, rep, series, 120, 4).to_numpy()
    second = nn.recursive(model, rep, changed, 120, 4).to_numpy()
    assert np.allclose(first, second)                                  # the forecast only depends on what is before month 120


def test_neural_models_train_and_forecast_on_a_short_run():
    from src import neural as nn
    series = fc.load_series()
    for rep_name in ("level", "yoy"):
        model, best_epoch, rep = nn.train("lstm", {"units": 4, "rep": rep_name}, series, 96, 120, seed=1, max_epochs=3, patience=3)
        assert 1 <= best_epoch <= 3
        one = nn.one_step(model, rep, series, 120, 5)
        many = nn.recursive(model, rep, series, 120, 5)
        assert len(one) == 5 and len(many) == 5 and np.isfinite(one).all() and np.isfinite(many).all()
    transformer, attention, rep = nn.train_final("transformer", {"blocks": 1, "rep": "level"}, series, 120, seed=1, epochs=2)
    weights = nn.attention_by_position(attention, rep, series, 120, 5)
    assert weights.shape == (nn.WINDOW,) and abs(weights.sum() - 1.0) < 1e-4
    assert np.isfinite(nn.one_step(transformer, rep, series, 120, 3)).all()
