"""Helpers shared by the notebooks: data, the shared split, metrics, forecasters and figures.

Every forecaster has the same shape: ``forecaster(history, horizon) -> numpy array of length horizon``, where ``history`` is the
pandas Series of everything that is known so far. A forecaster never sees anything after the end of ``history``.
"""

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
SEASON = 12                       # monthly data with a yearly pattern
N_TRAIN, N_VAL, N_TEST = 96, 24, 24
DATA = Path(__file__).resolve().parents[1] / "data" / "airline-passengers.csv"
ASSETS = Path(__file__).resolve().parents[1] / "assets"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="IProgress not found")


def quiet_prophet_logs():
    """Prophet and its Stan back end print two lines for every fit; keep only real problems (call after importing prophet)."""
    import cmdstanpy
    cmdstanpy.disable_logging()
    logging.getLogger("prophet").setLevel(logging.ERROR)


def save_figure(fig, name):
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / name, dpi=110, bbox_inches="tight")


# ---------------------------------------------------------------- data and the shared split
def load_series(path=DATA):
    """The 144 monthly passenger numbers (thousands) as a Series with a month-start date index."""
    frame = pd.read_csv(path)
    index = pd.DatetimeIndex(pd.to_datetime(frame["Month"]), freq="MS")
    return pd.Series(frame["Passengers"].to_numpy(dtype="float64"), index=index, name="passengers")


def split(series):
    """Train = first 96 months, validation = next 24, test = last 24, always in time order (no shuffling)."""
    train = series.iloc[:N_TRAIN]
    validation = series.iloc[N_TRAIN:N_TRAIN + N_VAL]
    test = series.iloc[N_TRAIN + N_VAL:]
    return train, validation, test


# ---------------------------------------------------------------- metrics
def mae(actual, predicted):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def mape(actual, predicted):
    """Mean absolute percentage error in percent."""
    actual = np.asarray(actual)
    return float(np.mean(np.abs((actual - np.asarray(predicted)) / actual)) * 100)


def scores(actual, predicted):
    return {"MAE": mae(actual, predicted), "RMSE": rmse(actual, predicted), "MAPE %": mape(actual, predicted)}


# ---------------------------------------------------------------- forecasters: (history, horizon) -> array
def naive(history, horizon):
    """Tomorrow looks like today: repeat the last known value."""
    return np.repeat(history.iloc[-1], horizon)


def seasonal_naive(history, horizon):
    """Repeat what happened in the same month last year."""
    last_year = history.iloc[-SEASON:].to_numpy()
    return np.array([last_year[i % SEASON] for i in range(horizon)])


def sarima(order, seasonal_order, use_log=True):
    """A seasonal ARIMA model (Box and Jenkins), optionally fitted to the logarithm of the series."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    def forecaster(history, horizon):
        y = np.log(history) if use_log else history
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SARIMAX(y.to_numpy(), order=order, seasonal_order=(*seasonal_order, SEASON),
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            result = fit.forecast(horizon)
        return np.exp(result) if use_log else np.asarray(result)

    return forecaster


def holt_winters(trend="add", seasonal="mul", damped=False):
    """Exponential smoothing with trend and yearly seasonality (Holt and Winters)."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    def forecaster(history, horizon):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(history.to_numpy(), trend=trend, damped_trend=damped, seasonal=seasonal,
                                       seasonal_periods=SEASON, initialization_method="estimated").fit()
            return np.asarray(fit.forecast(horizon))

    return forecaster


def prophet(seasonality_mode="multiplicative"):
    """Prophet (Taylor and Letham, 2018): a trend plus a yearly seasonal pattern, fitted by optimisation."""
    logging.getLogger("prophet.plot").setLevel(logging.CRITICAL)           # hides a harmless "plotly" notice
    from prophet import Prophet
    quiet_prophet_logs()

    def forecaster(history, horizon):
        frame = pd.DataFrame({"ds": history.index, "y": history.to_numpy()})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                        seasonality_mode=seasonality_mode)
        model.fit(frame)
        future = model.make_future_dataframe(periods=horizon, freq="MS")
        return model.predict(future)["yhat"].to_numpy()[-horizon:]

    return forecaster


# ---------------------------------------------------------------- the two evaluation protocols
def walk_forward(forecaster, series, start, n):
    """One month ahead, again and again: for each of the ``n`` months from position ``start`` the forecaster is given
    all *true* values before that month (and nothing else) and predicts that single month."""
    predictions = [forecaster(series.iloc[:t], 1)[0] for t in range(start, start + n)]
    return pd.Series(predictions, index=series.index[start:start + n])


def multi_step(forecaster, series, start, n):
    """24 months ahead at once: the forecaster sees the values before position ``start`` and predicts the next ``n`` months."""
    return pd.Series(forecaster(series.iloc[:start], n), index=series.index[start:start + n])


# ---------------------------------------------------------------- figures
def plot_series(series):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(series.index, series.values, color="tab:blue")
    ax.set_ylabel("passengers (thousands)"); ax.grid(alpha=.3)
    return fig


def plot_split(series):
    train, validation, test = split(series)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for part, color, label in [(train, "tab:blue", "train (96 months)"), (validation, "tab:orange", "validation (24)"),
                               (test, "tab:green", "test (24)")]:
        ax.plot(part.index, part.values, color=color, label=label)
        ax.axvspan(part.index[0], part.index[-1], color=color, alpha=.08)
    ax.set_ylabel("passengers (thousands)"); ax.legend(loc="upper left"); ax.grid(alpha=.3)
    return fig


def plot_decomposition(series):
    """Trend, yearly pattern and what is left over (STL decomposition of the logarithm)."""
    from statsmodels.tsa.seasonal import STL
    result = STL(np.log(series), period=SEASON, robust=True).fit()
    fig, axes = plt.subplots(3, 1, figsize=(9, 5.2), sharex=True)
    for ax, values, title in zip(axes, [result.trend, result.seasonal, result.resid],
                                 ["trend (log of passengers)", "yearly pattern", "what is left"]):
        ax.plot(series.index, values, color="tab:purple"); ax.set_title(title, fontsize=9, loc="left"); ax.grid(alpha=.3)
    return fig


def plot_forecasts(actual, predictions, title):
    """The true test values and the forecasts of several models."""
    fig, ax = plt.subplots(figsize=(9.5, 4))
    ax.plot(actual.index, actual.values, color="black", lw=2.5, label="actual")
    for k, (name, predicted) in enumerate(predictions.items()):
        ax.plot(predicted.index, predicted.values, marker="o", ms=3, lw=1.3, label=name, color=f"C{k}")
    ax.set_ylabel("passengers (thousands)"); ax.set_title(title, fontsize=10); ax.grid(alpha=.3)
    ax.legend(fontsize=8, ncol=2)
    return fig


def plot_bars(names, values, xlabel):
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(names) + 1.2))
    positions = np.arange(len(names))[::-1]
    ax.barh(positions, values, color="tab:blue", alpha=.85)
    for y, v in zip(positions, values):
        ax.text(v + max(values) * 0.01, y, f"{v:.1f}", va="center", fontsize=9)
    ax.set_yticks(positions); ax.set_yticklabels(names); ax.set_xlabel(xlabel); ax.grid(axis="x", alpha=.3)
    return fig


def plot_representations(series):
    """The same series as the network sees it: the standardised log (levels) and the year-over-year change of the log."""
    logged = np.log(series)
    change = (logged - logged.shift(SEASON)).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].plot(series.index, logged.values, color="tab:blue"); axes[0].set_title("levels: log of passengers (rises, with a yearly pattern)", fontsize=9)
    axes[1].plot(change.index, change.values, color="tab:green"); axes[1].axhline(0, color="gray", lw=.8)
    axes[1].set_title("year-over-year change of the log (no yearly pattern, no steady trend)", fontsize=9)
    for ax in axes:
        ax.grid(alpha=.3)
    return fig


def plot_attention(weights):
    """Average attention of the last month of a window on each of the twelve months before the forecast."""
    fig, ax = plt.subplots(figsize=(7, 3))
    labels = [f"t-{k}" for k in range(len(weights), 0, -1)]
    ax.bar(labels, weights, color="tab:purple")
    ax.set_ylabel("attention weight"); ax.set_xlabel("month of the window (t = the month to predict)")
    ax.set_title("where the Transformer looks (average over test windows and heads)", fontsize=9)
    return fig


def plot_conclusion(rows):
    """Test MAE of every model for both tasks. ``rows``: list of (name, mae_A, low_A, high_A, mae_B, low_B, high_B); the
    thin line spans the smallest to the largest value over seeds (equal to the bar for models without seeds)."""
    names = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(rows) + 1.6), sharey=True)
    positions = np.arange(len(rows))[::-1]
    for ax, (k, title) in zip(axes, [(1, "task A: one month ahead"), (4, "task B: 24 months ahead")]):
        values = [r[k] for r in rows]
        colors = ["tab:gray" if "naive" in n else ("tab:blue" if not (n.startswith("LSTM") or n.startswith("Transformer")) else "tab:orange")
                  for n in names]
        ax.barh(positions, values, color=colors, alpha=.85)
        ax.hlines(positions, [r[k + 1] for r in rows], [r[k + 2] for r in rows], color="black", lw=1.5)
        for y, v, high in zip(positions, values, [r[k + 2] for r in rows]):
            ax.text(max(v, high) + max(values) * 0.012, y, f"{v:.1f}", va="center", fontsize=8)
        ax.set_title(title + "  (test MAE, thousands; lower is better)", fontsize=9); ax.grid(axis="x", alpha=.3)
    axes[0].set_yticks(positions); axes[0].set_yticklabels(names)
    return fig
