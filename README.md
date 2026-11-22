# Forecasting airline passengers: from a broken LSTM to a fair comparison

> **What this repository shows.** In 2023 I tried to forecast the monthly number of airline passengers with an LSTM; the notebook stopped with an error and never produced a forecast.
> Here the same problem is redone as a **fair comparison on one shared time-ordered split**: simple baselines, ARIMA, exponential smoothing and Prophet (notebook 01, ready),
> then an LSTM and a small Transformer (notebook 02, planned), with honest metrics.

[![Open notebook 01 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maryammahmoudi1993/LSTM_passenges_prediction/blob/main/notebooks/01_shared_split_and_classical_baselines.ipynb)

> **Status: in progress.** Notebook 01 is ready. Notebook 02 (LSTM and small Transformer on the same split) is planned.

## Who this is for

- **For:** anyone who wants to see how to compare forecasting models fairly, and what a modest, honest "growth" story of a learning project looks like.
- **Prerequisites:** basic Python and pandas. **Time:** about 10 minutes for notebook 01 (it runs in a few minutes on a laptop CPU).

## The setup in one picture

```mermaid
flowchart LR
    S["144 months<br/>1949-1960"] --> T["train<br/>96 months"]
    S --> V["validation<br/>24 months<br/>(choose settings)"]
    S --> E["test<br/>24 months<br/>(used once)"]
    T --> M["naive, seasonal naive<br/>SARIMA, Holt-Winters, Prophet<br/>(LSTM, Transformer next)"]
    V --> M
    M --> R["task A: one month ahead<br/>task B: 24 months ahead"]
    E --> R
```

<p align="center">
  <img src="assets/01_forecasts_task_b.png" alt="Actual passengers and the forecasts of four models for the 24 test months, predicted 24 months ahead" width="760"><br>
  <em>Task B: forecasts made at the end of 1958 for the 24 months of 1959 and 1960. All classical models underestimate the 1960 summer peak.</em>
</p>

## How to run

**In the browser:** click the Colab badge (Colab needs `pip install prophet` first; the first cell downloads this repository).

**On your computer** (Python 3.13 was tested):

```bash
git clone https://github.com/maryammahmoudi1993/LSTM_passenges_prediction.git
cd LSTM_passenges_prediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name python3
jupyter lab notebooks/
```

If `jupyter` is not found, run `pip install jupyterlab` first (it is not needed to run the tests). Check that everything works (about a minute):

```bash
python -m pytest -q
```

## Contents

| # | Notebook | What it does | Status |
|---|---|---|---|
| 01 | [One fair test for every model](notebooks/01_shared_split_and_classical_baselines.ipynb) | the data, the shared split and its rules, baselines, SARIMA, Holt-Winters, Prophet, test results, Prophet's components | ready |
| 02 | LSTM and a small Transformer on the same split | planned | |

Other folders: [`src/`](src) shared code, [`tests/`](tests) smoke tests and a test that runs every notebook, [`data/`](data) the series (2 KB) and its notes,
[`legacy/`](legacy) the original 2023 files (untouched) and [`assets/`](assets) figures with [credits](assets/CREDITS.md).

## Results so far

Test months: January 1959 to December 1960 (24 months), used once. All numbers come from running notebook 01 (the classical models are deterministic). MAE and RMSE are in thousands of passengers.

| Task | Model | MAE | RMSE | MAPE |
|---|---|---|---|---|
| A: one month ahead (walk-forward) | naive | 44.2 | 51.8 | 9.7% |
| | seasonal naive | 47.6 | 50.0 | 10.5% |
| | SARIMA(1,1,0)(0,1,1) on the log | 12.5 | 15.8 | 2.8% |
| | Holt-Winters (damped trend, multiplicative) | 12.2 | 16.0 | 2.7% |
| | Prophet (multiplicative) | 17.6 | 21.9 | 3.8% |
| B: 24 months ahead | naive | 115.2 | 137.3 | 23.6% |
| | seasonal naive | 71.2 | 77.0 | 15.5% |
| | SARIMA(1,1,0)(0,1,1) on the log | 42.0 | 45.7 | 9.1% |
| | Holt-Winters (trend, multiplicative) | 29.0 | 32.5 | 6.4% |
| | Prophet (multiplicative) | 25.5 | 30.6 | 5.4% |

In task A, SARIMA and Holt-Winters are practically tied and 3.5 to 4 times better than the baselines. In task B all errors are two to three times larger and the ranking changes (Prophet first). With 24 test months, small gaps mean little.
The settings of every model were chosen on the validation months only (the notebook lists all candidates, including one that failed numerically).

## Then versus now

| | **Then (January 2023, in [`legacy/`](legacy))** | **Now** |
|---|---|---|
| Result | the training cell stops with a `ValueError`; no forecast was ever produced | forecasts and a results table for every model on the same test months |
| Input | a model built for 143 features per step (143 is the length of the series) | one value per month |
| Target | the target of each window is the last value inside the window (leakage; checked with a small example) | a forecaster only sees data before the month it predicts, enforced and tested in code |
| Data | first month dropped by `iloc[1:]` | all 144 months |
| Baseline | none | naive and seasonal naive |
| Evaluation | no metric in the original units | MAE, RMSE, MAPE; validation for choices, test used once; two tasks |

What I would do differently, and did: start with baselines, write the split and the rules down before training anything, and test that the code cannot peek into the future.
See [`legacy/ABOUT.md`](legacy/ABOUT.md) for what I notice in the original notebook today.

## Limitations

- **One series, 144 points, 24 test months:** the results do not say how the models behave on other series, and small gaps between models are not meaningful.
- The test months (the highest values of the series) lie outside the range of the training data, which favours models that extrapolate a trend.
- Only a few settings per model family were tried; the notebook lists them all.
- No prediction intervals or statistical tests are reported.
- The data licence could not be verified (see [`data/README.md`](data/README.md)).

## Further reading

- Hyndman and Athanasopoulos, *Forecasting: Principles and Practice* (free online book): <https://otexts.com/fpp3/>.
- Box and Jenkins, *Time Series Analysis: Forecasting and Control* (the source of the airline series and of the SARIMA family).
- Taylor and Letham (2018), *Forecasting at scale*, The American Statistician 72(1), 37-45 (Prophet).
- Documentation of [statsmodels](https://www.statsmodels.org/) (SARIMAX, Holt-Winters, STL) and [Prophet](https://facebook.github.io/prophet/).

## License

MIT, see [`LICENSE`](LICENSE). The data file has its own provenance, see [`data/README.md`](data/README.md).
