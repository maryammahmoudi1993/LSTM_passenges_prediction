# Legacy files

These are my original files from the first version of this project (January 2023), kept unchanged:
`time_series_passengers.ipynb` (the notebook with its saved output), `airline-passengers.txt` (the data) and the first `README.md`.

What I notice today, from reading the notebook and its saved output:

- **The notebook ends with an error and never produced a prediction.** The training cell stops with a `ValueError`: the model was built with the
  input shape `(4, 143)`, where 143 is the *number of months in the series* (printed earlier as `passengers.shape[-1]`), but the data has one value per time step.
  The saved output shows no loss, no accuracy and no forecast.
- **The target leaks into the input.** With `sequence_length = 4` and `delay = 3`, the target of every window is `passengers[delay:]`, i.e. the *last value inside the window*.
  (I checked this with a ten-number example: the windows `[0,1,2,3]`, `[1,2,3,4]`, `[2,3,4,5]` get the targets 3, 4, 5.)
  A model could therefore just copy its last input, and even a working version of this notebook would have looked far better than it is.
- **The first month is dropped.** `raw_data.iloc[1:,1]` skips the first data row (January 1949, 112 thousand passengers): 143 of the 144 months are used.
- **The windows are shuffled** (`shuffle=True`) for validation and test as well; harmless for a model without state, but it hides the time order when one looks at the results.
- The split is a plain 67% / 15% / 18% cut in time order (good), but there is no baseline to compare with, and the README does not say how the series behaves
  (a rising trend with a seasonal swing that grows over the years), which a model has to handle.
- No random seed, no metric in the original units (the values were standardised with the training mean and standard deviation, which is fine, but the notebook never converts back).

The new notebooks in `../notebooks/` rebuild the project as a fair comparison on one shared split: simple baselines, classical models (ARIMA, exponential smoothing, Prophet)
and, in part 2, an LSTM and a small Transformer.
