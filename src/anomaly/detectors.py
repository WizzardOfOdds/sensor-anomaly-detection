"""Anomaly detectors, from a fixed threshold to an isolation forest.

All detectors share one interface: `fit` on a stretch of healthy data, then
`flags` returns a boolean array marking suspicious readings. They see the
data only up to the current reading, so everything here would also work on
a live stream.

Each detector can work on the raw temperature or on the *residual*, the
temperature minus what a simple model predicts from the motor current. The
residual strips out normal load swings, so what is left is mostly noise and
faults.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression


class ExpectedTemperature:
    """Predicts temperature from current.

    The motor heats up with a delay, so the features are the current smoothed
    over several time spans, and the regression picks the right mix.
    """

    spans = (1, 20, 40, 80)   # minutes

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        cur = df["current"]
        return np.column_stack([cur.ewm(span=s).mean() for s in self.spans])

    def fit(self, df: pd.DataFrame) -> ExpectedTemperature:
        self.model = LinearRegression().fit(self._features(df), df["temperature"])
        return self

    def residual(self, df: pd.DataFrame) -> pd.Series:
        return df["temperature"] - self.model.predict(self._features(df))


class Threshold:
    """Flag anything more than k standard deviations above the healthy mean."""

    def __init__(self, k: float = 3.0):
        self.k = k

    def fit(self, x: pd.Series):
        self.limit = x.mean() + self.k * x.std()
        return self

    def flags(self, x: pd.Series) -> np.ndarray:
        return (x > self.limit).to_numpy()


class RollingZ:
    """Compare each reading to the last `window` readings.

    Reacts quickly to sudden jumps but gets used to slow drift. A wearing
    bearing looks normal once the window has filled with already-warm readings.
    """

    def __init__(self, window: int = 120, k: float = 4.0):
        self.window, self.k = window, k

    def fit(self, x: pd.Series):
        return self

    def flags(self, x: pd.Series) -> np.ndarray:
        past = x.shift(1).rolling(self.window, min_periods=self.window)
        z = (x - past.mean()) / past.std()
        return (z > self.k).fillna(False).to_numpy()


class EWMAChart:
    """EWMA control chart, standard in manufacturing quality control.

    Averaging with weight `lam` smooths noise and glitches, so a small but
    persistent shift in the mean shows up after a few dozen readings.

    The textbook limit, sd * sqrt(lam / (2 - lam)), assumes independent
    readings. Sensor data is autocorrelated, so the limit would be far too
    tight. Instead, the limit comes from how much the EWMA itself moved
    during the healthy training data. Readings are capped at 4 sd first,
    otherwise one glitch keeps the average raised for an hour.
    """

    def __init__(self, lam: float = 0.05, k: float = 4.0):
        self.lam, self.k = lam, k

    def _ewma(self, x: pd.Series, start: float) -> pd.Series:
        x = x.clip(self.mu - self.cap, self.mu + self.cap)
        return pd.concat([pd.Series([start]), x]).ewm(alpha=self.lam, adjust=False).mean().iloc[1:]

    def fit(self, x: pd.Series):
        self.mu, self.cap = x.mean(), 4 * x.std()
        z = self._ewma(x, self.mu)
        self.limit = self.mu + self.k * z.std()
        return self

    def flags(self, x: pd.Series) -> np.ndarray:
        return (self._ewma(x, self.mu) > self.limit).to_numpy()


class CUSUM:
    """One-sided cumulative sum. It adds up small excesses over the healthy mean.

    `drift` is the allowance in standard deviations: excesses smaller than
    that are forgiven. `h` is the alarm level.
    """

    def __init__(self, drift: float = 0.5, h: float = 25.0):
        self.drift, self.h = drift, h

    def fit(self, x: pd.Series):
        self.mu, self.sd = x.mean(), x.std()
        return self

    def flags(self, x: pd.Series) -> np.ndarray:
        # Clip single readings so that one glitch cannot fill the sum on its own.
        z = np.clip((x.to_numpy() - self.mu) / self.sd, -4, 4)
        s, out = 0.0, np.zeros(len(z), dtype=bool)
        for i, zi in enumerate(z):
            s = max(0.0, s + zi - self.drift)
            out[i] = s > self.h
        return out


class Forest:
    """Isolation forest on a few rolling features of the signal.

    Features: the value, its 30-minute mean and its 30-minute standard
    deviation. That lets it catch the noisier signal of a worn bearing, not
    only higher readings.
    """

    def __init__(self, contamination: float = 0.002, seed: int = 0):
        self.contamination, self.seed = contamination, seed

    @staticmethod
    def _features(x: pd.Series) -> np.ndarray:
        r = x.rolling(30, min_periods=1)
        return np.column_stack([x, r.mean(), r.std().fillna(0)])

    def fit(self, x: pd.Series):
        self.model = IsolationForest(contamination=self.contamination, random_state=self.seed)
        self.model.fit(self._features(x))
        return self

    def flags(self, x: pd.Series) -> np.ndarray:
        return self.model.predict(self._features(x)) == -1


def debounce(flags: np.ndarray, n: int = 3) -> np.ndarray:
    """Raise an alarm only after n flagged readings in a row. Filters out glitches."""
    run = np.zeros(len(flags), dtype=int)
    count = 0
    for i, f in enumerate(flags):
        count = count + 1 if f else 0
        run[i] = count
    return run >= n


DETECTORS = {
    "threshold": Threshold,
    "rolling z": RollingZ,
    "EWMA chart": EWMAChart,
    "CUSUM": CUSUM,
    "isolation forest": Forest,
}
