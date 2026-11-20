"""
features.py
-----------
Loads the hourly electricity data, fixes small problems, and builds the
features (inputs) the model uses.

Used by eda.py, train.py and app.py so everything is prepared the same way.
"""

from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "PJME_hourly.csv"
TARGET = "demand_mw"

# Time-of-day / calendar features (known in advance for any future hour).
CALENDAR_FEATURES = [
    "hour",
    "dayofweek",
    "is_weekend",
    "is_holiday",
    "month",
    "quarter",
    "dayofyear",
    "weekofyear",
]

# Past-demand features. Every one of them looks at least 24 hours back,
# so the model can forecast a full day ahead without "seeing the future".
LAG_FEATURES = [
    "lag_24h",          # same hour yesterday
    "lag_48h",          # same hour two days ago
    "lag_168h",         # same hour one week ago
    "rolling_mean_24h",  # average of the 24 hours that end 24 hours ago
    "rolling_mean_168h",  # average of the 7 days that end 24 hours ago
]

ALL_FEATURES = CALENDAR_FEATURES + LAG_FEATURES


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Read the CSV and return one row per hour with no gaps.

    Two small problems in the raw data come from clocks changing for
    daylight saving time (summer time):
      * In autumn one hour happens twice  -> 4 repeated timestamps.
        We take the average of the two values.
      * In spring one hour is skipped     -> 30 missing hours.
        We fill each one with the straight line between its neighbours.
    """
    df = pd.read_csv(path, parse_dates=["Datetime"])
    df = df.rename(columns={"Datetime": "datetime", "PJME_MW": TARGET})

    # Average the repeated hours and sort by time.
    s = df.groupby("datetime")[TARGET].mean().sort_index()

    # Make a complete hourly timeline and fill the few missing hours.
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="h")
    s = s.reindex(full_index).interpolate(method="linear")
    s.index.name = "datetime"
    return s.to_frame()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and lag features. Input must be hourly with no gaps."""
    out = df.copy()
    idx = out.index

    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek          # Monday=0 ... Sunday=6
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["month"] = idx.month
    out["quarter"] = idx.quarter
    out["dayofyear"] = idx.dayofyear
    out["weekofyear"] = idx.isocalendar().week.astype(int).values

    holidays = USFederalHolidayCalendar().holidays(start=idx.min(), end=idx.max())
    out["is_holiday"] = idx.normalize().isin(holidays).astype(int)

    y = out[TARGET]
    out["lag_24h"] = y.shift(24)
    out["lag_48h"] = y.shift(48)
    out["lag_168h"] = y.shift(168)
    out["rolling_mean_24h"] = y.shift(24).rolling(24).mean()
    out["rolling_mean_168h"] = y.shift(24).rolling(168).mean()
    return out


def load_with_features() -> pd.DataFrame:
    """Load data, add features, drop the first rows that have no lag values."""
    return add_features(load_data()).dropna()


# Time-based split: never shuffle time series data.
TRAIN_END = "2015-12-31 23:00"
VAL_START, VAL_END = "2016-01-01 00:00", "2016-12-31 23:00"
TEST_START = "2017-01-01 00:00"


def split(df: pd.DataFrame):
    train = df.loc[:TRAIN_END]
    val = df.loc[VAL_START:VAL_END]
    test = df.loc[TEST_START:]
    return train, val, test
