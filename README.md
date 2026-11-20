# Electricity Demand Forecasting

A machine learning project that forecasts **hourly electricity demand one day ahead** for the
PJM East region (USA). It compares two simple baselines with two XGBoost models and shows the
results in an interactive Streamlit dashboard.

## Results (test set: 1 Jan 2017 to 3 Aug 2018, 13,897 hours never seen in training)

| Model | MAE (MW) | RMSE (MW) | MAPE |
|---|---|---|---|
| Baseline: same hour yesterday | 2,300 | 3,134 | 7.32% |
| Baseline: same hour last week | 3,511 | 4,794 | 10.99% |
| XGBoost: calendar only | 3,127 | 3,958 | 10.16% |
| **XGBoost: calendar + past demand** | **1,896** | **2,525** | **6.03%** |

The best model has about **18% lower error** than the "same hour yesterday" baseline.
The calendar-only model is worse than the simple baseline, which shows that recent demand
(lag features) carries most of the useful information.

## Dataset

[Hourly Energy Consumption (PJM)](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption),
file `PJME_hourly.csv`: hourly demand in megawatts (MW) for PJM East, Jan 2002 to Aug 2018
(145,366 rows). A copy is included in `data/PJME_hourly.csv`.

## Project structure

```
electricity-demand-forecasting/
├── app.py                   # Streamlit dashboard
├── docs/                    # detailed project guide (PDF)
├── requirements.txt
├── data/
│   └── PJME_hourly.csv
├── src/
│   ├── features.py          # loading, cleaning, features, time split
│   ├── eda.py               # charts that explore the data
│   ├── train.py             # baselines + XGBoost, scores, charts
│   └── plot_style.py        # chart colors
├── models/                  # saved XGBoost model (created by train.py)
└── reports/
    ├── metrics.json
    ├── model_comparison.csv
    ├── feature_importance.csv
    ├── test_predictions.csv # actual vs forecast for every test hour
    └── figures/             # all charts
```

## How to run

```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac / Linux

# 2. install libraries
pip install -r requirements.txt

# 3. explore the data (saves charts in reports/figures)
python src/eda.py

# 4. train and compare the models (under a minute)
python src/train.py

# 5. open the dashboard
streamlit run app.py
```

## Method in short

1. **Cleaning:** the raw data has 4 repeated hours and 30 missing hours, all caused by
   daylight saving clock changes. Repeated hours are averaged; missing hours are filled
   by straight-line interpolation.
2. **Calendar features:** hour, day of week, weekend flag, US federal holiday flag, month,
   quarter, day of year, week of year.
3. **Lag features** (all at least 24 hours old, so the forecast is a true day-ahead forecast):
   demand 24 h, 48 h and 168 h (1 week) earlier, plus 24 h and 7-day rolling averages that
   end 24 hours before the forecast hour.
4. **Time-based split:** train 2002-2015, validation 2016, test 2017-Aug 2018. No shuffling.
5. **Models:** two baselines (copy yesterday, copy last week) and XGBoost with early stopping
   on the validation year.
6. **Metrics:** MAE, RMSE and MAPE on the test set.

## Key findings

- Demand is lowest around 4-5 am and highest around 6-7 pm on weekdays.
- Summer (July: ~37,900 MW average) and winter (January: ~34,300 MW) are the peak seasons;
  April and October are the quietest.
- Weekends use about 10% less electricity than weekdays.
- The most important feature is demand at the same hour yesterday (about 62% of importance),
  followed by the same hour last week (about 18%).
- Errors are largest in the afternoon (3-6 pm), probably because heat or cold has the
  biggest effect on demand then, and the model has no weather data.

## Limitations and next steps

- No weather data: temperature is the biggest driver of demand, so adding a temperature
  forecast would likely cut the error a lot.
- Only one region (US). The same code could be run on Indian state-level demand data.
- One model forecasts all hours; a model per hour or a deep learning model could be tried next.
