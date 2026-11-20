"""
app.py - Streamlit dashboard for the Electricity Demand Forecasting project.

Run from the project folder (after python src/train.py):

    streamlit run app.py
"""

import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from features import ALL_FEATURES, TARGET, TEST_START, load_with_features  # noqa: E402

MODEL_PATH = ROOT / "models" / "xgb_demand_model.json"
METRICS_PATH = ROOT / "reports" / "metrics.json"

st.set_page_config(page_title="Electricity Demand Forecasting", page_icon="⚡", layout="wide")


@st.cache_data
def get_data():
    return load_with_features()


@st.cache_resource
def get_model():
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


@st.cache_data
def get_test_predictions():
    df = get_data()
    test = df.loc[TEST_START:].copy()
    test["Forecast (XGBoost)"] = get_model().predict(test[ALL_FEATURES])
    test["Baseline (yesterday)"] = test["lag_24h"]
    return test.rename(columns={TARGET: "Actual demand"})


def mape(actual, pred) -> float:
    actual, pred = np.asarray(actual), np.asarray(pred)
    return float(np.mean(np.abs(actual - pred) / actual) * 100)


st.title("⚡ Electricity Demand Forecasting")
st.write(
    "Hourly electricity demand for the PJM East region (USA), 2002-2018. "
    "The model forecasts each hour **one day ahead**, using the calendar "
    "(hour, weekday, holiday...) and demand from 1, 2 and 7 days earlier."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Open a terminal in the project folder and run "
             "`python src/train.py`, then refresh this page.")
    st.stop()

test = get_test_predictions()

tab1, tab2, tab3 = st.tabs(["Forecast explorer", "Demand patterns", "Model scores"])

# ---------------- Tab 1: forecast explorer ----------------
with tab1:
    st.subheader("Pick any period from the test set (2017 - Aug 2018)")
    first_day = test.index.min().date()
    last_day = test.index.max().date()
    c1, c2 = st.columns([2, 1])
    start = c1.date_input("Start date", value=pd.Timestamp("2018-01-08").date(),
                          min_value=first_day, max_value=last_day)
    days = c2.slider("Number of days", 1, 28, 7)
    end = start + timedelta(days=days)
    window = test.loc[str(start):str(end - timedelta(hours=1))]

    if window.empty:
        st.warning("No data in this period. Please pick another date.")
    else:
        cols = ["Actual demand", "Forecast (XGBoost)", "Baseline (yesterday)"]
        st.line_chart(window[cols], color=["#2a78d6", "#eb6834", "#1baf7a"],
                      y_label="Megawatts (MW)")
        m1, m2, m3 = st.columns(3)
        model_err = mape(window["Actual demand"], window["Forecast (XGBoost)"])
        base_err = mape(window["Actual demand"], window["Baseline (yesterday)"])
        m1.metric("Model error (MAPE)", f"{model_err:.2f}%")
        m2.metric("Baseline error (MAPE)", f"{base_err:.2f}%")
        m3.metric("Peak demand in period", f"{window['Actual demand'].max():,.0f} MW")
        st.caption("MAPE = average % error. Lower is better. Baseline (yesterday) = simply copy the demand from the same hour yesterday. If the model error is smaller "
                   "than the baseline error, the model beat the simple rule for this period.")
        with st.expander("See the numbers"):
            st.dataframe(window[cols].round(0))

# ---------------- Tab 2: patterns ----------------
with tab2:
    data = get_data()
    st.subheader("Average demand by hour of the day")
    by_hour = data.groupby(["hour", "is_weekend"])[TARGET].mean().unstack()
    by_hour.columns = ["Weekday", "Weekend"]
    st.line_chart(by_hour, color=["#2a78d6", "#eb6834"], x_label="Hour of the day",
                  y_label="Average demand (MW)")

    st.subheader("Average demand by month")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    by_month = data.groupby("month")[TARGET].mean()
    by_month.index = pd.CategoricalIndex(months, categories=months, ordered=True)
    st.bar_chart(by_month, color="#2a78d6", y_label="Average demand (MW)")
    st.caption("Demand is highest in summer (air conditioning) and winter (heating), "
               "lowest in spring and autumn.")

# ---------------- Tab 3: scores ----------------
with tab3:
    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text())
        st.subheader(f"Test set: {m['test_period']} ({m['test_hours']:,} hours)")
        st.dataframe(pd.DataFrame(m["comparison"]).set_index("model"))
        st.write(f"The best model has **{m['error_reduction_vs_yesterday_baseline_%']}% lower error** "
                 "than simply copying yesterday's value.")
    for name, caption in [("08_feature_importance.png", "What the model relies on most"),
                          ("09_error_by_hour.png", "Average error by hour of the day")]:
        fig = ROOT / "reports" / "figures" / name
        if fig.exists():
            st.image(str(fig), caption=caption)
