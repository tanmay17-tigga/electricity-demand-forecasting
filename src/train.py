"""
train.py
--------
Builds the forecasting models, compares them, and saves the best one.

Run from the project folder:

    python src/train.py

Steps:
  1. Load data and build features (features.py).
  2. Split by time:  train = 2002-2015,  validation = 2016,  test = 2017 to Aug 2018.
  3. Two simple baselines (no learning at all):
        - "Same hour yesterday"
        - "Same hour last week"
  4. Two XGBoost models:
        - Calendar only (hour, day, month, holiday ...)
        - Calendar + past demand (lags)
     XGBoost stops adding trees when the validation score stops improving.
  5. Score every model on the test set with MAE, RMSE and MAPE.
  6. Save the best model, the scores, test predictions and charts.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from features import ALL_FEATURES, CALENDAR_FEATURES, ROOT, TARGET, load_with_features, split
from plot_style import ACTUAL_COLOR, BASELINE_COLOR, MUTED, PRED_COLOR, apply_style

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
RANDOM_STATE = 42


def metrics(actual, pred) -> dict:
    actual, pred = np.asarray(actual), np.asarray(pred)
    err = actual - pred
    return {
        "MAE_MW": round(float(np.mean(np.abs(err))), 1),
        "RMSE_MW": round(float(np.sqrt(np.mean(err ** 2))), 1),
        "MAPE_%": round(float(np.mean(np.abs(err) / actual) * 100), 2),
    }


def make_xgb() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=2000,          # upper limit; early stopping picks the real number
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="reg:squarederror",
        early_stopping_rounds=100,  # stop if validation does not improve for 100 trees
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    df = load_with_features()
    train, val, test = split(df)
    print(f"Train: {train.index.min():%Y-%m-%d} to {train.index.max():%Y-%m-%d} ({len(train):,} hours)")
    print(f"Valid: {val.index.min():%Y-%m-%d} to {val.index.max():%Y-%m-%d} ({len(val):,} hours)")
    print(f"Test : {test.index.min():%Y-%m-%d} to {test.index.max():%Y-%m-%d} ({len(test):,} hours)")

    y_test = test[TARGET]
    preds = pd.DataFrame({"actual": y_test})

    # ---- Baselines ----
    preds["same_hour_yesterday"] = test["lag_24h"]
    preds["same_hour_last_week"] = test["lag_168h"]

    # ---- XGBoost: calendar only ----
    print("Training XGBoost (calendar only) ...")
    model_cal = make_xgb()
    model_cal.fit(train[CALENDAR_FEATURES], train[TARGET],
                  eval_set=[(val[CALENDAR_FEATURES], val[TARGET])], verbose=False)
    preds["xgb_calendar_only"] = model_cal.predict(test[CALENDAR_FEATURES])
    print(f"   best number of trees: {model_cal.best_iteration + 1}")

    # ---- XGBoost: calendar + lags ----
    print("Training XGBoost (calendar + past demand) ...")
    model = make_xgb()
    model.fit(train[ALL_FEATURES], train[TARGET],
              eval_set=[(val[ALL_FEATURES], val[TARGET])], verbose=False)
    preds["xgb_with_lags"] = model.predict(test[ALL_FEATURES])
    print(f"   best number of trees: {model.best_iteration + 1}")

    # ---- Score everything on the test set ----
    names = {
        "same_hour_yesterday": "Baseline: same hour yesterday",
        "same_hour_last_week": "Baseline: same hour last week",
        "xgb_calendar_only": "XGBoost: calendar only",
        "xgb_with_lags": "XGBoost: calendar + past demand",
    }
    rows = [{"model": label, **metrics(preds["actual"], preds[col])} for col, label in names.items()]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    print("\nTest set comparison (lower is better):")
    print(comparison.to_string(index=False))

    best = comparison.loc[comparison["MAPE_%"].idxmin()]
    baseline_mape = comparison.loc[0, "MAPE_%"]
    improvement = (1 - best["MAPE_%"] / baseline_mape) * 100
    print(f"\nBest: {best['model']}  ->  MAPE {best['MAPE_%']}% "
          f"({improvement:.0f}% lower error than 'same hour yesterday')")

    # ---- Save ----
    model.save_model(MODEL_DIR / "xgb_demand_model.json")
    preds.round(1).to_csv(REPORT_DIR / "test_predictions.csv", index_label="datetime")
    (REPORT_DIR / "metrics.json").write_text(json.dumps({
        "best_model": best["model"],
        "test_period": f"{test.index.min():%Y-%m-%d} to {test.index.max():%Y-%m-%d}",
        "test_hours": len(test),
        "comparison": rows,
        "error_reduction_vs_yesterday_baseline_%": round(improvement, 1),
        "best_iteration": int(model.best_iteration),
        "features": ALL_FEATURES,
    }, indent=2))

    # Error by hour of day (useful extra insight)
    abs_pct = (preds["actual"] - preds["xgb_with_lags"]).abs() / preds["actual"] * 100
    by_hour = abs_pct.groupby(preds.index.hour).mean()
    print("Average % error by hour (best model):", by_hour.round(2).to_dict())

    # ---- Charts ----
    # 1. Model comparison
    fig, ax = plt.subplots(figsize=(8, 3.8))
    order = comparison.sort_values("MAPE_%", ascending=False)
    colors = [PRED_COLOR if m == best["model"] else ACTUAL_COLOR for m in order["model"]]
    ax.barh(order["model"], order["MAPE_%"], color=colors, height=0.55)
    for i, v in enumerate(order["MAPE_%"]):
        ax.text(v + 0.1, i, f"{v:.2f}%", va="center", color=MUTED)
    ax.set_xlabel("MAPE on test set (average % error, lower is better)")
    ax.set_title("Model comparison")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG_DIR / "06_model_comparison.png")
    plt.close(fig)

    # 2. Actual vs predicted for one test week
    wk = preds.loc["2018-01-08":"2018-01-14"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(wk.index, wk["actual"], color=ACTUAL_COLOR, linewidth=2, label="Actual demand")
    ax.plot(wk.index, wk["xgb_with_lags"], color=PRED_COLOR, linewidth=2, label="XGBoost forecast")
    ax.plot(wk.index, wk["same_hour_yesterday"], color=BASELINE_COLOR, linewidth=1.5,
            linestyle="--", label="Baseline: same hour yesterday")
    ax.set_title("Test week 8-14 January 2018: forecast vs actual")
    ax.set_ylabel("Megawatts (MW)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.18)  # room for the legend
    ax.legend(frameon=False, loc="upper center", ncol=3)
    fig.autofmt_xdate()
    fig.savefig(FIG_DIR / "07_forecast_week.png")
    plt.close(fig)

    # 3. Feature importance
    imp = pd.Series(model.feature_importances_, index=ALL_FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index, imp.values, color=ACTUAL_COLOR, height=0.6)
    ax.set_xlabel("Importance (share of total)")
    ax.set_title("What the model relies on most")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG_DIR / "08_feature_importance.png")
    plt.close(fig)
    imp.sort_values(ascending=False).round(4).to_csv(REPORT_DIR / "feature_importance.csv", header=["importance"])

    # 4. Error by hour of the day
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(by_hour.index, by_hour.values, color=ACTUAL_COLOR, width=0.7)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of the day")
    ax.set_ylabel("Average % error")
    ax.set_title("When is the forecast most and least accurate?")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG_DIR / "09_error_by_hour.png")
    plt.close(fig)

    # 5. Predicted vs actual scatter
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(preds["actual"], preds["xgb_with_lags"], s=3, alpha=0.25, color=ACTUAL_COLOR)
    lo, hi = preds["actual"].min(), preds["actual"].max()
    ax.plot([lo, hi], [lo, hi], color=PRED_COLOR, linewidth=1.5, label="Perfect forecast")
    ax.set_xlabel("Actual demand (MW)")
    ax.set_ylabel("Forecast (MW)")
    ax.set_title("Every test hour: forecast vs actual")
    ax.legend(frameon=False)
    fig.savefig(FIG_DIR / "10_pred_vs_actual.png")
    plt.close(fig)

    print(f"\nSaved model to {MODEL_DIR / 'xgb_demand_model.json'}")
    print(f"Saved scores, predictions and charts to {REPORT_DIR}")


if __name__ == "__main__":
    main()
