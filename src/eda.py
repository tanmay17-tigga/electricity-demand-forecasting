"""
eda.py  (Exploratory Data Analysis)
-----------------------------------
Looks at the electricity data BEFORE building any model and saves charts
to reports/figures/. Run from the project folder:

    python src/eda.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

from features import ROOT, TARGET, add_features, load_data
from plot_style import ACTUAL_COLOR, MUTED, PRED_COLOR, apply_style

FIG_DIR = ROOT / "reports" / "figures"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    df = add_features(load_data())
    y = df[TARGET]
    print(f"Hours of data: {len(df):,}  from {df.index.min()} to {df.index.max()}")
    print(f"Average demand: {y.mean():,.0f} MW   lowest: {y.min():,.0f} MW   highest: {y.max():,.0f} MW")
    print(f"Lowest hour: {y.idxmin()}   Highest hour: {y.idxmax()}")

    # ---------- 1. The whole history (daily average) ----------
    daily = y.resample("D").mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily.index, daily.values, color=ACTUAL_COLOR, linewidth=0.8)
    ax.set_title("Daily average electricity demand, 2002-2018 (PJM East)")
    ax.set_ylabel("Megawatts (MW)")
    fig.savefig(FIG_DIR / "01_full_history.png")
    plt.close(fig)

    # ---------- 2. Average demand by hour: weekday vs weekend ----------
    by_hour = df.groupby(["is_weekend", "hour"])[TARGET].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(by_hour.index, by_hour[0], color=ACTUAL_COLOR, linewidth=2, marker="o", markersize=4, label="Weekday")
    ax.plot(by_hour.index, by_hour[1], color=PRED_COLOR, linewidth=2, marker="o", markersize=4, label="Weekend")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of the day")
    ax.set_ylabel("Average demand (MW)")
    ax.set_title("Demand is low at night and peaks in the evening")
    ax.legend(frameon=False)
    fig.savefig(FIG_DIR / "02_by_hour.png")
    plt.close(fig)
    print("Weekday average by hour (MW):", by_hour[0].round(0).astype(int).to_dict())

    # ---------- 3. Average demand by month ----------
    by_month = df.groupby("month")[TARGET].mean()
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, by_month.values, color=ACTUAL_COLOR, width=0.6)
    ax.set_ylim(by_month.min() * 0.85, by_month.max() * 1.05)
    ax.set_ylabel("Average demand (MW)")
    ax.set_title("Two peaks a year: summer (air conditioning) and winter (heating)")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG_DIR / "03_by_month.png")
    plt.close(fig)
    print("Average by month (MW):", by_month.round(0).astype(int).to_dict())

    # ---------- 4. Average demand by day of the week ----------
    by_dow = df.groupby("dayofweek")[TARGET].mean()
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(days, by_dow.values, color=[ACTUAL_COLOR] * 5 + [PRED_COLOR] * 2, width=0.6)
    ax.set_ylim(by_dow.min() * 0.9, by_dow.max() * 1.03)
    ax.set_ylabel("Average demand (MW)")
    ax.set_title("Weekends use less electricity")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG_DIR / "04_by_dayofweek.png")
    plt.close(fig)

    # ---------- 5. One example week ----------
    week = y.loc["2017-07-17":"2017-07-23"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(week.index, week.values, color=ACTUAL_COLOR, linewidth=2)
    ax.set_title("One summer week (17-23 July 2017): a clear daily cycle")
    ax.set_ylabel("Megawatts (MW)")
    fig.autofmt_xdate()
    fig.savefig(FIG_DIR / "05_example_week.png")
    plt.close(fig)

    print(f"Charts saved in {FIG_DIR}")


if __name__ == "__main__":
    main()
