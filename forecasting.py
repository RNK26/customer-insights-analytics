"""Weekly revenue forecasting: Prophet against a seasonal-naive baseline.

The split is time-ordered, never random.

Run:  python forecasting.py
Writes: outputs/forecast_actual_vs_predicted.png, outputs/forecast_metrics.csv
"""

import duckdb
import matplotlib
matplotlib.use("Agg")           # write PNGs without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet

DB = "data/retail.duckdb"
TEST_WEEKS = 13                 # hold out the final quarter
SEASONAL_LAG = 52               # weeks in a year -> the seasonal-naive lookback


def load_weekly() -> pd.DataFrame:
    """One revenue figure per week.

    Weekly rather than daily because this retailer is shut on Saturdays -- one
    trading Saturday in two years -- so a daily series carries a structural
    zero every week that is noise rather than signal.
    """
    con = duckdb.connect(DB, read_only=True)
    df = con.execute(
        """
        SELECT DATE_TRUNC('week', InvoiceDate) AS week,
               SUM(Revenue)                    AS revenue
        FROM sales
        GROUP BY 1
        ORDER BY 1
        """
    ).df()
    con.close()

    # First and last weeks are partial: the data starts 2009-12-01 and ends
    # 2011-12-09, both mid-week. Left in, the series looks like it starts low
    # and crashes at the end, and the model learns that shape.
    df = df.iloc[1:-1].reset_index(drop=True)
    return df


def seasonal_naive(series: pd.Series, lag: int = SEASONAL_LAG) -> pd.Series:
    """Predicts each week from the value `lag` weeks earlier.

    No fitting, no library, just "this week will look like the same week last
    year". On a business with seasonality this strong it is a hard bar, which
    is why it is the one worth measuring against.
    """
    return series.shift(lag)


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Average miss as a percentage of actual.

    Scale-free, so it compares across series. Blows up when actuals approach
    zero, which is not a risk here.
    """
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def main() -> None:
    weekly = load_weekly()
    print(f"complete weeks: {len(weekly)}  "
          f"({weekly.week.min():%Y-%m-%d} -> {weekly.week.max():%Y-%m-%d})")

    # Last TEST_WEEKS weeks are the test set. A random split or k-fold would
    # put future weeks in train and past weeks in test, so the model would be
    # learning from data it would not have at prediction time.
    split_at = len(weekly) - TEST_WEEKS
    train = weekly.iloc[:split_at].copy()
    test = weekly.iloc[split_at:].copy()
    print(f"train: {len(train)} weeks  ({train.week.min():%Y-%m-%d} -> {train.week.max():%Y-%m-%d})")
    print(f"test : {len(test)} weeks  ({test.week.min():%Y-%m-%d} -> {test.week.max():%Y-%m-%d})")

    actual = test["revenue"].to_numpy()

    # Computed over the full series so the 52-week lookback can reach back
    # into training, then sliced down to the test rows.
    baseline_full = seasonal_naive(weekly["revenue"])
    baseline_pred = baseline_full.iloc[split_at:].to_numpy()
    assert not np.isnan(baseline_pred).any(), "baseline needs >=52 weeks of history"

    # Prophet insists on columns named ds and y. Fitted on train only.
    prophet_train = train.rename(columns={"week": "ds", "revenue": "y"})
    model = Prophet(
        yearly_seasonality=True,    # the Sep-Nov gift-retail peak
        weekly_seasonality=False,   # meaningless: the data is already weekly
        daily_seasonality=False,
    )
    model.fit(prophet_train)

    future = model.make_future_dataframe(periods=TEST_WEEKS, freq="W-MON")
    forecast = model.predict(future)
    prophet_pred = forecast["yhat"].iloc[split_at:].to_numpy()

    rows = [
        {"model": "Seasonal naive (52w)", "MAE": mae(actual, baseline_pred),
         "MAPE_%": mape(actual, baseline_pred)},
        {"model": "Prophet", "MAE": mae(actual, prophet_pred),
         "MAPE_%": mape(actual, prophet_pred)},
    ]
    results = pd.DataFrame(rows)
    results["MAE_vs_baseline_%"] = (
        (results["MAE"].iloc[0] - results["MAE"]) / results["MAE"].iloc[0] * 100
    )
    results.to_csv("outputs/forecast_metrics.csv", index=False)

    print("\n--- test-set accuracy (13 held-out weeks) ---")
    print(results.to_string(index=False,
                            float_format=lambda v: f"{v:,.1f}"))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train["week"], train["revenue"], color="#888", lw=1, label="Train (actual)")
    ax.plot(test["week"], actual, color="#111", lw=2, marker="o", ms=4, label="Test (actual)")
    ax.plot(test["week"], baseline_pred, color="#d1495b", lw=1.8, ls="--", label="Seasonal naive")
    ax.plot(test["week"], prophet_pred, color="#0077b6", lw=1.8, ls="-.", label="Prophet")
    ax.axvline(test["week"].iloc[0], color="#aaa", ls=":", lw=1)
    ax.annotate("train / test split", (test["week"].iloc[0], ax.get_ylim()[1] * 0.95),
                fontsize=8, color="#666", ha="right")
    ax.set_title("Weekly revenue: actual vs forecast (13-week holdout)")
    ax.set_ylabel("Revenue (GBP)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v/1000:,.0f}k")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("outputs/forecast_actual_vs_predicted.png", dpi=140)
    print("\nsaved outputs/forecast_actual_vs_predicted.png")

    # does either model actually see the Q4 spike?
    peak_i = int(np.argmax(actual))
    print(f"\npeak test week   : {test['week'].iloc[peak_i]:%Y-%m-%d}")
    print(f"  actual         : {actual[peak_i]:,.0f}")
    print(f"  seasonal naive : {baseline_pred[peak_i]:,.0f}")
    print(f"  prophet        : {prophet_pred[peak_i]:,.0f}")


if __name__ == "__main__":
    main()
