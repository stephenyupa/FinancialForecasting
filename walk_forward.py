"""Walk-forward directional forecasting for ^GSPC next-day returns.

Predicts the SIGN of the next trading day's log return using only
information available at the close of the current day, evaluated with an
expanding-window walk-forward protocol (no lookahead).
"""
import os

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import binomtest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

SYMBOL = "^GSPC"
START = "2010-01-01"
FIRST_TEST_YEAR = 2015  # train through end of 2014, then predict 2015 onward

LAG_DAYS = (1, 2, 3, 5, 10)
VOL_WINDOWS = (5, 21)
MOMENTUM_WINDOW = 10

FEATURE_COLUMNS = (
    [f"lag_return_{d}" for d in LAG_DAYS]
    + [f"vol_{w}" for w in VOL_WINDOWS]
    + [f"momentum_{MOMENTUM_WINDOW}"]
    + [f"dow_{d}" for d in range(5)]
)

EXPORT_DIR = "export"


# ----------- 1. DATA COLLECTION -------------

def download_data(symbol=SYMBOL, start=START, end=None):
    """Download daily OHLCV data and flatten yfinance's MultiIndex columns."""
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.ffill().dropna()
    return df


# ----------- 2. FEATURE / TARGET CONSTRUCTION -------------

def build_feature_frame(df):
    """Build features known at the close of day t and the sign of day t+1's
    log return as the target.

    Lag convention: lag_return_k is the log return realized k trading days
    before the day being *predicted* (t+1). So lag_return_1 = today's
    (day t) return, lag_return_2 = yesterday's return, etc. All lag, rolling
    volatility, and momentum features are computed strictly from data at or
    before day t's close, so none of them touch the day t+1 return used as
    the target.
    """
    close = df["Close"]
    log_return = np.log(close / close.shift(1))

    feat = pd.DataFrame(index=df.index)
    for k in LAG_DAYS:
        feat[f"lag_return_{k}"] = log_return.shift(k - 1)

    for w in VOL_WINDOWS:
        feat[f"vol_{w}"] = log_return.rolling(w).std()

    feat[f"momentum_{MOMENTUM_WINDOW}"] = (
        close / close.shift(MOMENTUM_WINDOW) - 1
    )

    dow = pd.get_dummies(df.index.dayofweek, prefix="dow")
    dow.index = df.index
    for d in range(5):
        col = f"dow_{d}"
        feat[col] = dow[col].astype(int) if col in dow.columns else 0

    feat["target"] = (log_return.shift(-1) > 0).astype(int)

    feat = feat.dropna()
    return feat


# ----------- 3. WALK-FORWARD FOLDS -------------

def make_folds(dates, first_test_year=FIRST_TEST_YEAR):
    """Expanding-window folds: train on all years < Y, test on year Y."""
    years = pd.Index(dates).year
    test_years = sorted(y for y in years.unique() if y >= first_test_year)

    folds = []
    for year in test_years:
        train_mask = years < year
        test_mask = years == year
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        folds.append((train_mask, test_mask, year))
    return folds


# ----------- 4. MODELS -------------

def predict_baseline(y_train, n_test):
    majority_class = int(round(y_train.mean())) if len(y_train) else 1
    return np.full(n_test, majority_class)


def predict_logreg(X_train, y_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_s, y_train)
    return model.predict(X_test_s)


def predict_gbc(X_train, y_train, X_test):
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test)


MODEL_FUNCS = {
    "baseline": lambda X_train, y_train, X_test: predict_baseline(y_train, len(X_test)),
    "logreg": predict_logreg,
    "gbc": predict_gbc,
}


# ----------- 5. WALK-FORWARD RUN -------------

def run_walk_forward(feat, model_names=("baseline", "logreg", "gbc")):
    """Run expanding-window walk-forward evaluation for each model.

    Returns a long DataFrame with one row per (date, model) prediction.
    """
    dates = feat.index
    folds = make_folds(dates)
    X = feat[FEATURE_COLUMNS].values
    y = feat["target"].values

    records = []
    for train_mask, test_mask, year in folds:
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)
        assert not set(train_idx) & set(test_idx), (
            f"train/test index overlap in fold year {year}"
        )

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        test_dates = dates[test_idx]

        for name in model_names:
            preds = MODEL_FUNCS[name](X_train, y_train, X_test)
            for date, true_val, pred_val in zip(test_dates, y_test, preds):
                records.append(
                    {
                        "date": date,
                        "year": year,
                        "model": name,
                        "y_true": int(true_val),
                        "y_pred": int(pred_val),
                    }
                )

    return pd.DataFrame.from_records(records)


# ----------- 6. EVALUATION -------------

def evaluate(results):
    """Compute overall + per-year metrics and a binomial test vs baseline."""
    baseline_rate = (
        results.loc[results["model"] == "baseline", "y_true"]
        .eq(results.loc[results["model"] == "baseline", "y_pred"])
        .mean()
    )

    overall_rows = []
    yearly_rows = []

    for name, group in results.groupby("model"):
        y_true = group["y_true"].values
        y_pred = group["y_pred"].values
        n = len(y_true)
        n_correct = int((y_true == y_pred).sum())
        acc = n_correct / n

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        p_value = np.nan
        if name != "baseline":
            test = binomtest(n_correct, n, p=baseline_rate, alternative="greater")
            p_value = test.pvalue

        overall_rows.append(
            {
                "model": name,
                "n": n,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "p_value_vs_baseline": p_value,
            }
        )

        for year, ygroup in group.groupby("year"):
            yt, yp = ygroup["y_true"].values, ygroup["y_pred"].values
            yearly_rows.append(
                {
                    "model": name,
                    "year": year,
                    "n": len(yt),
                    "accuracy": (yt == yp).mean(),
                }
            )

    overall_df = pd.DataFrame(overall_rows).set_index("model")
    yearly_df = pd.DataFrame(yearly_rows)
    return overall_df, yearly_df, baseline_rate


# ----------- 7. REPORTING -------------

def print_summary(overall_df, yearly_df, baseline_rate):
    print("\n---- WALK-FORWARD DIRECTIONAL FORECAST SUMMARY ----")
    print(f"Baseline (majority class) accuracy: {baseline_rate:.4f}\n")

    print("Overall metrics by model:")
    print(
        overall_df[
            ["n", "accuracy", "precision", "recall", "p_value_vs_baseline"]
        ].to_string(float_format=lambda v: f"{v:.4f}")
    )

    print("\nAccuracy by year:")
    pivot = yearly_df.pivot(index="year", columns="model", values="accuracy")
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))
    print()


def save_results(results, overall_df, yearly_df):
    os.makedirs(EXPORT_DIR, exist_ok=True)
    results.to_csv(os.path.join(EXPORT_DIR, "walk_forward_predictions.csv"), index=False)
    overall_df.to_csv(os.path.join(EXPORT_DIR, "walk_forward_overall_metrics.csv"))
    yearly_df.to_csv(os.path.join(EXPORT_DIR, "walk_forward_yearly_metrics.csv"), index=False)


# ----------- MAIN -------------

def main():
    df = download_data()
    feat = build_feature_frame(df)
    results = run_walk_forward(feat)
    overall_df, yearly_df, baseline_rate = evaluate(results)
    print_summary(overall_df, yearly_df, baseline_rate)
    save_results(results, overall_df, yearly_df)


if __name__ == "__main__":
    main()
