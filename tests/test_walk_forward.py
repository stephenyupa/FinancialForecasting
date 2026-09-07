"""Tests for the walk-forward directional forecasting module.

These focus on the two properties that matter most for a walk-forward
backtest: that features carry no lookahead into the target, and that no
fold ever trains and tests on the same day.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import walk_forward as wf


def _synthetic_price_frame(n=800, seed=0):
    """A synthetic daily price series with a mild autocorrelated drift, so
    that a real (if modest) directional edge exists at 1-day-ahead but
    should decay for a target shifted further into the future.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n)
    drift = np.zeros(n)
    noise = rng.normal(0, 1.0, n)
    for i in range(1, n):
        drift[i] = 0.15 * drift[i - 1] + noise[i]
    returns = 0.0002 + 0.001 * np.tanh(drift / 5)
    price = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({"Close": price}, index=dates)
    return df


def _accuracy_for_target(feat, target_col):
    """Single expanding train/test split accuracy for a given target column,
    used only to compare relative signal strength (not the full walk-forward
    protocol, which is exercised separately in test_make_folds_no_overlap).
    """
    feat = feat.dropna(subset=wf.FEATURE_COLUMNS + [target_col])
    split = int(len(feat) * 0.7)
    train, test = feat.iloc[:split], feat.iloc[split:]

    X_train, y_train = train[wf.FEATURE_COLUMNS].values, train[target_col].values
    X_test, y_test = test[wf.FEATURE_COLUMNS].values, test[target_col].values

    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(scaler.transform(X_train), y_train)
    preds = model.predict(scaler.transform(X_test))

    baseline_rate = max(y_train.mean(), 1 - y_train.mean())
    accuracy = (preds == y_test).mean()
    return accuracy, baseline_rate


def test_features_use_no_future_information():
    """Every feature column at day t must be derivable from data available
    at or before day t's close, i.e. from log_return at lags >= 0 relative
    to day t. Perturbing the day t+1 return must leave day t's feature row
    unchanged.
    """
    df = _synthetic_price_frame()
    feat = wf.build_feature_frame(df)

    df_perturbed = df.copy()
    close = df_perturbed["Close"].copy()
    close.iloc[-1] = close.iloc[-1] * 1.5  # blow up the final day's return
    df_perturbed["Close"] = close
    feat_perturbed = wf.build_feature_frame(df_perturbed)

    common_index = feat.index.intersection(feat_perturbed.index)
    # The last common row's features are computed from the second-to-last
    # close onward, so they must be identical even though the final day's
    # price (and therefore the day-before's target) changed drastically.
    second_last_common = common_index[-2] if len(common_index) > 1 else None
    assert second_last_common is not None
    row = feat.loc[second_last_common, wf.FEATURE_COLUMNS]
    row_perturbed = feat_perturbed.loc[second_last_common, wf.FEATURE_COLUMNS]
    pd.testing.assert_series_equal(row, row_perturbed, check_names=False)


def test_no_lookahead_shifted_target_degrades_toward_baseline():
    """If we shift the target one more day into the future (t+2 instead of
    t+1) using the SAME features, accuracy should degrade toward the
    baseline rate. If it didn't, that would indicate the features leak
    information about the target rather than genuinely predicting it.
    """
    df = _synthetic_price_frame()
    feat = wf.build_feature_frame(df)

    close = df["Close"]
    log_return = np.log(close / close.shift(1))
    feat = feat.copy()
    feat["target_plus2"] = (log_return.shift(-2) > 0).astype(int)
    feat = feat.dropna()

    acc_t1, baseline_rate = _accuracy_for_target(feat, "target")
    acc_t2, _ = _accuracy_for_target(feat, "target_plus2")

    edge_t1 = acc_t1 - baseline_rate
    edge_t2 = acc_t2 - baseline_rate

    assert edge_t2 <= edge_t1 + 1e-9, (
        f"edge over baseline did not shrink when the target was shifted "
        f"further into the future (t+1 edge={edge_t1:.4f}, "
        f"t+2 edge={edge_t2:.4f})"
    )


def test_make_folds_no_train_test_overlap():
    df = _synthetic_price_frame(n=1800)
    feat = wf.build_feature_frame(df)
    folds = wf.make_folds(feat.index, first_test_year=feat.index.year.min() + 2)

    assert len(folds) > 0
    for train_mask, test_mask, year in folds:
        train_idx = set(np.flatnonzero(train_mask))
        test_idx = set(np.flatnonzero(test_mask))
        assert train_idx.isdisjoint(test_idx), f"overlap found in fold {year}"
        # Expanding window: every training index must precede every test index.
        assert max(train_idx) < min(test_idx)


def test_run_walk_forward_has_no_overlap_end_to_end():
    """run_walk_forward asserts internally on every fold that train and test
    indices are disjoint; this just confirms it runs clean and produces one
    row per test-set day with no duplicated dates for a single model.
    """
    df = _synthetic_price_frame(n=1800)
    feat = wf.build_feature_frame(df)
    results = wf.run_walk_forward(feat, model_names=("baseline",))
    assert not results.empty
    assert results["date"].is_unique
    assert set(results["model"].unique()) == {"baseline"}
