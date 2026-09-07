# FinancialForecasting

A risk management-centric financial forecasting and scenario analysis tool for S&P 500, built in Python. Designed for internships, professional demos, and decision support.

## Key Features

- Automated S&P 500 data import (via Yahoo Finance)
- Data cleaning and missing value handling
- Forecasting with Prophet and ARIMA (confidence intervals included)
- Scenario analysis (best, worst, stress, expected cases)
- Risk metrics: VaR, CVaR, volatility, max drawdown
- Automated risk insights and warnings
- Exports files directly for Tableau dashboarding
- Modular, commented, and internship-ready Python code

## Getting Started

### 1. Clone the repository

```sh
git clone https://github.com/stephenyupa/FinancialForecasting.git
cd FinancialForecasting
```

### 2. Install dependencies

```sh
pip install -r requirements.txt
```

### 3. Run the forecasting model

```sh
python forecast.py
```

Exports will appear in the `export/` directory.

## Project Structure

```
FinancialForecasting/
├── forecast.py            # Prophet forecasting, risk metrics & scenario analysis
├── walk_forward.py        # Walk-forward directional (up/down) forecasting
├── get_sample_data.py     # Small sample data exporter
├── tests/                 # No-lookahead and fold-overlap tests
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── export/                # Auto-generated output for dashboards
└── data/                  # (optional) Sample/raw data storage
```

## Methodology Overview

- **Data**: Pulled using yfinance’s S&P 500 ticker (^GSPC).
- **Preprocessing**: NaNs handled, daily returns computed, extra features optionally engineered.
- **Forecasting**: Combines classical (ARIMA) and machine learning (Prophet) time series models.
- **Scenario Analysis**: Simulates market shocks, quantifies impact for best, worst, expected, and stress scenarios.
- **Risk Metrics**: Calculates Value at Risk (VaR), Conditional VaR (CVaR), max drawdown, annualized volatility.
- **Insights**: Alerts for excessive risk and adverse forecasts.
- **Visualization**: Outputs plug-and-play CSVs for Tableau/etc.

## For Tableau

- Import `export/forecast.csv` and `export/historical_data.csv`.
- Filter and compare scenarios.
- Build dynamic dashboards with KPIs, scenario toggling, and confidence intervals.

## Customization

- Switch tickers (AAPL, MSFT, ETFs, etc.) by editing the symbol in `forecast.py`.
- Extend by adding new risk metrics, scenario tests, or models.

## Walk-Forward Directional Forecasting (`walk_forward.py`)

A second, independent module that predicts the **sign** of the next
trading day's log return for `^GSPC`, evaluated with a walk-forward
protocol designed to rule out lookahead.

### Data and features

- Daily `^GSPC` data from 2010-01-01 to the present, via `yfinance`.
- Target: `1` if the next day's log return is positive, else `0`.
- Features, all computed from information available at the close of day
  `t` (none touch day `t+1`, the day being predicted):
  - Lagged log returns at 1, 2, 3, 5, and 10 days back from the
    prediction target (`lag_return_1` = today's return, `lag_return_2` =
    yesterday's, etc.)
  - Rolling realized volatility (std of log returns) over trailing 5- and
    21-day windows ending at day `t`.
  - A 10-day momentum term, `close_t / close_{t-10} - 1`.
  - Day-of-week dummies for day `t`.
- `tests/test_walk_forward.py::test_features_use_no_future_information`
  asserts this directly: perturbing the final day's closing price leaves
  every earlier day's feature row byte-for-byte unchanged.

### Walk-forward protocol

Expanding window, refit every year, no lookahead:

1. Train on all data through the end of 2014.
2. Predict every trading day in year `Y`.
3. Refit on all data through the end of year `Y`.
4. Advance to year `Y + 1` and repeat, through the present.

Any scaler (`StandardScaler`, used for logistic regression) is fit on the
training slice only and applied to the test slice — never fit on the full
history. `run_walk_forward()` asserts that no fold's train and test
indices overlap, and `tests/test_walk_forward.py` checks this both at the
fold-construction level and end-to-end.

### Models

- **Baseline**: always predicts the majority class observed in that
  fold's training data (in practice this is "up" in every fold, since
  `^GSPC` closes up more often than not). This is the number the other
  models have to beat.
- **Logistic regression** (scaled features).
- **Gradient-boosted trees** (`sklearn.ensemble.GradientBoostingClassifier`).

### Results (measured, 2015–2026 walk-forward, n = 2,936 trading days/model)

| Model    | Accuracy | Precision | Recall | p-value vs. baseline |
|----------|---------:|----------:|-------:|----------------------:|
| Baseline | 0.5402   | 0.5402    | 1.0000 | —                      |
| Logistic regression | 0.5388 | 0.5418 | 0.9470 | 0.5664 |
| Gradient-boosted trees | 0.5310 | 0.5483 | 0.7484 | 0.8457 |

The p-values come from a one-sided binomial test (`scipy.stats.binomtest`)
of each model's correct-prediction count against the baseline's accuracy
rate as the null success probability. **Neither model beats the
always-predict-up baseline, and neither result is statistically
distinguishable from noise** — both p-values are far above any
conventional significance threshold. Both models degrade recall relative
to the baseline (they occasionally predict "down"), without buying back
enough accuracy to justify it.

Accuracy by year (baseline / logistic regression / gradient-boosted trees):

| Year | Baseline | LogReg | GBC |
|-----:|---------:|-------:|----:|
| 2015 | 0.4722 | 0.4841 | 0.4960 |
| 2016 | 0.5238 | 0.5198 | 0.5357 |
| 2017 | 0.5697 | 0.5657 | 0.4861 |
| 2018 | 0.5259 | 0.5219 | 0.4940 |
| 2019 | 0.5952 | 0.5794 | 0.5238 |
| 2020 | 0.5692 | 0.5731 | 0.5771 |
| 2021 | 0.5714 | 0.5714 | 0.6111 |
| 2022 | 0.4263 | 0.4622 | 0.4821 |
| 2023 | 0.5480 | 0.5240 | 0.5000 |
| 2024 | 0.5675 | 0.5754 | 0.5476 |
| 2025 | 0.5800 | 0.5640 | 0.5840 |
| 2026\* | 0.5294 | 0.5176 | 0.5353 |

\*2026 is a partial year (through early September).

Full per-day predictions and metrics are written to `export/`:
`walk_forward_predictions.csv`, `walk_forward_overall_metrics.csv`, and
`walk_forward_yearly_metrics.csv`.

### Running it

```sh
python walk_forward.py       # runs the walk-forward backtest and prints/export results
python -m pytest tests/      # no-lookahead and fold-overlap tests
```

### Takeaway

At the daily horizon with this feature set, next-day directional
prediction on `^GSPC` is consistent with the efficient-market expectation:
no model here produces an edge over always predicting "up" that survives
a significance test. That is itself a legitimate, useful result — it
rules out a class of naive daily-direction strategies rather than
asserting one works.

## License

[MIT]

---

Made by [Stephen Yupa] for professional development and risk management internship demonstration.
