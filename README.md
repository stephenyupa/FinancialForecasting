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
- Modular, commented, and Python code

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
├── forecast.py            # Main forecasting & risk tool
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

## License

[MIT]

---

Made by [Stephen Yupa] for professional development and risk management internship demonstration.
