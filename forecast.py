import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# ----------- 1. DATA COLLECTION & PREPROCESSING -------------

def download_data(symbol='^GSPC', start='2010-01-01', end='2025-01-01'):
    """Download historical data for S&P 500"""
    df = yf.download(symbol, start=start, end=end)
    df = df.ffill().dropna()
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    return df

# ----------- 2. RISK METRIC CALCULATIONS ---------------------

def calc_volatility(returns):
    return returns.std() * np.sqrt(252)  # Annualized

def calc_var(returns, alpha=0.05):
    return np.percentile(returns, 100 * alpha)

def calc_cvar(returns, alpha=0.05):
    var = calc_var(returns, alpha)
    return returns[returns <= var].mean()

def max_drawdown(series):
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min()  # This returns a scalar

# ----------- 3. FORECASTING (Prophet example) ----------------

def run_prophet(df):
    # Grab ONLY Date and Close, ensure no index column is present and types are right
    prophet_df = df.reset_index()[['Date', 'Close']].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    prophet_df['y'] = pd.to_numeric(prophet_df['y'])
    m = Prophet(interval_width=0.95)
    m.fit(prophet_df[['ds', 'y']])
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    return forecast, m

# ----------- 4. SCENARIO STRESS TESTING ----------------------

def scenario_analysis(last_price, scenario='expected'):
    # You can enhance with more nuanced models
    shifts = {'best': 1.1, 'expected': 1.0, 'worst': 0.85, 'stress': 0.7}
    adjustment = shifts.get(scenario, 1.0)
    return last_price * adjustment

# ----------- 5. RISK INSIGHT GENERATION ----------------------

def risk_report(df, forecast, VaR_threshold=-0.03):
    returns = df['Return']
    vol = calc_volatility(returns)
    var_95 = calc_var(returns, 0.05)
    cvar_95 = calc_cvar(returns, 0.05)
    mdd = max_drawdown(df['Close'].squeeze())

    warnings = []
    if var_95 < VaR_threshold:
        warnings.append("WARNING: VaR exceeds threshold (high downside risk).")
    if mdd < -0.25:
        warnings.append("CAUTION: Maximum historical drawdown indicates high risk.")

    report = f"""
    ---- RISK REPORT ----
    Annualized Volatility: {vol:.2%}
    95% Value at Risk (VaR): {var_95:.2%}
    95% Conditional VaR (CVaR): {cvar_95:.2%}
    Max Drawdown: {mdd:.2%}
    {' '.join(warnings) if warnings else 'No heightened risk detected.'}
    """
    print(report)
    return report

# ----------- 6. VISUALIZATION (to be pushed to Tableau) -------

def save_results_for_tableau(df, forecast):
    os.makedirs('export', exist_ok=True)  # Ensure the export directory exists
    df.to_csv('export/historical_data.csv')
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('export/forecast.csv')

# ----------- MAIN EXECUTION FLOW -----------------------------

if __name__ == "__main__":
    df = download_data()
    print("Downloaded data. Calculating risk metrics...")
    # Ensure last_close is always a scalar float, not a pandas Series, for scenario analysis
    last_close = float(df['Close'].iloc[-1])

    # Financial forecasting
    forecast, model = run_prophet(df)

    # Scenario analysis
    scenarios = { name: scenario_analysis(last_close, name)
                  for name in ('best', 'expected', 'worst', 'stress') }
    print("Scenario Simulation:", scenarios)

    # Risk reporting
    risk_report(df, forecast)

    # Export
    save_results_for_tableau(df, forecast)

    # Visualization (example)
    plt.figure(figsize=(10,6))
    plt.plot(df['Close'], label="Historical")
    plt.plot(forecast['ds'], forecast['yhat'], label="Prophet Forecast")
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color="gray", alpha=0.2)
    plt.title("S&P 500 Forecast with Confidence Intervals")
    plt.legend()
    plt.savefig('export/forecast_plot.png')
    plt.show()