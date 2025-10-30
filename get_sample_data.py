import yfinance as yf
import pandas as pd
import os

def fetch_and_export_sp500(start='2019-01-01', end='2024-01-01', output_folder='export'):
    df = yf.download('^GSPC', start=start, end=end)
    df = df.ffill().dropna()
    os.makedirs(output_folder, exist_ok=True)
    outpath = os.path.join(output_folder, 'historical_data_sample.csv')
    df.to_csv(outpath)
    print(f"Exported S&P 500 data to {outpath}")

if __name__ == '__main__':
    fetch_and_export_sp500()
