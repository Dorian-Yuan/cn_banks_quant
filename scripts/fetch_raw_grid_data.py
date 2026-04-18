import os
import yfinance as yf
import pandas as pd
from datetime import datetime

STOCKS = {
    "601398": "601398.SS", # 工行
    "601288": "601288.SS", # 农行
    "601988": "601988.SS", # 中行
    "601939": "601939.SS", # 建行
    "601328": "601328.SS", # 交行
    "601658": "601658.SS", # 邮储
}

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_raw_data():
    for symbol, yf_symbol in STOCKS.items():
        print(f"Fetching {yf_symbol}...")
        df = yf.download(yf_symbol, start="2019-12-01", auto_adjust=False, actions=True)
        if not df.empty:
            # Drop multi-index if exists
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            # We need Date, Open, High, Low, Close, Dividends
            output_file = os.path.join(DATA_DIR, f"{symbol}_raw.csv")
            df.to_csv(output_file, index=False)
            print(f"Saved to {output_file}")
        else:
            print(f"Failed to fetch {yf_symbol}")

if __name__ == "__main__":
    fetch_raw_data()
