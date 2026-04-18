import os
import pandas as pd
import yfinance as yf
import akshare as ak
from datetime import datetime, timedelta
import time

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "ashare")
os.makedirs(DATA_DIR, exist_ok=True)

BIG6_BANKS = {
    "601398": "601398.SS", # 工行
    "601288": "601288.SS", # 农行
    "601988": "601988.SS", # 中行
    "601939": "601939.SS", # 建行
    "601328": "601328.SS", # 交行
    "601658": "601658.SS", # 邮储
}

def fetch_bank_data_10yr(symbol, yf_symbol):
    print(f"Fetching 10-year data for {symbol} ({yf_symbol})...")
    
    # 1. Fetch Prices & Dividends from yfinance (10Y)
    start_date = (datetime.now() - timedelta(days=365 * 10 + 7)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    df = pd.DataFrame()
    for attempt in range(3):
        try:
            df = yf.download(yf_symbol, start=start_date, end=end_date, actions=True)
            if not df.empty:
                break
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {symbol}: {e}")
            time.sleep(2)
            
    if df.empty:
        print(f"Failed to fetch data for {symbol} after retries")
        return
    
    # Flatten columns if multi-indexed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df.reset_index()
    df.rename(columns={'Date': 'Date', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume', 'Dividends': 'DivCash'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'DivCash']]
    
    # 2. Fetch Daily PB from AkShare (EastMoney)
    print(f"Fetching PB data from AkShare for {symbol}...")
    try:
        # stock_value_em provides daily valuation data
        val_df = ak.stock_value_em(symbol=symbol)
        # Rename columns based on observed output
        # Actually: 0: Date, 1: Close, 9: PB (市净率), 10: PEG, 11: PCF, 12: PS
        val_df.columns = ['Date', 'Close_EM', 'Pct', 'Total_Market_Cap', 'Float_Market_Cap', 'Total_Shares', 'Float_Shares', 'PE_TTM', 'PE_Static', 'PB', 'PEG', 'PCF', 'PS']
        val_df['Date'] = pd.to_datetime(val_df['Date']).dt.date
        val_df = val_df[['Date', 'PB']]
        
        # Merge
        df = pd.merge(df, val_df, on='Date', how='left')
    except Exception as e:
        print(f"Could not fetch PB for {symbol}: {e}")
        df['PB'] = None
        
    # 3. Add DividendYieldTTM (re-calculate using 10Y logic if needed, 
    # but for now we'll just keep standard columns)
    # The user didn't explicitly ask for DividendYieldTTM in the new CSV format, 
    # but our existing scripts expect it. Let's calculate a simple one.
    
    # Calculate DividendYieldTTM (Rolling sum of dividends over last 365 days / Close)
    df = df.sort_values('Date')
    df['DivSum1Y'] = df['DivCash'].rolling(window=252, min_periods=0).sum() # Approximation for TTM
    df['DividendYieldTTM'] = (df['DivSum1Y'] / df['Close']) * 100
    
    # Clean up
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'DividendYieldTTM', 'DivCash', 'PB']]
    
    # Save
    csv_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {symbol}.csv to {csv_path}")

if __name__ == "__main__":
    for symbol, yf_symbol in BIG6_BANKS.items():
        fetch_bank_data_10yr(symbol, yf_symbol)
        time.sleep(1) # Polite delay
    print("All data fetched successfully.")
