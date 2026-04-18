import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- Config ---
STOCKS = {
    "601398": "工行",
    "601288": "农行",
    "601988": "中行",
    "601939": "建行",
    "601328": "交行",
    "601658": "邮储",
}
INITIAL_CASH = 30000
COMMISSION_RATE = 0.0000854
MIN_COMMISSION = 0.5
STAMP_DUTY_RATE = 0.0005 # 0.05%
DATA_DIR = "data/raw"

def get_closest_shares(amount, price):
    """Calculates number of shares closest to target amount, in multiples of 100."""
    lot_size = 100
    if price <= 0: return 0
    raw_shares = amount / price
    low_shares = (int(raw_shares / lot_size)) * lot_size
    high_shares = low_shares + lot_size
    
    if low_shares == 0:
        return high_shares
        
    # User rule: "按照绝对值更近 ... 的原则"
    if abs(low_shares * price - amount) < abs(high_shares * price - amount):
        return low_shares
    else:
        return high_shares

def calculate_fees(amount, is_sell=False):
    comm = amount * COMMISSION_RATE
    if comm < MIN_COMMISSION:
        comm = MIN_COMMISSION
    
    total_fee = comm
    if is_sell:
        total_fee += amount * STAMP_DUTY_RATE
    return total_fee

class GridBacktester:
    def __init__(self, stocks, initial_cash):
        self.stocks = stocks
        self.cash = initial_cash
        self.portfolio = {s: 0 for s in stocks}
        self.base_p = {s: 0.0 for s in stocks}
        self.history = []

    def run(self, df_merged):
        dates = df_merged.index.unique().sort_values()
        
        # Start only when we have data for all 6 (Wait for 601658)
        # PSBC (601658) listing date is 2019-12-10
        start_date = pd.to_datetime("2019-12-10")
        dates = [d for d in dates if d >= start_date]

        # Initial Buy on first day
        first_day = dates[0]
        row_first = df_merged.loc[first_day]
        target_per_stock = self.cash / 6
        
        print(f"--- Initialization on {first_day.date()} ---")
        for s in self.stocks:
            price = row_first[f"{s}_Close"]
            if np.isnan(price): continue
            
            shares = get_closest_shares(target_per_stock, price)
            cost = shares * price
            fees = calculate_fees(cost, is_sell=False)
            
            self.portfolio[s] = shares
            self.cash -= (cost + fees)
            self.base_p[s] = price
            # print(f"Bought {s} ({STOCKS[s]}): {shares} shares @ {price:.2f}, base P set.")

        for date in dates:
            row = df_merged.loc[date]
            
            # 1. Dividend Handling (Check if DivCash > 0)
            for s in self.stocks:
                div = row.get(f"{s}_Dividends", 0)
                if not np.isnan(div) and div > 0:
                    dividend_received = self.portfolio[s] * div
                    self.cash += dividend_received
                    self.base_p[s] -= div
                    # print(f"[{date.date()}] {s} Dividend: received {dividend_received:.2f}, P adjusted to {self.base_p[s]:.3f}")

            # 2. Grid Trading Logic (Intraday checks)
            for s in self.stocks:
                if self.portfolio[s] == 0: continue
                
                low = row[f"{s}_Low"]
                high = row[f"{s}_High"]
                open_p = row[f"{s}_Open"]
                
                # Buy Trigger (Price <= 0.95 * P)
                loop_count = 0
                while low <= self.base_p[s] * 0.95 and loop_count < 5: # Limit loops to prevent infinite if price crashes
                    loop_count += 1
                    trigger_price = self.base_p[s] * 0.95
                    # If open is already lower, we execute at open
                    exec_price = min(open_p, trigger_price)
                    
                    buy_shares = round(self.portfolio[s] * 0.2 / 100) * 100
                    if buy_shares < 100: buy_shares = 0
                    
                    cost = buy_shares * exec_price
                    fees = calculate_fees(cost, is_sell=False)
                    
                    if buy_shares > 0 and self.cash >= (cost + fees):
                        self.portfolio[s] += buy_shares
                        self.cash -= (cost + fees)
                        self.base_p[s] = exec_price
                        # print(f"[{date.date()}] GRID BUY {s}: {buy_shares} @ {exec_price:.3f}, P update.")
                    else:
                        break # No more cash or shares too small
                
                # Sell Trigger (Price >= 1.05 * P)
                loop_count = 0
                while high >= self.base_p[s] * 1.05 and loop_count < 5:
                    loop_count += 1
                    trigger_price = self.base_p[s] * 1.05
                    exec_price = max(open_p, trigger_price)
                    
                    sell_shares = round(self.portfolio[s] * 0.2 / 100) * 100
                    if sell_shares < 100: sell_shares = 0
                    
                    if sell_shares > 0 and self.portfolio[s] >= sell_shares:
                        proceeds = sell_shares * exec_price
                        fees = calculate_fees(proceeds, is_sell=True)
                        self.portfolio[s] -= sell_shares
                        self.cash += (proceeds - fees)
                        self.base_p[s] = exec_price
                        # print(f"[{date.date()}] GRID SELL {s}: {sell_shares} @ {exec_price:.3f}, P update.")
                    else:
                        break

            # 3. Quarterly Rebalancing
            # Check if tomorrow is a new quarter or if today is last trading day
            is_quarter_end = False
            # Simple check: is today one of 3-31, 6-30, 9-30, 12-31? 
            # Better: check if it's the last day in dates list for this quarter
            curr_q = (date.month - 1) // 3
            next_idx = dates.index(date) + 1
            if next_idx < len(dates):
                next_date = dates[next_idx]
                next_q = (next_date.month - 1) // 3
                if curr_q != next_q:
                    is_quarter_end = True
            else:
                is_quarter_end = True # Last day of data
            
            if is_quarter_end:
                # Calculate Total Value
                total_value = self.cash
                stock_prices = {}
                for s in self.stocks:
                    p = row[f"{s}_Close"]
                    total_value += self.portfolio[s] * p
                    stock_prices[s] = p
                
                target_per_stock = total_value / 6
                # print(f"--- Quarterly Rebalance on {date.date()} | Total: {total_value:.2f} ---")
                
                for s in self.stocks:
                    price = stock_prices[s]
                    target_shares = get_closest_shares(target_per_stock, price)
                    diff = target_shares - self.portfolio[s]
                    
                    if diff > 0: # Buy
                        cost = diff * price
                        fees = calculate_fees(cost, is_sell=False)
                        if self.cash >= (cost + fees):
                            self.portfolio[s] += diff
                            self.cash -= (cost + fees)
                        else:
                            # Limited buy with remaining cash
                            pass 
                    elif diff < 0: # Sell
                        abs_diff = abs(diff)
                        proceeds = abs_diff * price
                        fees = calculate_fees(proceeds, is_sell=True)
                        self.portfolio[s] -= abs_diff
                        self.cash += (proceeds - fees)
                    
                    # reset P to close price as per user request
                    self.base_p[s] = price

            # Record Daily Value
            day_value = self.cash
            for s in self.stocks:
                day_value += self.portfolio[s] * row[f"{s}_Close"]
            self.history.append({'Date': date, 'Value': day_value})

        return pd.DataFrame(self.history).set_index('Date')

def load_all_data():
    dfs = []
    for s in STOCKS:
        path = os.path.join(DATA_DIR, f"{s}_raw.csv")
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        # Standardize columns
        df = df[['Open', 'High', 'Low', 'Close', 'Dividends']]
        df.columns = [f"{s}_{c}" for c in df.columns]
        dfs.append(df)
    
    # Merge on Date. Resulting dataframe will have all bank prices daily.
    # We use 'outer' join and then ffill/bfill to handle PSBC listing late.
    merged = pd.concat(dfs, axis=1, join='outer').sort_index()
    # Fill missing values for the days before listing or if tiny gaps
    merged = merged.ffill().bfill() 
    return merged

def calculate_returns(history):
    history['Year'] = history.index.year
    results = []
    
    # Yearly Returns
    years = sorted(history['Year'].unique())
    for y in years:
        y_data = history[history['Year'] == y]
        # Start value (end of prev year or first available)
        prev_year_end = history[history.index < pd.to_datetime(f"{y}-01-01")]
        start_val = prev_year_end['Value'].iloc[-1] if not prev_year_end.empty else history['Value'].iloc[0]
        end_val = y_data['Value'].iloc[-1]
        ret = (end_val / start_val - 1) * 100
        results.append({'Period': f"{y} 年度", 'Return (%)': f"{ret:.2f}%", 'Val': ret})

    # Rolling Periods
    def get_rolling(n_years):
        rolling_res = []
        for i in range(len(years) - n_years + 1):
            start_y = years[i]
            end_y = years[i + n_years - 1]
            
            # Start of period
            prev_year_start = history[history.index < pd.to_datetime(f"{start_y}-01-01")]
            start_v = prev_year_start['Value'].iloc[-1] if not prev_year_start.empty else history['Value'].iloc[0]
            # End of period
            end_v = history[history['Year'] == end_y]['Value'].iloc[-1]
            
            # Cumulative return
            ret = (end_v / start_v - 1) * 100
            rolling_res.append({'Period': f"{start_y}-{end_y} ({n_years}年期)", 'Return (%)': f"{ret:.2f}%", 'Val': ret})
        return rolling_res

    results.extend(get_rolling(2))
    results.extend(get_rolling(3))
    results.extend(get_rolling(5))
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    data = load_all_data()
    backtester = GridBacktester(STOCKS.keys(), INITIAL_CASH)
    history_df = backtester.run(data)
    
    # Print summary
    final_val = history_df['Value'].iloc[-1]
    print(f"\nFinal Portfolio Value: {final_val:.2f}")
    print(f"Total Return: {(final_val / INITIAL_CASH - 1)*100:.2f}%")
    
    returns_table = calculate_returns(history_df)
    print("\n--- Return Report ---")
    print(returns_table[['Period', 'Return (%)']].to_string(index=False))
    
    # Save results to csv for reporting
    returns_table.to_csv("backtests/rolling_returns.csv", index=False)
    history_df.to_csv("backtests/equity_curve.csv")
