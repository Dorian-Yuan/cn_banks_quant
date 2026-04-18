import pandas as pd
import numpy as np
import os
from dateutil.relativedelta import relativedelta

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "ashare")
symbols_6 = ['601398', '601288', '601988', '601939', '601328', '601658']

def get_buy_fee(amount):
    return max(amount * 0.0000854, 0.5)

def get_sell_fee(amount):
    return max(amount * 0.0000854, 0.5) + amount * 0.0005

def load_data(symbols):
    data = {}
    for sym in symbols:
        df = pd.read_csv(os.path.join(data_dir, f"{sym}.csv"))
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df.set_index('Date', inplace=True)
        data[sym] = df
    common_dates = sorted(list(set(data[symbols[0]].index).intersection(*[set(data[sym].index) for sym in symbols])))
    return data, common_dates

def get_mdd(navs):
    if not len(navs) > 1: return 0.0
    arr = np.array(navs)
    peak = arr[0]
    mdd = 0.0
    for n in arr:
        if n > peak: peak = n
        dd = 1 - n / peak
        if dd > mdd: mdd = dd
    return mdd

def backtest_periodic(init_cap, s_date, e_date, data, symbols, common_dates):
    dates = [d for d in common_dates if s_date <= d <= e_date]
    if not dates: return 0.0, 0.0
    
    df_dates = pd.DataFrame({'Date': dates})
    df_dates['Q'] = df_dates['Date'].dt.to_period('Q')
    reb_quarter_ends = set(df_dates.groupby('Q')['Date'].max())
    
    cash = 0.0
    shares = {sym: 0.0 for sym in symbols}
    d0 = dates[0]
    num = len(symbols)
    target = init_cap / num
    for sym in symbols:
        fee = get_buy_fee(target)
        shares[sym] = (target - fee) / data[sym].loc[d0, 'Close']
        
    navs = [init_cap]
    for d in dates[1:]:
        for sym in symbols:
            div = data[sym].loc[d, 'DivCash']
            if div > 0: cash += shares[sym] * div
        if d in reb_quarter_ends:
            vals = {s: shares[s] * data[s].loc[d, 'Close'] for s in symbols}
            target_p = (cash + sum(vals.values())) / num
            for s in symbols:
                if vals[s] > target_p:
                    diff = vals[s] - target_p
                    fee = get_sell_fee(diff)
                    cash += (diff - fee)
                    shares[s] -= diff / data[s].loc[d, 'Close']
            for s in symbols:
                val = shares[s] * data[s].loc[d, 'Close']
                if val < target_p:
                    diff = target_p - val
                    buy = min(diff, cash)
                    if buy > 0.5:
                        fee = get_buy_fee(buy)
                        cash -= buy
                        shares[s] += (buy - fee) / data[s].loc[d, 'Close']
        navs.append(cash + sum(shares[s] * data[s].loc[d, 'Close'] for s in symbols))
    return navs[-1] / init_cap - 1, get_mdd(navs)

def backtest_full_grid_sweep(init_cap, s_date, e_date, data, symbols, common_dates, grid_ratio, step=0.05):
    dates = [d for d in common_dates if s_date <= d <= e_date]
    if not dates: return 0.0, 0.0
    
    df_dates = pd.DataFrame({'Date': dates})
    df_dates['Q'] = df_dates['Date'].dt.to_period('Q')
    reb_quarter_ends = set(df_dates.groupby('Q')['Date'].max())
    
    cash = 0.0
    shares, p_base = {sym: 0.0 for sym in symbols}, {sym: 0.0 for sym in symbols}
    d0 = dates[0]
    num = len(symbols)
    target = init_cap / num
    for sym in symbols:
        fee = get_buy_fee(target)
        shares[sym] = (target - fee) / data[sym].loc[d0, 'Close']
        p_base[sym] = data[sym].loc[d0, 'Close']
    
    # grid_ratio is the % of position traded, e.g., 0.10, 0.15, etc.
    grid_val = target * grid_ratio
    portion_shares = {s: grid_val / p_base[s] for s in symbols}
    navs = [init_cap]
    
    for d in dates[1:]:
        for s in symbols:
            div = data[s].loc[d, 'DivCash']
            if div > 0:
                cash += shares[s] * div
                p_base[s] -= div
        for s in symbols:
            lo, hi, op = data[s].loc[d, 'Low'], data[s].loc[d, 'High'], data[s].loc[d, 'Open']
            if lo <= p_base[s] * (1 - step):
                exec_p = min(op, p_base[s] * (1 - step))
                fee = get_buy_fee(grid_val)
                if cash >= grid_val + fee:
                    cash -= (grid_val + fee)
                    shares[s] += grid_val / exec_p
                    p_base[s] = exec_p
            elif hi >= p_base[s] * (1 + step):
                exec_p = max(op, p_base[s] * (1 + step))
                if shares[s] >= portion_shares[s]:
                    v = portion_shares[s] * exec_p
                    fee = get_sell_fee(v)
                    cash += (v - fee)
                    shares[s] -= portion_shares[s]
                    p_base[s] = exec_p
        if d in reb_quarter_ends:
            vals = {s: shares[s] * data[s].loc[d, 'Close'] for s in symbols}
            tot = cash + sum(vals.values())
            target_p = tot / num
            for s in symbols:
                if vals[s] > target_p:
                    diff = vals[s] - target_p
                    fee = get_sell_fee(diff)
                    cash += (diff - fee)
                    shares[s] -= diff / data[s].loc[d, 'Close']
            for s in symbols:
                val = shares[s] * data[s].loc[d, 'Close']
                if val < target_p:
                    diff = target_p - val
                    buy = min(diff, cash)
                    if buy > 0.5:
                        fee = get_buy_fee(buy)
                        cash -= buy
                        shares[s] += (buy - fee) / data[s].loc[d, 'Close']
            # Update grid size based on new value
            grid_val = target_p * grid_ratio
            for s in symbols:
                p_base[s] = data[s].loc[d, 'Close']
                portion_shares[s] = grid_val / p_base[s]
        navs.append(cash + sum(shares[s] * data[s].loc[d, 'Close'] for s in symbols))
    return navs[-1] / init_cap - 1, get_mdd(navs)

data_6, dates_6 = load_data(symbols_6)
max_date = dates_6[-1]
horizons = ['1Y', '2Y', '3Y', '5Y']
grid_ratios = [0.10, 0.15, 0.20, 0.25]
init_cap = 30000.0

print("| 策略方案 | 1Y收益 | 2Y收益 | 3Y收益 | 5Y收益 | 5Y最大回撤 |")
print("|---|---|---|---|---|---|")

# Baseline
rets_bl = {}
mdd_5y_bl = 0
for h in horizons:
    sd = max_date - relativedelta(years=int(h[0]))
    r, m = backtest_periodic(init_cap, sd, max_date, data_6, symbols_6, dates_6)
    rets_bl[h] = r
    if h == '5Y': mdd_5y_bl = m
print(f"| 纯季度平准(对照) | {rets_bl['1Y']*100:.2f}% | {rets_bl['2Y']*100:.2f}% | {rets_bl['3Y']*100:.2f}% | {rets_bl['5Y']*100:.2f}% | {mdd_5y_bl*100:.2f}% |")

# Grid Sizes Sweep
for ratio in grid_ratios:
    rets_grid = {}
    mdd_5y_grid = 0
    for h in horizons:
        sd = max_date - relativedelta(years=int(h[0]))
        r, m = backtest_full_grid_sweep(init_cap, sd, max_date, data_6, symbols_6, dates_6, ratio)
        rets_grid[h] = r
        if h == '5Y': mdd_5y_grid = m
    print(f"| 5%网格+每次成交{int(ratio*100)}%仓位 | {rets_grid['1Y']*100:.2f}% | {rets_grid['2Y']*100:.2f}% | {rets_grid['3Y']*100:.2f}% | {rets_grid['5Y']*100:.2f}% | {mdd_5y_grid*100:.2f}% |")
