import pandas as pd
import numpy as np
import os
from dateutil.relativedelta import relativedelta

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "ashare")
symbols_5 = ['601398', '601288', '601988', '601939', '601328']
symbols_6 = symbols_5 + ['601658']

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

def backtest_realistic(init_cap, s_date, e_date, data, symbols, common_dates, use_grid=False, grid_ratio=0.20):
    dates = [d for d in common_dates if s_date <= d <= e_date]
    if not dates: return 0.0, 0.0
    
    df_dates = pd.DataFrame({'Date': dates})
    df_dates['Q'] = df_dates['Date'].dt.to_period('Q')
    reb_quarter_ends = set(df_dates.groupby('Q')['Date'].max())
    
    cash = init_cap
    shares = {sym: 0 for sym in symbols}
    p_base = {sym: 0.0 for sym in symbols}
    
    d0 = dates[0]
    num = len(symbols)
    target_val = init_cap / num
    
    # Initial allocation in 100-share lots
    for sym in symbols:
        price = data[sym].loc[d0, 'Close']
        max_shares = int(target_val // (price * (1 + 0.0000854)) // 100) * 100
        if max_shares > 0:
            cost = max_shares * price
            fee = get_buy_fee(cost)
            if cash >= (cost + fee):
                cash -= (cost + fee)
                shares[sym] = max_shares
                p_base[sym] = price
        else:
            p_base[sym] = price # Still set base price even if we couldn't afford 1 lot
            
    navs = [init_cap]
    
    # Daily loop
    for d in dates[1:]:
        # 1. Dividends
        for sym in symbols:
            div = data[sym].loc[d, 'DivCash']
            if div > 0:
                cash += shares[sym] * div
                p_base[sym] -= div # Dividend adjustment for grid
        
        # 2. Grid (if enabled)
        if use_grid:
            for sym in symbols:
                if shares[sym] == 0 and p_base[sym] == 0: continue
                lo, hi, op = data[sym].loc[d, 'Low'], data[sym].loc[d, 'High'], data[sym].loc[d, 'Open']
                
                # Buy at -5%
                if lo <= p_base[sym] * 0.95:
                    exec_p = min(op, p_base[sym] * 0.95)
                    # Use Option 2: Target 20% of current group-target value
                    cur_tot = sum(shares[s] * data[s].loc[d, 'Close'] for s in symbols) + cash
                    target_grid_val = (cur_tot / num) * grid_ratio
                    
                    buy_lots = int(target_grid_val // exec_p // 100)
                    if buy_lots > 0:
                        buy_q = buy_lots * 100
                        cost = buy_q * exec_p
                        fee = get_buy_fee(cost)
                        if cash >= (cost + fee):
                            cash -= (cost + fee)
                            shares[sym] += buy_q
                            p_base[sym] = exec_p
                
                # Sell at +5%
                elif hi >= p_base[sym] * 1.05:
                    exec_p = max(op, p_base[sym] * 1.05)
                    cur_tot = sum(shares[s] * data[s].loc[d, 'Close'] for s in symbols) + cash
                    target_grid_val = (cur_tot / num) * grid_ratio
                    
                    sell_lots = int(target_grid_val // exec_p // 100)
                    if sell_lots > 0:
                        sell_q = sell_lots * 100
                        if shares[sym] >= sell_q:
                            val = sell_q * exec_p
                            fee = get_sell_fee(val)
                            cash += (val - fee)
                            shares[sym] -= sell_q
                            p_base[sym] = exec_p
        
        # 3. Quarterly Rebalancing
        if d in reb_quarter_ends:
            vals = {s: shares[s] * data[s].loc[d, 'Close'] for s in symbols}
            total = cash + sum(vals.values())
            target_p = total / num
            
            # Sell overweighted (can sell odd lots in rebalancing in A-shares if clearing, 
            # but usually we'll stick to 100 multiples here for consistency)
            for s in symbols:
                if vals[s] > target_p:
                    diff = vals[s] - target_p
                    sell_q = int(diff // data[s].loc[d, 'Close'] // 100) * 100
                    if sell_q >= 100:
                        v = sell_q * data[s].loc[d, 'Close']
                        fee = get_sell_fee(v)
                        cash += (v - fee)
                        shares[s] -= sell_q
                        vals[s] -= v
            
            # Buy underweighted
            for s in symbols:
                if vals[s] < target_p:
                    diff = target_p - vals[s]
                    buy_q = int(diff // (data[s].loc[d, 'Close'] * (1 + 0.0000854)) // 100) * 100
                    if buy_q >= 100:
                        cost = buy_q * data[s].loc[d, 'Close']
                        fee = get_buy_fee(cost)
                        if cash >= (cost + fee):
                            cash -= (cost + fee)
                            shares[s] += buy_q
                            
            # Reset grid base price to current close
            for s in symbols:
                p_base[s] = data[s].loc[d, 'Close']
                
        # Eval NAV
        navs.append(cash + sum(shares[s] * data[s].loc[d, 'Close'] for s in symbols))
        
    return navs[-1] / init_cap - 1, get_mdd(navs)

data_5, dates_5 = load_data(symbols_5)
data_6, dates_6 = load_data(symbols_6)
max_date = dates_6[-1]
horizons = ['1Y', '2Y', '3Y', '5Y']
init_cap = 30000.0

print("| 组合规模 | 策略方案 | 1Y收益 | 2Y收益 | 3Y收益 | 5Y收益 | 5Y最大回撤 |")
print("|---|---|---|---|---|---|---|")

# 5 Banks
print(f"| 五大行 | 理论对标(之前) | 12.53% | 45.80% | 93.49% | 110.89% | 14.62% |") # From previous backtest_3w_horizons
for grid in [False, True]:
    r = {}
    m5y = 0
    for h in horizons:
        sd = max_date - relativedelta(years=int(h[0]))
        ret, mdd = backtest_realistic(init_cap, sd, max_date, data_5, symbols_5, dates_5, use_grid=grid)
        r[h] = ret
        if h == '5Y': m5y = mdd
    name = "实盘-纯平准" if not grid else "实盘-平准+5%网格"
    print(f"| 五大行 | {name} | {r['1Y']*100:.2f}% | {r['2Y']*100:.2f}% | {r['3Y']*100:.2f}% | {r['5Y']*100:.2f}% | {m5y*100:.2f}% |")

# 6 Banks
for grid in [False, True]:
    r = {}
    m5y = 0
    for h in horizons:
        sd = max_date - relativedelta(years=int(h[0]))
        ret, mdd = backtest_realistic(init_cap, sd, max_date, data_6, symbols_6, dates_6, use_grid=grid)
        r[h] = ret
        if h == '5Y': m5y = mdd
    name = "实盘-纯平准" if not grid else "实盘-平准+5%网格"
    print(f"| 六大行 | {name} | {r['1Y']*100:.2f}% | {r['2Y']*100:.2f}% | {r['3Y']*100:.2f}% | {r['5Y']*100:.2f}% | {m5y*100:.2f}% |")
