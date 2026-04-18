import pandas as pd
import numpy as np
import os
from dateutil.relativedelta import relativedelta

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "ashare")
symbols = ['601398', '601288', '601988', '601939', '601328']

def get_buy_fee(amount):
    return max(amount * 0.0000854, 0.5)

def get_sell_fee(amount):
    return max(amount * 0.0000854, 0.5) + amount * 0.0005

data = {}
for sym in symbols:
    df = pd.read_csv(os.path.join(data_dir, f"{sym}.csv"))
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df.set_index('Date', inplace=True)
    data[sym] = df

common_dates = sorted(list(set(data['601328'].index).intersection(*[set(data[sym].index) for sym in symbols])))

df_dates = pd.DataFrame({'Date': common_dates})
df_dates['Q'] = df_dates['Date'].dt.to_period('Q')
reb_quarter_ends = set(df_dates.groupby('Q')['Date'].max())

max_date = common_dates[-1]
start_dates = {
    '1Y': max_date - relativedelta(years=1),
    '2Y': max_date - relativedelta(years=2),
    '3Y': max_date - relativedelta(years=3),
    '5Y': max_date - relativedelta(years=5)
}

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

def backtest_full_grid_reb(s_date, e_date, step):
    dates = [d for d in common_dates if s_date <= d <= e_date]
    if not dates: return 0.0, 0.0
    
    cash = 0.0
    shares = {sym: 0.0 for sym in symbols}
    p_base = {sym: 0.0 for sym in symbols}
    portion_shares = {sym: 0.0 for sym in symbols}
    
    d0 = dates[0]
    target = 50000.0 / 5
    for sym in symbols:
        fee = get_buy_fee(target)
        shares[sym] = (target - fee) / data[sym].loc[d0, 'Close']
        p_base[sym] = data[sym].loc[d0, 'Close']
        
    grid_value = target * 0.20
    for sym in symbols:
        portion_shares[sym] = grid_value / p_base[sym]
        
    navs = [50000.0]
    
    for d in dates[1:]:
        for sym in symbols:
            div = data[sym].loc[d, 'DivCash']
            if div > 0:
                cash += shares[sym] * div
                p_base[sym] -= div
                
        for sym in symbols:
            low = data[sym].loc[d, 'Low']
            high = data[sym].loc[d, 'High']
            open_p = data[sym].loc[d, 'Open']
            
            if low <= p_base[sym] * (1 - step):
                exec_p = min(open_p, p_base[sym] * (1 - step))
                fee = get_buy_fee(grid_value)
                if cash >= grid_value + fee:
                    cash -= (grid_value + fee)
                    shares[sym] += grid_value / exec_p
                    p_base[sym] = exec_p
            elif high >= p_base[sym] * (1 + step):
                exec_p = max(open_p, p_base[sym] * (1 + step))
                sell_s = portion_shares[sym]
                if shares[sym] >= sell_s:
                    val = sell_s * exec_p
                    fee = get_sell_fee(val)
                    cash += (val - fee)
                    shares[sym] -= sell_s
                    p_base[sym] = exec_p
                    
        if d in reb_quarter_ends:
            val_map = {sym: shares[sym] * data[sym].loc[d, 'Close'] for sym in symbols}
            total_val = cash + sum(val_map.values())
            new_target = total_val / 5.0
            
            for sym in symbols:
                if val_map[sym] > new_target:
                    excess = val_map[sym] - new_target
                    fee = get_sell_fee(excess)
                    cash += (excess - fee)
                    shares[sym] -= excess / data[sym].loc[d, 'Close']
                    val_map[sym] = new_target
            
            for sym in symbols:
                val = shares[sym] * data[sym].loc[d, 'Close']
                if val < new_target:
                    deficit = new_target - val
                    actual_buy = deficit if cash >= deficit else cash
                    if actual_buy > 1:
                        fee = get_buy_fee(actual_buy)
                        cash -= actual_buy
                        shares[sym] += (actual_buy - fee) / data[sym].loc[d, 'Close']
            
            grid_value = new_target * 0.20
            for sym in symbols:
                p_base[sym] = data[sym].loc[d, 'Close']
                portion_shares[sym] = grid_value / p_base[sym]
                
        cur_val = cash + sum(shares[sym] * data[sym].loc[d, 'Close'] for sym in symbols)
        navs.append(cur_val)
        
    abs_ret = navs[-1] / 50000 - 1
    mdd = get_mdd(navs)
    return abs_ret, mdd

periods = ['1Y', '2Y', '3Y', '5Y']
steps = [0.01, 0.02, 0.03, 0.04, 0.05]
res_table = {s: {} for s in steps}

for p in periods:
    sd = start_dates[p]
    ed = common_dates[-1]
    for s in steps:
        ret, mdd = backtest_full_grid_reb(sd, ed, s)
        res_table[s][p] = ret
        res_table[s][f'{p}_MDD'] = mdd

print(f"满仓平准+5%网格 1年期 MDD: {res_table[0.05]['1Y_MDD']*100:.2f}%")
