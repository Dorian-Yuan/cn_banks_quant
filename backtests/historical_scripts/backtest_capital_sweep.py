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
s_date = max_date - relativedelta(years=5)
e_date = common_dates[-1]

def backtest_periodic(init_cap):
    dates = [d for d in common_dates if s_date <= d <= e_date]
    cash = 0.0
    shares = {sym: 0.0 for sym in symbols}
    d0 = dates[0]
    
    target = init_cap / 5
    for sym in symbols:
        fee = get_buy_fee(target)
        shares[sym] = (target - fee) / data[sym].loc[d0, 'Close']
        
    navs = [init_cap]
    
    for d in dates[1:]:
        for sym in symbols:
            div = data[sym].loc[d, 'DivCash']
            if div > 0: cash += shares[sym] * div
                
        if d in reb_quarter_ends:
            val_map = {sym: shares[sym] * data[sym].loc[d, 'Close'] for sym in symbols}
            target_amount = (cash + sum(val_map.values())) * 0.2
            for sym in symbols:
                if val_map[sym] > target_amount:
                    excess = val_map[sym] - target_amount
                    fee = get_sell_fee(excess)
                    cash += (excess - fee)
                    shares[sym] -= excess / data[sym].loc[d, 'Close']
            for sym in symbols:
                val = shares[sym] * data[sym].loc[d, 'Close']
                if val < target_amount:
                    deficit = target_amount - val
                    actual_buy = deficit if cash >= deficit else cash
                    if actual_buy > 1:
                        fee = get_buy_fee(actual_buy)
                        cash -= actual_buy
                        shares[sym] += (actual_buy - fee) / data[sym].loc[d, 'Close']
                        
        cur_val = cash + sum(shares[sym] * data[sym].loc[d, 'Close'] for sym in symbols)
        navs.append(cur_val)
        
    return navs[-1] / init_cap - 1

def backtest_full_grid_reb(init_cap, step=0.05):
    dates = [d for d in common_dates if s_date <= d <= e_date]
    cash = 0.0
    shares = {sym: 0.0 for sym in symbols}
    p_base = {sym: 0.0 for sym in symbols}
    portion_shares = {sym: 0.0 for sym in symbols}
    
    d0 = dates[0]
    target = init_cap / 5
    for sym in symbols:
        fee = get_buy_fee(target)
        shares[sym] = (target - fee) / data[sym].loc[d0, 'Close']
        p_base[sym] = data[sym].loc[d0, 'Close']
        
    grid_value = target * 0.20
    for sym in symbols:
        portion_shares[sym] = grid_value / p_base[sym]
        
    navs = [init_cap]
    
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
                
        navs.append(cash + sum(shares[sym] * data[sym].loc[d, 'Close'] for sym in symbols))
        
    return navs[-1] / init_cap - 1

caps = [30000.0, 40000.0, 50000.0, 60000.0, 70000.0]
print("| 起始本金 | 纯满仓季度平准(5年收益) | 满仓平准+5%网格(5年收益) | 收益落差 |")
print("|---|---|---|---|")
for c in caps:
    r1 = backtest_periodic(c)
    r2 = backtest_full_grid_reb(c, 0.05)
    gap = r2 - r1
    print(f"| {int(c/10000)}w | {r1*100:.2f}% | {r2*100:.2f}% | {'+' if gap>=0 else ''}{gap*100:.2f}% |")
