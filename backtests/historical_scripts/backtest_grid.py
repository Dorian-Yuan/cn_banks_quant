import pandas as pd
import numpy as np
import os
from dateutil.relativedelta import relativedelta

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "ashare")
sym = '601328'  # 交通银行

def get_buy_fee(amount):
    return max(amount * 0.0000854, 0.5)

def get_sell_fee(amount):
    return max(amount * 0.0000854, 0.5) + amount * 0.0005

df = pd.read_csv(os.path.join(data_dir, f"{sym}.csv"))
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df.set_index('Date', inplace=True)

common_dates = df.index.tolist()
max_date = common_dates[-1]
start_dates = {
    '1Y': max_date - relativedelta(years=1),
    '2Y': max_date - relativedelta(years=2),
    '3Y': max_date - relativedelta(years=3),
    '5Y': max_date - relativedelta(years=5)
}

def get_mdd(navs):
    if not navs: return 0.0
    peak = navs[0]
    mdd = 0.0
    for n in navs:
        if n > peak: peak = n
        dd = 1 - n / peak
        if dd > mdd: mdd = dd
    return mdd

def backtest_grid(s_date, e_date, step):
    dates = [d for d in common_dates if s_date <= d <= e_date]
    if not dates: return 0.0, 0.0
    
    p_0 = df.loc[dates[0], 'Close']
    share_val = 25000.0
    buy_fee = get_buy_fee(share_val)
    cash = 50000.0 - share_val - buy_fee
    shares = share_val / p_0
    
    p_base = p_0
    portion_shares = 2500.0 / p_0
    navs = [cash + shares * p_0]
    
    for d in dates[1:]:
        div = df.loc[d, 'DivCash']
        if div > 0:
            cash += shares * div
            p_base -= div  # 扣除分红除息影响，防止发生除息假突破
            
        low = df.loc[d, 'Low']
        high = df.loc[d, 'High']
        open_p = df.loc[d, 'Open']
        close_p = df.loc[d, 'Close']
        
        if low <= p_base * (1 - step):
            exec_p = min(open_p, p_base * (1 - step))
            buy_val = 2500.0
            fee = get_buy_fee(buy_val)
            if cash >= buy_val + fee:
                cash -= (buy_val + fee)
                shares += buy_val / exec_p
                p_base = exec_p
        elif high >= p_base * (1 + step):
            exec_p = max(open_p, p_base * (1 + step))
            if shares >= portion_shares:
                val = portion_shares * exec_p
                fee = get_sell_fee(val)
                cash += val - fee
                shares -= portion_shares
                p_base = exec_p
                
        navs.append(cash + shares * close_p)
        
    return navs[-1] / 50000 - 1, get_mdd(navs)

periods = ['1Y', '2Y', '3Y', '5Y']
steps = [0.01, 0.02, 0.03, 0.04, 0.05]
results = {s: {} for s in steps}

for p in periods:
    sd = start_dates[p]
    for s in steps:
        ret, mdd = backtest_grid(sd, max_date, s)
        results[s][p] = ret
        if p == '5Y':
            results[s]['MDD'] = mdd

print("| 网格步长 | 近1年累计收益率 | 近2年累计收益率 | 近3年累计收益率 | 近5年累计收益率 | 最大回撤(近5年) |")
print("|---|---|---|---|---|---|")
for s in steps:
    row = f"| {int(s*100)}% | {results[s]['1Y']*100:.2f}% | {results[s]['2Y']*100:.2f}% | {results[s]['3Y']*100:.2f}% | {results[s]['5Y']*100:.2f}% | {results[s]['MDD']*100:.2f}% |"
    print(row)
