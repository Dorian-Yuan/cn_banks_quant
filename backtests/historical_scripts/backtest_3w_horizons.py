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

def backtest_periodic(init_cap, s_date, e_date):
    dates = [d for d in common_dates if s_date <= d <= e_date]
    if not dates: return 0.0, 0.0
    cash, shares = 0.0, {sym: 0.0 for sym in symbols}
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
            vals = {s: shares[s] * data[s].loc[d, 'Close'] for s in symbols}
            target_p = (cash + sum(vals.values())) * 0.2
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

def backtest_full_grid(init_cap, s_date, e_date, step=0.05):
    dates = [d for d in common_dates if s_date <= d <= e_date]
    if not dates: return 0.0, 0.0
    cash, shares, p_base = 0.0, {sym: 0.0 for sym in symbols}, {sym: 0.0 for sym in symbols}
    d0 = dates[0]
    target = init_cap / 5
    for sym in symbols:
        fee = get_buy_fee(target)
        shares[sym] = (target - fee) / data[sym].loc[d0, 'Close']
        p_base[sym] = data[sym].loc[d0, 'Close']
    
    grid_val = target * 0.2
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
            target_p = tot / 5.0
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
            grid_val = target_p * 0.2
            for s in symbols:
                p_base[s] = data[s].loc[d, 'Close']
                portion_shares[s] = grid_val / p_base[s]
        navs.append(cash + sum(shares[s] * data[s].loc[d, 'Close'] for s in symbols))
    return navs[-1] / init_cap - 1, get_mdd(navs)

max_date = common_dates[-1]
horizons = ['1Y', '2Y', '3Y', '5Y']
init_cap = 30000.0

print("| 时间跨度 | 纯季度平准(收益) | 纯季度平准(回撤) | 平准+5%网格(收益) | 平准+5%网格(回撤) |")
print("|---|---|---|---|---|")
for h in horizons:
    sd = max_date - relativedelta(years=int(h[0]))
    r1, m1 = backtest_periodic(init_cap, sd, max_date)
    r2, m2 = backtest_full_grid(init_cap, sd, max_date, 0.05)
    print(f"| {h} | {r1*100:.2f}% | {m1*100:.2f}% | {r2*100:.2f}% | {m2*100:.2f}% |")
