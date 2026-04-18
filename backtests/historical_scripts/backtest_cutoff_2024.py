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

def backtest_realistic(init_cap, s_date, e_date, data, symbols, common_dates, use_grid=False):
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
    
    navs = [init_cap]
    for d in dates[1:]:
        for sym in symbols:
            div = data[sym].loc[d, 'DivCash']
            if div > 0:
                cash += shares[sym] * div
                p_base[sym] -= div
        
        if use_grid:
            for sym in symbols:
                lo, hi, op = data[sym].loc[d, 'Low'], data[sym].loc[d, 'High'], data[sym].loc[d, 'Open']
                if lo <= p_base[sym] * 0.95:
                    exec_p = min(op, p_base[sym] * 0.95)
                    cur_tot = sum(shares[s]*data[s].loc[d,'Close'] for s in symbols) + cash
                    target_grid = (cur_tot/num) * 0.20
                    buy_lots = int(target_grid // exec_p // 100)
                    if buy_lots > 0:
                        buy_q = buy_lots * 100
                        cost = buy_q * exec_p
                        fee = get_buy_fee(cost)
                        if cash >= (cost + fee):
                            cash -= (cost + fee)
                            shares[sym] += buy_q
                            p_base[sym] = exec_p
                elif hi >= p_base[sym] * 1.05:
                    exec_p = max(op, p_base[sym] * 1.05)
                    cur_tot = sum(shares[s]*data[s].loc[d,'Close'] for s in symbols) + cash
                    target_grid = (cur_tot/num) * 0.20
                    sell_lots = int(target_grid // exec_p // 100)
                    if sell_lots > 0 and shares[sym] >= sell_lots * 100:
                        val = sell_lots * 100 * exec_p
                        fee = get_sell_fee(val)
                        cash += (val - fee)
                        shares[sym] -= sell_lots * 100
                        p_base[sym] = exec_p

        if d in reb_quarter_ends:
            v_all = {s: shares[s] * data[s].loc[d, 'Close'] for s in symbols}
            tot = cash + sum(v_all.values())
            target_p = tot / num
            for s in symbols:
                if v_all[s] > target_p:
                    diff = v_all[s] - target_p
                    sq = int(diff // data[s].loc[d, 'Close'] // 100) * 100
                    if sq >= 100:
                        v = sq * data[s].loc[d, 'Close']
                        cash += (v - get_sell_fee(v))
                        shares[s] -= sq
                        v_all[s] -= v
            for s in symbols:
                if v_all[s] < target_p:
                    diff = target_p - v_all[s]
                    bq = int(diff // (data[s].loc[d, 'Close'] * 1.0000854) // 100) * 100
                    if bq >= 100:
                        cost = bq * data[s].loc[d, 'Close']
                        fee = get_buy_fee(cost)
                        if cash >= (cost + fee):
                            cash -= (cost + fee)
                            shares[s] += bq
            for s in symbols: p_base[s] = data[s].loc[d, 'Close']
            
        navs.append(cash + sum(shares[s] * data[s].loc[d, 'Close'] for s in symbols))
    
    return navs[-1] / init_cap - 1, get_mdd(navs)

data_6, dates_6 = load_data(symbols_6)
start_date = dates_6[0]
end_date = pd.Timestamp('2024-06-01')

print(f"回测区间: {start_date.date()} 至 {end_date.date()}")
init_cap = 30000.0

ret_p, mdd_p = backtest_realistic(init_cap, start_date, end_date, data_6, symbols_6, dates_6, use_grid=False)
ret_g, mdd_g = backtest_realistic(init_cap, start_date, end_date, data_6, symbols_6, dates_6, use_grid=True)

print("| 策略名称 | 累计收益率 | 最大回撤 | 实际净赚 |")
print("|---|---|---|---|")
print(f"| 实盘-纯季度平准 | {ret_p*100:.2f}% | {mdd_p*100:.2f}% | {int(30000*ret_p)} 元 |")
print(f"| 实盘-平准+5%网格 | {ret_g*100:.2f}% | {mdd_g*100:.2f}% | {int(30000*ret_g)} 元 |")
