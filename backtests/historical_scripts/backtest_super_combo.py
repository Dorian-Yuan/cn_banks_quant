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
    df['Ret'] = 0.0
    df.loc[1:, 'Ret'] = (df['Close'].values[1:] + df['DivCash'].values[1:] - df['Close'].values[:-1]) / df['Close'].values[:-1]
    df['AdjFactor'] = (1.0 + df['Ret']).cumprod()
    df['AdjClose'] = df['Close'].iloc[0] * df['AdjFactor']
    df.set_index('Date', inplace=True)
    data[sym] = df

common_dates = sorted(list(set(data['601328'].index).intersection(*[set(data[sym].index) for sym in symbols])))

# Equal Weight MA250
eq_adj_close = pd.Series(0.0, index=common_dates)
for sym in symbols:
    eq_adj_close += data[sym].loc[common_dates, 'AdjClose']
eq_adj_close /= len(symbols)
eq_ma250 = eq_adj_close.rolling(250).mean()

df_dates = pd.DataFrame({'Date': common_dates})
df_dates['YM'] = df_dates['Date'].dt.to_period('M')
reb_end_dates = set(df_dates.groupby('YM')['Date'].max())

max_date = common_dates[-1]
start_dates = {
    '1Y': max_date - relativedelta(years=1),
    '2Y': max_date - relativedelta(years=2),
    '3Y': max_date - relativedelta(years=3),
    '5Y': max_date - relativedelta(years=5)
}

def get_perf_metrics(navs):
    if not len(navs) > 1: return 0.0
    arr = np.array(navs)
    peak = arr[0]
    mdd = 0.0
    for n in arr:
        if n > peak: peak = n
        dd = 1 - n / peak
        if dd > mdd: mdd = dd
    return mdd

def backtest_super_combo(s_date, e_date, step):
    dates = [d for d in common_dates if s_date <= d <= e_date]
    if not dates: return 0.0, 0.0
    
    cash = 25000.0
    portfolio_shares = {sym: 0.0 for sym in symbols}
    p_base = {sym: 0.0 for sym in symbols}
    portion_shares = {sym: 0.0 for sym in symbols}
    
    d0 = dates[0]
    div_yields_0 = {sym: data[sym].loc[d0, 'DividendYieldTTM'] for sym in symbols}
    top2_initial = [x[0] for x in sorted(div_yields_0.items(), key=lambda x: x[1], reverse=True)[:2]]
    
    for sym in top2_initial:
        fee = get_buy_fee(12500)
        portfolio_shares[sym] = (12500 - fee) / data[sym].loc[d0, 'Close']
        p_base[sym] = data[sym].loc[d0, 'Close']
        portion_shares[sym] = 2500.0 / p_base[sym]
        
    navs = [50000.0]
    
    for d in dates[1:]:
        # 1. Dividend tracking
        for sym in symbols:
            if portfolio_shares[sym] > 0:
                div = data[sym].loc[d, 'DivCash']
                if div > 0:
                    cash += portfolio_shares[sym] * div
                    p_base[sym] -= div
        
        # 2. MA250 Ratio
        adj = eq_adj_close[d]
        ma = eq_ma250[d]
        ratio = adj / ma if pd.notna(ma) and ma > 0 else 1.0
        
        # Determine dynamic buy multiplier based on MA250
        buy_mult = 1.0
        if ratio > 1.15: buy_mult = 0.0
        elif 1.05 < ratio <= 1.15: buy_mult = 0.5
        elif 0.95 <= ratio <= 1.05: buy_mult = 1.0
        elif 0.90 <= ratio < 0.95: buy_mult = 1.5
        elif ratio < 0.90: buy_mult = 2.0
            
        # 3. Grid Execution
        for sym in symbols:
            if portfolio_shares[sym] > 0:
                low = data[sym].loc[d, 'Low']
                high = data[sym].loc[d, 'High']
                open_p = data[sym].loc[d, 'Open']
                
                if low <= p_base[sym] * (1 - step) and buy_mult > 0:
                    exec_p = min(open_p, p_base[sym] * (1 - step))
                    buy_val = 2500.0 * buy_mult
                    fee = get_buy_fee(buy_val)
                    if cash >= buy_val + fee:
                        cash -= (buy_val + fee)
                        portfolio_shares[sym] += buy_val / exec_p
                        p_base[sym] = exec_p
                elif high >= p_base[sym] * (1 + step):
                    exec_p = max(open_p, p_base[sym] * (1 + step))
                    sell_s = portion_shares[sym]
                    if portfolio_shares[sym] >= sell_s:
                        val = sell_s * exec_p
                        fee = get_sell_fee(val)
                        cash += (val - fee)
                        portfolio_shares[sym] -= sell_s
                        p_base[sym] = exec_p
                        
        # 4. Rotation (End of Month)
        if d in reb_end_dates:
            dy = {s: data[s].loc[d, 'DividendYieldTTM'] for s in symbols}
            ranked = [x[0] for x in sorted(dy.items(), key=lambda x: x[1], reverse=True)]
            held_syms = [s for s in symbols if portfolio_shares[s] > 0]
            
            for sym in held_syms:
                if sym not in ranked[:3]:
                    # Sell out
                    sell_val = portfolio_shares[sym] * data[sym].loc[d, 'Close']
                    fee = get_sell_fee(sell_val)
                    cash += (sell_val - fee)
                    portfolio_shares[sym] = 0
                    p_base[sym] = 0
                    portion_shares[sym] = 0
                    
                    # Buy missing top 2
                    cur_held = [s for s in symbols if portfolio_shares[s] > 0]
                    target = None
                    for ts in ranked[:2]:
                        if ts not in cur_held:
                            target = ts
                            break
                    
                    if target is not None:
                        # use Exact Proceeds
                        proceeds = sell_val - fee
                        if proceeds > 0:
                            buy_fee = get_buy_fee(proceeds)
                            if cash >= proceeds: # Sanity check
                                cash -= proceeds
                            else: # If we are broke for some reason, use all cash
                                proceeds = cash
                                cash = 0
                                buy_fee = get_buy_fee(proceeds)
                            
                            actual_buy = proceeds - buy_fee
                            portfolio_shares[target] = actual_buy / data[target].loc[d, 'Close']
                            p_base[target] = data[target].loc[d, 'Close']
                            portion_shares[target] = 2500.0 / p_base[target]
                            
        # Eval NAV (No external cash flow so pure NAV works)
        cur_val = cash + sum(portfolio_shares[sym] * data[sym].loc[d, 'Close'] for sym in symbols)
        navs.append(cur_val)
        
    mdd = get_perf_metrics(navs)
    return navs[-1] / 50000 - 1, mdd

periods = ['1Y', '2Y', '3Y', '5Y']
steps = [0.01, 0.02, 0.03, 0.04, 0.05]
res_table = {s: {} for s in steps}

for p in periods:
    sd = start_dates[p]
    ed = common_dates[-1]
    for s in steps:
        ret, mdd = backtest_super_combo(sd, ed, s)
        res_table[s][p] = ret
        if p == '5Y': res_table[s]['MDD'] = mdd

import textwrap
print("| 网格步长 | 近1年累计收益率 | 近2年累计收益率 | 近3年累计收益率 | 近5年累计收益率 | 最大回撤(近5年) |")
print("|---|---|---|---|---|---|")
for s in steps:
    row = f"| {int(s*100)}% | {res_table[s]['1Y']*100:.2f}% | {res_table[s]['2Y']*100:.2f}% | {res_table[s]['3Y']*100:.2f}% | {res_table[s]['5Y']*100:.2f}% | {res_table[s]['MDD']*100:.2f}% |"
    print(row)
