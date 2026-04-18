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

max_date = common_dates[-1]
start_dates = {
    '1Y': max_date - relativedelta(years=1),
    '2Y': max_date - relativedelta(years=2),
    '3Y': max_date - relativedelta(years=3),
    '5Y': max_date - relativedelta(years=5)
}

df_dates = pd.DataFrame({'Date': common_dates})
df_dates['YM'] = df_dates['Date'].dt.to_period('M')
reb_start_dates = set(df_dates.groupby('YM')['Date'].min())
reb_end_dates = set(df_dates.groupby('YM')['Date'].max())

def get_perf_metrics(unit_navs):
    if not len(unit_navs) > 1: return 0.0, 0.0, 0.0
    arr = np.array(unit_navs)
    rets = np.diff(arr) / arr[:-1]
    win_rate = np.mean(rets > 0)
    vol = np.std(rets) * np.sqrt(250)
    
    peak = arr[0]
    mdd = 0.0
    for n in arr:
        if n > peak: peak = n
        dd = 1 - n / peak
        if dd > mdd: mdd = dd
        
    return mdd, win_rate, vol

def backtest_benchmark(s_date, e_date):
    dates = [d for d in common_dates if s_date <= d <= e_date]
    if not dates: return 0.0, 0.0, 0.0, 0.0
    
    cash = 0.0
    shares = {sym: 0.0 for sym in symbols}
    target_amount = 50000 / len(symbols)
    d0 = dates[0]
    for sym in symbols:
        price = data[sym].loc[d0, 'Close']
        fee = get_buy_fee(target_amount)
        shares[sym] = (target_amount - fee) / price
        
    navs = [50000.0]
    for d in dates[1:]:
        for sym in symbols:
            div = data[sym].loc[d, 'DivCash']
            if div > 0: cash += shares[sym] * div
        
        d_val = cash + sum(shares[sym] * data[sym].loc[d, 'Close'] for sym in symbols)
        navs.append(d_val)
        
    mdd, wr, vol = get_perf_metrics(navs)
    return navs[-1] / 50000 - 1, mdd, wr, vol

def backtest_composite(s_date, e_date):
    dates = [d for d in common_dates if s_date <= d <= e_date]
    if not dates: return 0.0, 0.0, 0.0, 0.0
    
    # Init Day 0
    d0 = dates[0]
    div_yields_0 = {sym: data[sym].loc[d0, 'DividendYieldTTM'] for sym in symbols}
    top2_initial = [x[0] for x in sorted(div_yields_0.items(), key=lambda x: x[1], reverse=True)[:2]]
    
    cash = 25000.0
    total_input = 50000.0
    portfolio_shares = {sym: 0.0 for sym in symbols}
    p_base = {sym: 0.0 for sym in symbols}
    
    for sym in top2_initial:
        fee = get_buy_fee(12500)
        actual_buy = 12500 - fee
        portfolio_shares[sym] = actual_buy / data[sym].loc[d0, 'Close']
        p_base[sym] = data[sym].loc[d0, 'Close']
        
    cur_val = cash + sum(portfolio_shares[sym] * data[sym].loc[d0, 'Close'] for sym in top2_initial)
    unit_nav = 1.0
    total_units = total_input / unit_nav
    unit_navs = [unit_nav]
    
    prev_unav = unit_nav
    
    for d in dates[1:]:
        # 0. Dividends processing
        for sym in symbols:
            if portfolio_shares[sym] > 0:
                div = data[sym].loc[d, 'DivCash']
                if div > 0:
                    cash += portfolio_shares[sym] * div
                    p_base[sym] -= div
        
        # 1. Start of Month DCA (module 1)
        if d in reb_start_dates:
            adj = eq_adj_close[d]
            ma = eq_ma250[d]
            ratio = adj / ma if pd.notna(ma) and ma > 0 else 1.0
            
            invest = 0
            if 0.95 <= ratio <= 1.05: invest = 2000
            elif 0.90 <= ratio < 0.95: invest = 3000
            elif ratio < 0.90: invest = 4000
            elif 1.05 < ratio <= 1.15: invest = 1000
            
            if invest > 0:
                total_input += invest
                total_units += invest / prev_unav  # Add units using previous end of day NAV
                cash += invest
                
        # 2. Grid (module 3)
        for sym in symbols:
            if portfolio_shares[sym] > 0:
                low = data[sym].loc[d, 'Low']
                high = data[sym].loc[d, 'High']
                open_p = data[sym].loc[d, 'Open']
                
                if low <= p_base[sym] * 0.95:
                    exec_p = min(open_p, p_base[sym] * 0.95)
                    buy_val = 2500.0
                    fee = get_buy_fee(buy_val)
                    if cash >= buy_val + fee:
                        cash -= (buy_val + fee)
                        portfolio_shares[sym] += buy_val / exec_p
                        p_base[sym] = exec_p
                elif high >= p_base[sym] * 1.05:
                    exec_p = max(open_p, p_base[sym] * 1.05)
                    sell_shares = 2500.0 / p_base[sym]
                    if portfolio_shares[sym] >= sell_shares:
                        sell_val = sell_shares * exec_p
                        fee = get_sell_fee(sell_val)
                        cash += (sell_val - fee)
                        portfolio_shares[sym] -= sell_shares
                        p_base[sym] = exec_p
                        
        # 3. End of Month Rotation (module 2)
        if d in reb_end_dates:
            dy = {s: data[s].loc[d, 'DividendYieldTTM'] for s in symbols}
            ranked = [x[0] for x in sorted(dy.items(), key=lambda x: x[1], reverse=True)]
            held_syms = [s for s in symbols if portfolio_shares[s] > 0]
            
            for sym in held_syms:
                if sym not in ranked[:3]: # Out of top 3
                    # Liquidate entirely
                    sell_val = portfolio_shares[sym] * data[sym].loc[d, 'Close']
                    fee = get_sell_fee(sell_val)
                    cash += (sell_val - fee)
                    portfolio_shares[sym] = 0
                    p_base[sym] = 0
                    
                    # Check what top 2 are unheld
                    current_holders = [s for s in symbols if portfolio_shares[s] > 0]
                    target = None
                    for ts in ranked[:2]:
                        if ts not in current_holders:
                            target = ts
                            break
                    if target is not None:
                        # Reinvest the specific proceeds
                        proceeds = sell_val - fee
                        if proceeds > 0:
                            buy_fee = get_buy_fee(proceeds)
                            actual_buy = proceeds - buy_fee
                            cash -= proceeds  # The proceeds were added to cash, now subtract to buy
                            # Wait, the above logic added proceeds to pure free cash. 
                            # 'proceeds' is the exact cash we just got. We take it from 'cash' and buy target.
                            portfolio_shares[target] = actual_buy / data[target].loc[d, 'Close']
                            p_base[target] = data[target].loc[d, 'Close']
                            
        # End of day valuation
        cur_val = cash + sum(portfolio_shares[sym] * data[sym].loc[d, 'Close'] for sym in symbols)
        prev_unav = cur_val / total_units
        unit_navs.append(prev_unav)
        
    mdd, wr, vol = get_perf_metrics(unit_navs)
    abs_ret = (cur_val / total_input - 1)
    return abs_ret, mdd, wr, vol

periods = ['1Y', '2Y', '3Y', '5Y']
res_table = {'B&H':{}, 'Comp':{}}

for p in periods:
    sd = start_dates[p]
    ed = common_dates[-1]
    
    ret, mdd, wr, vol = backtest_benchmark(sd, ed)
    res_table['B&H'][p] = {'ret': ret, 'mdd': mdd, 'wr': wr, 'vol': vol}
    
    ret, mdd, wr, vol = backtest_composite(sd, ed)
    res_table['Comp'][p] = {'ret': ret, 'mdd': mdd, 'wr': wr, 'vol': vol}

print("| 策略名称 | 近1年累计收益率 | 近2年累计收益率 | 近3年累计收益率 | 近5年累计收益率 | 最大回撤(近5年) | 胜率/波动率说明(近5年) |")
print("|---|---|---|---|---|---|---|")

bh_r1 = res_table['B&H']['1Y']['ret']*100
bh_r2 = res_table['B&H']['2Y']['ret']*100
bh_r3 = res_table['B&H']['3Y']['ret']*100
bh_r5 = res_table['B&H']['5Y']['ret']*100
bh_mdd = res_table['B&H']['5Y']['mdd']*100
bh_wr = res_table['B&H']['5Y']['wr']*100
bh_vol = res_table['B&H']['5Y']['vol']*100
print(f"| 基准：五大行等权持有 | {bh_r1:.2f}% | {bh_r2:.2f}% | {bh_r3:.2f}% | {bh_r5:.2f}% | {bh_mdd:.2f}% | 年化波动率 {bh_vol:.2f}%, 日胜率 {bh_wr:.2f}% |")

cp_r1 = res_table['Comp']['1Y']['ret']*100
cp_r2 = res_table['Comp']['2Y']['ret']*100
cp_r3 = res_table['Comp']['3Y']['ret']*100
cp_r5 = res_table['Comp']['5Y']['ret']*100
cp_mdd = res_table['Comp']['5Y']['mdd']*100
cp_wr = res_table['Comp']['5Y']['wr']*100
cp_vol = res_table['Comp']['5Y']['vol']*100
print(f"| 综合增强策略 | {cp_r1:.2f}% | {cp_r2:.2f}% | {cp_r3:.2f}% | {cp_r5:.2f}% | {cp_mdd:.2f}% | 年化波动率 {cp_vol:.2f}%, 日胜率 {cp_wr:.2f}% |")
