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

def backtest_p_variants(init_cap, s_date, e_date, data, symbols, common_dates, variant='dynamic'):
    dates = [d for d in common_dates if s_date <= d <= e_date]
    if not dates: return 0.0, 0.0
    
    df_dates = pd.DataFrame({'Date': dates})
    df_dates['Q'] = df_dates['Date'].dt.to_period('Q')
    reb_quarter_ends = set(df_dates.groupby('Q')['Date'].max())
    
    cash = init_cap
    shares = {sym: 0 for sym in symbols}
    p_base = {sym: 0.0 for sym in symbols}
    
    # We need to track if we have already traded the +5% or -5% levels in 'static' mode
    # to avoid repeating the trade every day the price stays above/below the line.
    # In 'dynamic' mode, P moves, so it naturally handles this.
    # In 'static' mode, we'll use a simple 'has_traded' flag that resets when crossing P.
    traded_levels = {sym: 0 for sym in symbols} # 1 for +5% sold, -1 for -5% bought, 0 for neutral
    
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
                p_base[sym] -= div # Dividend adjustment
        
        for sym in symbols:
            lo, hi, op = data[sym].loc[d, 'Low'], data[sym].loc[d, 'High'], data[sym].loc[d, 'Open']
            step = 0.05
            ratio = 0.20
            
            # Dynamic Variant (Current)
            if variant == 'dynamic':
                if lo <= p_base[sym] * (1 - step):
                    exec_p = min(op, p_base[sym] * (1 - step))
                    target_v = ((sum(shares[s]*data[s].loc[d,'Close'] for s in symbols)+cash)/num) * ratio
                    buy_q = int(target_v // exec_p // 100) * 100
                    if buy_q > 0:
                        fee = get_buy_fee(buy_q * exec_p)
                        if cash >= (buy_q * exec_p + fee):
                            cash -= (buy_q * exec_p + fee)
                            shares[sym] += buy_q
                            p_base[sym] = exec_p
                elif hi >= p_base[sym] * (1 + step):
                    exec_p = max(op, p_base[sym] * (1 + step))
                    target_v = ((sum(shares[s]*data[s].loc[d,'Close'] for s in symbols)+cash)/num) * ratio
                    sell_q = int(target_v // exec_p // 100) * 100
                    if sell_q > 0 and shares[sym] >= sell_q:
                        val = sell_q * exec_p
                        fee = get_sell_fee(val)
                        cash += (val - fee)
                        shares[sym] -= sell_q
                        p_base[sym] = exec_p
            
            # Static Variant (Fixed P for the quarter)
            else:
                # If price returns to P_base, reset the trade flag (can trade +5% or -5% again)
                # For banking stocks, let's keep it simple: can only sell once per quarter at hi, buy once per quarter at lo.
                if lo <= p_base[sym] * (1 - step) and traded_levels[sym] != -1:
                    exec_p = min(op, p_base[sym] * (1 - step))
                    target_v = ((sum(shares[s]*data[s].loc[d,'Close'] for s in symbols)+cash)/num) * ratio
                    buy_q = int(target_v // exec_p // 100) * 100
                    if buy_q > 0:
                        fee = get_buy_fee(buy_q * exec_p)
                        if cash >= (buy_q * exec_p + fee):
                            cash -= (buy_q * exec_p + fee)
                            shares[sym] += buy_q
                            traded_levels[sym] = -1 # Flag as bought at this level
                elif hi >= p_base[sym] * (1 + step) and traded_levels[sym] != 1:
                    exec_p = max(op, p_base[sym] * (1 + step))
                    target_v = ((sum(shares[s]*data[s].loc[d,'Close'] for s in symbols)+cash)/num) * ratio
                    sell_q = int(target_v // exec_p // 100) * 100
                    if sell_q > 0 and shares[sym] >= sell_q:
                        val = sell_q * exec_p
                        fee = get_sell_fee(val)
                        cash += (val - fee)
                        shares[sym] -= sell_q
                        traded_levels[sym] = 1 # Flag as sold at this level

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
            # Rebalance Day: ALWAYS reset P_base to current close and clear flags
            for s in symbols:
                p_base[s] = data[s].loc[d, 'Close']
                traded_levels[s] = 0
                
        navs.append(cash + sum(shares[s] * data[s].loc[d, 'Close'] for s in symbols))
    return navs[-1] / init_cap - 1, get_mdd(navs)

data_6, dates_6 = load_data(symbols_6)
max_date = dates_6[-1]
horizons = ['1Y', '2Y', '3Y', '5Y']
init_cap = 30000.0

print("| 时间跨度 | 动态爬梯P(收益) | 动态爬梯P(回撤) | 季度固定P(收益) | 季度固定P(回撤) |")
print("|---|---|---|---|---|")
for h in horizons:
    sd = max_date - relativedelta(years=int(h[0]))
    r_dyn, m_dyn = backtest_p_variants(init_cap, sd, max_date, data_6, symbols_6, dates_6, 'dynamic')
    r_sta, m_sta = backtest_p_variants(init_cap, sd, max_date, data_6, symbols_6, dates_6, 'static')
    print(f"| {h} | {r_dyn*100:.2f}% | {m_dyn*100:.2f}% | {r_sta*100:.2f}% | {m_sta*100:.2f}% |")
