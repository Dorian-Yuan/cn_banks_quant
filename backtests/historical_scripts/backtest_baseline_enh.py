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

def rebalance_portfolio(cash, shares, date, target_weight=0.2):
    val_map = {sym: shares[sym] * data[sym].loc[date, 'Close'] for sym in symbols}
    target_amount = (cash + sum(val_map.values())) * target_weight
    
    for sym in symbols:
        if val_map[sym] > target_amount:
            excess = val_map[sym] - target_amount
            fee = get_sell_fee(excess)
            cash += (excess - fee)
            shares[sym] -= excess / data[sym].loc[date, 'Close']
            
    for sym in symbols:
        val = shares[sym] * data[sym].loc[date, 'Close']
        if val < target_amount:
            deficit = target_amount - val
            actual_buy = deficit if cash >= deficit else cash
            if actual_buy > 1:
                fee = get_buy_fee(actual_buy)
                cash -= actual_buy
                shares[sym] += (actual_buy - fee) / data[sym].loc[date, 'Close']
                
    return cash, shares

def backtest_periodic(s_date, e_date):
    dates = [d for d in common_dates if s_date <= d <= e_date]
    if not dates: return 0.0, 0.0
    
    cash = 0.0
    shares = {sym: 0.0 for sym in symbols}
    d0 = dates[0]
    
    target = 50000.0 / 5
    for sym in symbols:
        fee = get_buy_fee(target)
        shares[sym] = (target - fee) / data[sym].loc[d0, 'Close']
        
    navs = [50000.0]
    
    for d in dates[1:]:
        for sym in symbols:
            div = data[sym].loc[d, 'DivCash']
            if div > 0: cash += shares[sym] * div
                
        if d in reb_quarter_ends:
            cash, shares = rebalance_portfolio(cash, shares, d, 0.2)
            
        cur_val = cash + sum(shares[sym] * data[sym].loc[d, 'Close'] for sym in symbols)
        navs.append(cur_val)
        
    return navs[-1] / 50000 - 1, get_mdd(navs)

def backtest_dca_equal_buy(s_date, e_date):
    dates = [d for d in common_dates if s_date <= d <= e_date]
    if not dates: return 0.0, 0.0
    
    cash = 0.0
    shares = {sym: 0.0 for sym in symbols}
    d0 = dates[0]
    
    total_input = 50000.0
    unit_nav = 1.0
    total_units = 50000.0
    
    target = 50000.0 / 5
    for sym in symbols:
        fee = get_buy_fee(target)
        shares[sym] = (target - fee) / data[sym].loc[d0, 'Close']
        
    navs = [unit_nav]
    
    for d in dates[1:]:
        for sym in symbols:
            div = data[sym].loc[d, 'DivCash']
            if div > 0: cash += shares[sym] * div
                
        cur_val_before = cash + sum(shares[sym] * data[sym].loc[d, 'Close'] for sym in symbols)
        if total_units > 0:
            unit_nav = cur_val_before / total_units
            
        if d in reb_quarter_ends:
            deposit = 5000.0
            total_input += deposit
            total_units += deposit / unit_nav
            cash += deposit
            
            buy_target = deposit / len(symbols)
            for sym in symbols:
                if cash >= buy_target:
                    fee = get_buy_fee(buy_target)
                    shares[sym] += (buy_target - fee) / data[sym].loc[d, 'Close']
                    cash -= buy_target
                    
        cur_val_after = cash + sum(shares[sym] * data[sym].loc[d, 'Close'] for sym in symbols)
        navs.append(unit_nav)
        
    abs_ret = (cur_val_after / total_input) - 1
    return abs_ret, get_mdd(navs)

def backtest_dca_with_rebalance(s_date, e_date):
    dates = [d for d in common_dates if s_date <= d <= e_date]
    if not dates: return 0.0, 0.0
    
    cash = 0.0
    shares = {sym: 0.0 for sym in symbols}
    d0 = dates[0]
    
    total_input = 50000.0
    unit_nav = 1.0
    total_units = 50000.0
    
    target = 50000.0 / 5
    for sym in symbols:
        fee = get_buy_fee(target)
        shares[sym] = (target - fee) / data[sym].loc[d0, 'Close']
        
    navs = [unit_nav]
    
    for d in dates[1:]:
        for sym in symbols:
            div = data[sym].loc[d, 'DivCash']
            if div > 0: cash += shares[sym] * div
                
        cur_val_before = cash + sum(shares[sym] * data[sym].loc[d, 'Close'] for sym in symbols)
        if total_units > 0:
            unit_nav = cur_val_before / total_units
            
        if d in reb_quarter_ends:
            deposit = 5000.0
            total_input += deposit
            total_units += deposit / unit_nav
            cash += deposit
            
            cash, shares = rebalance_portfolio(cash, shares, d, 0.2)
                
        cur_val_after = cash + sum(shares[sym] * data[sym].loc[d, 'Close'] for sym in symbols)
        navs.append(unit_nav)
        
    abs_ret = (cur_val_after / total_input) - 1
    return abs_ret, get_mdd(navs)

periods = ['1Y', '2Y', '3Y', '5Y']
res_table = {'Periodic': {}, 'DCA_Equal': {}, 'DCA_Reb': {}}

for p in periods:
    sd = start_dates[p]
    ed = common_dates[-1]
    
    ret, mdd = backtest_periodic(sd, ed)
    res_table['Periodic'][p] = ret
    if p == '5Y': res_table['Periodic']['MDD'] = mdd
    
    ret, mdd = backtest_dca_equal_buy(sd, ed)
    res_table['DCA_Equal'][p] = ret
    if p == '5Y': res_table['DCA_Equal']['MDD'] = mdd
    
    ret, mdd = backtest_dca_with_rebalance(sd, ed)
    res_table['DCA_Reb'][p] = ret
    if p == '5Y': res_table['DCA_Reb']['MDD'] = mdd

print("| 策略名称 | 近1年累计收益 | 近2年累计收益 | 近3年累计收益 | 近5年累计收益 | 最大回撤(近5年) |")
print("|---|---|---|---|---|---|")
pb = res_table['Periodic']
print(f"| (无定投) 纯季度再平准 | {pb['1Y']*100:.2f}% | {pb['2Y']*100:.2f}% | {pb['3Y']*100:.2f}% | {pb['5Y']*100:.2f}% | {pb['MDD']*100:.2f}% |")
de = res_table['DCA_Equal']
print(f"| 持续投入：每季等比买入不定盘 | {de['1Y']*100:.2f}% | {de['2Y']*100:.2f}% | {de['3Y']*100:.2f}% | {de['5Y']*100:.2f}% | {de['MDD']*100:.2f}% |")
dr = res_table['DCA_Reb']
print(f"| 持续投入：新资金注资即强制平准 | {dr['1Y']*100:.2f}% | {dr['2Y']*100:.2f}% | {dr['3Y']*100:.2f}% | {dr['5Y']*100:.2f}% | {dr['MDD']*100:.2f}% |")
