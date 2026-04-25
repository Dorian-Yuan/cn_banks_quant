import sys
import os
import time
import numpy as np
import pandas as pd
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from monte_carlo.backtest_engine import (
    BacktestConfig, round_price, get_closest_shares,
    get_closest_shares_from_count, calc_fees, _date_ordinal, _days_between,
    consume_lots, is_period_end, _merged_to_fast
)
from monte_carlo.data_loader import load_all_data, build_merged_data
from monte_carlo.config import FIVE_BANK_CODES, SIX_BANK_CODES, PSBC_START

GRID_SELL_PCT = 0.012
GRID_BUY_PCT = 0.005
TRADE_RATIO = 0.30
MIN_POSITION_PCT = 0.50
COMMISSION_RATE = 0.0000854
COMMISSION_MIN = 0.5
EXEMPT_FIVE = True
STAMP_DUTY_RATE = 0.0005
GRID_MAX_LOOPS = 5
CAPITAL = 30000
DIVIDEND_TAX = True
DIVIDEND_REINVEST = False
SLIPPAGE = 0.0
LIMIT_CHECK = True
T_PLUS_1 = True

PSBC_ORDINAL = _date_ordinal(PSBC_START)

HOLD_PERIODS = {
    '1个月': 21,
    '2个月': 42,
    '3个月': 63,
    '6个月': 126,
    '9个月': 189,
    '12个月': 252,
    '18个月': 378,
    '24个月': 504,
}

N_SAMPLES = 2500
CONVERGENCE_BATCHES = [500, 1000, 1500, 2000, 2500]
RANDOM_SEED = 42


def run_backtest_with_random_entry(banks_init, all_banks_data, fast, date_strs, n_days,
                                   entry_prices, hold_days, psbc_ordinal=PSBC_ORDINAL):
    N = len(banks_init)
    capital_used = CAPITAL
    cash = capital_used
    portfolio = {s: 0 for s in banks_init}
    base_p = {s: 0.0 for s in banks_init}
    lots = {s: [] for s in banks_init}
    init_shares = {s: 0 for s in banks_init}

    per_stock = capital_used / N
    init_plan = []
    init_total_cost = 0.0

    for s in banks_init:
        price = entry_prices.get(s, 0)
        if price <= 0:
            init_plan.append({'stock': s, 'shares': 0, 'price': 0, 'cost': 0, 'fees': 0})
            continue
        shares = get_closest_shares(per_stock, price)
        cost = shares * price
        fees = calc_fees(cost, False, BacktestConfig(
            banks=banks_init, start_date='2020-01-01', end_date='2020-12-31',
            commission_rate=COMMISSION_RATE, commission_min=COMMISSION_MIN,
            exempt_five=EXEMPT_FIVE, stamp_duty_rate=STAMP_DUTY_RATE,
        ))
        init_plan.append({'stock': s, 'shares': shares, 'price': price, 'cost': cost, 'fees': fees})
        init_total_cost += cost + fees

    if init_total_cost > capital_used:
        capital_used = init_total_cost
        cash = capital_used

    first_date_ordinal = _date_ordinal(date_strs[0])
    for plan in init_plan:
        s = plan['stock']
        if plan['shares'] <= 0:
            portfolio[s] = 0
            base_p[s] = round_price(entry_prices.get(s, fast[s]['close'][0]))
            continue
        portfolio[s] = plan['shares']
        cash -= (plan['cost'] + plan['fees'])
        base_p[s] = plan['price']
        lots[s].append({'date_ordinal': first_date_ordinal, 'shares': plan['shares'], 'div_received': 0.0})
        init_shares[s] = plan['shares']

    config = BacktestConfig(
        banks=banks_init, start_date='2020-01-01', end_date='2020-12-31',
        capital=CAPITAL, grid_sell_pct=GRID_SELL_PCT, grid_buy_pct=GRID_BUY_PCT,
        trade_ratio=TRADE_RATIO, min_position_pct=MIN_POSITION_PCT,
        dividend_tax=DIVIDEND_TAX, commission_rate=COMMISSION_RATE,
        commission_min=COMMISSION_MIN, exempt_five=EXEMPT_FIVE,
        stamp_duty_rate=STAMP_DUTY_RATE, slippage=SLIPPAGE,
        limit_check=LIMIT_CHECK, enable_rebalance=True, enable_grid=True,
        grid_max_loops=GRID_MAX_LOOPS, dividend_reinvest=DIVIDEND_REINVEST,
        t_plus_1=T_PLUS_1,
    )

    current_banks = list(banks_init)
    psbc_added = '601658' in current_banks
    equity_curve = []
    peak = capital_used
    max_drawdown = 0.0

    target_end_idx = min(hold_days, n_days)

    for idx in range(target_end_idx):
        date_str = date_strs[idx]
        next_date_str = date_strs[idx + 1] if idx + 1 < n_days else None
        current_ordinal = _date_ordinal(date_str)

        if not psbc_added and current_ordinal >= psbc_ordinal:
            if '601658' in fast and fast['601658']['close'][idx] > 0:
                current_banks.append('601658')
                portfolio['601658'] = 0
                base_p['601658'] = round_price(fast['601658']['close'][idx])
                lots['601658'] = []
                init_shares['601658'] = 0
                N = len(current_banks)
                psbc_added = True

        for s in current_banks:
            if s not in fast:
                continue
            div_cash = fast[s]['div'][idx]
            if div_cash > 0 and portfolio.get(s, 0) > 0:
                income = portfolio[s] * div_cash
                cash += income
                for lot in lots.get(s, []):
                    lot['div_received'] += div_cash * lot['shares']

        open_position = {s: (0 if idx == 0 else (portfolio.get(s, 0) or 0)) for s in current_banks}

        if idx > 0:
            for s in current_banks:
                if s not in fast:
                    continue
                open_p = fast[s]['open'][idx]
                high = fast[s]['high'][idx]
                low = fast[s]['low'][idx]

                if base_p.get(s, 0) <= 0 or open_p <= 0:
                    continue

                prev_close = fast[s]['close'][idx - 1] if idx > 0 else fast[s]['close'][idx]
                limit_up = round_price(prev_close * 1.10) if LIMIT_CHECK else 999999
                limit_down = round_price(prev_close * 0.90) if LIMIT_CHECK else 0

                loop_count = 0
                while low <= base_p[s] * (1 - GRID_BUY_PCT) and loop_count < GRID_MAX_LOOPS:
                    loop_count += 1
                    trigger_price = round_price(base_p[s] * (1 - GRID_BUY_PCT))
                    exec_price = round_price(min(open_p, trigger_price))
                    if exec_price > limit_up or exec_price < limit_down:
                        break
                    buy_shares = get_closest_shares_from_count(portfolio[s] * TRADE_RATIO)
                    if buy_shares <= 0:
                        break
                    cost = buy_shares * exec_price
                    fees = calc_fees(cost, False, config)
                    if cash >= cost + fees:
                        portfolio[s] += buy_shares
                        cash -= (cost + fees)
                        base_p[s] = exec_price
                        lots[s].append({'date_ordinal': current_ordinal, 'shares': buy_shares, 'div_received': 0.0})
                    else:
                        break

                loop_count = 0
                while high >= base_p[s] * (1 + GRID_SELL_PCT) and loop_count < GRID_MAX_LOOPS:
                    loop_count += 1
                    trigger_price = round_price(base_p[s] * (1 + GRID_SELL_PCT))
                    exec_price = round_price(max(open_p, trigger_price))
                    if exec_price > limit_up or exec_price < limit_down:
                        break
                    sell_shares = get_closest_shares_from_count(portfolio[s] * TRADE_RATIO)
                    if sell_shares <= 0:
                        break
                    if portfolio[s] < sell_shares:
                        break
                    if T_PLUS_1:
                        max_sellable = (open_position.get(s, 0) // 100) * 100
                        if sell_shares > max_sellable:
                            sell_shares = max_sellable
                        if sell_shares <= 0:
                            break
                    if MIN_POSITION_PCT > 0 and init_shares.get(s, 0) > 0:
                        min_shares = get_closest_shares_from_count(init_shares[s] * MIN_POSITION_PCT)
                        if portfolio[s] - sell_shares < min_shares:
                            sell_shares = max(0, portfolio[s] - min_shares)
                            if sell_shares <= 0:
                                break
                    proceeds = sell_shares * exec_price
                    fees = calc_fees(proceeds, True, config)
                    div_tax = consume_lots(s, sell_shares, current_ordinal, lots, DIVIDEND_TAX)
                    portfolio[s] -= sell_shares
                    cash += (proceeds - fees - div_tax)
                    base_p[s] = exec_price

        if is_period_end(date_str, next_date_str, 'month'):
            total_value = cash
            stock_prices = {}
            for s in current_banks:
                if s not in fast:
                    continue
                p = round_price(fast[s]['close'][idx])
                stock_prices[s] = p
                total_value += (portfolio.get(s, 0) or 0) * p

            cur_N = len(current_banks)
            target_per_stock = total_value / cur_N
            rebalance_sells = []
            rebalance_buys = []

            for s in current_banks:
                if s not in stock_prices or stock_prices[s] <= 0:
                    continue
                price = stock_prices[s]
                target_shares = get_closest_shares(target_per_stock, price)
                diff = target_shares - (portfolio.get(s, 0) or 0)
                if diff < 0:
                    rebalance_sells.append({'stock': s, 'diff': abs(diff), 'price': price})
                elif diff > 0:
                    rebalance_buys.append({'stock': s, 'diff': diff, 'price': price})
                base_p[s] = round_price(price)

            for a in rebalance_sells:
                sell_diff = a['diff']
                if MIN_POSITION_PCT > 0 and init_shares.get(a['stock'], 0) > 0:
                    min_shares = get_closest_shares_from_count(
                        init_shares[a['stock']] * MIN_POSITION_PCT)
                    if (portfolio.get(a['stock'], 0) or 0) - sell_diff < min_shares:
                        sell_diff = max(0, (portfolio.get(a['stock'], 0) or 0) - min_shares)
                if T_PLUS_1:
                    max_sellable = (open_position.get(a['stock'], 0) // 100) * 100
                    if sell_diff > max_sellable:
                        sell_diff = max_sellable
                if sell_diff <= 0:
                    continue
                proceeds = sell_diff * a['price']
                fees = calc_fees(proceeds, True, config)
                div_tax = consume_lots(a['stock'], sell_diff, current_ordinal, lots, DIVIDEND_TAX)
                portfolio[a['stock']] = (portfolio.get(a['stock'], 0) or 0) - sell_diff
                cash += (proceeds - fees - div_tax)

            for a in rebalance_buys:
                cost = a['diff'] * a['price']
                fees = calc_fees(cost, False, config)
                if cash >= cost + fees:
                    portfolio[a['stock']] = (portfolio.get(a['stock'], 0) or 0) + a['diff']
                    cash -= (cost + fees)
                    lots[a['stock']].append(
                        {'date_ordinal': current_ordinal, 'shares': a['diff'], 'div_received': 0.0})

        day_value = cash
        for s in current_banks:
            if s in fast:
                day_value += (portfolio.get(s, 0) or 0) * fast[s]['close'][idx]
        equity_curve.append(day_value)

        if day_value > peak:
            peak = day_value
        dd = (day_value - peak) / peak if peak > 0 else 0
        if dd < max_drawdown:
            max_drawdown = dd

    if not equity_curve:
        return None, None, None

    final_value = equity_curve[-1]
    total_return = (final_value / capital_used - 1) * 100

    return total_return, max_drawdown * 100, capital_used


def main():
    print("=" * 90)
    print("  蒙特卡洛网格策略盈利概率分析")
    print("  策略: 卖1.2%买0.5% + 单次30% + 月度再平准 + 3万本金")
    print("=" * 90)
    print(f"\n  参数:")
    print(f"    网格: 卖{GRID_SELL_PCT*100}% / 买{GRID_BUY_PCT*100}%")
    print(f"    单次调仓: {TRADE_RATIO*100:.0f}%")
    print(f"    月度再平准: 是")
    print(f"    底仓保护: {MIN_POSITION_PCT*100:.0f}%")
    print(f"    佣金: 万{COMMISSION_RATE*10000}, 最低{COMMISSION_MIN}元, 免5")
    print(f"    印花税: 卖出万{STAMP_DUTY_RATE*10000}")
    print(f"    初始资金: {CAPITAL}元")
    print(f"    2019-12-10前: 五大行, 之后: 六大行")
    print(f"    入场价: 当天最高~最低之间均匀随机")
    print(f"    样本数: {N_SAMPLES}")

    all_bank_codes = list(set(FIVE_BANK_CODES + SIX_BANK_CODES))
    data_dict = load_all_data(all_bank_codes)
    full_merged, full_dates = build_merged_data(data_dict, '2015-01-01', '2099-12-31')

    print(f"\n  数据范围: {full_dates[0].strftime('%Y-%m-%d')} ~ {full_dates[-1].strftime('%Y-%m-%d')}")
    print(f"  交易日数: {len(full_dates)}")

    n_total_dates = len(full_dates)
    max_hold = max(HOLD_PERIODS.values())

    fast_all, date_strs_all = _merged_to_fast(full_merged, full_dates, all_bank_codes)

    rng = np.random.RandomState(RANDOM_SEED)

    hold_labels = list(HOLD_PERIODS.keys())
    hold_days_list = list(HOLD_PERIODS.values())

    sample_indices = {}
    for label, h_days in HOLD_PERIODS.items():
        max_start = n_total_dates - h_days - 1
        if max_start <= 0:
            continue
        indices = rng.randint(0, max_start + 1, size=N_SAMPLES)
        sample_indices[label] = indices

    convergence_results = {batch: {label: [] for label in hold_labels} for batch in CONVERGENCE_BATCHES}

    print(f"\n  开始蒙特卡洛模拟 ({N_SAMPLES}个样本 x {len(hold_labels)}个持有期)...")
    t0 = time.time()

    all_sample_results = {label: [] for label in hold_labels}

    for i in range(N_SAMPLES):
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"    进度: {i + 1}/{N_SAMPLES} ({elapsed:.1f}秒)")

        for label, h_days in HOLD_PERIODS.items():
            if label not in sample_indices:
                continue
            start_idx = sample_indices[label][i]

            start_date_str = date_strs_all[start_idx]
            end_idx = min(start_idx + h_days + 1, n_total_dates - 1)

            sub_dates = full_dates[start_idx:end_idx + 1]
            if len(sub_dates) < h_days:
                all_sample_results[label].append(None)
                continue

            sub_merged = {}
            skip = False
            for s in all_bank_codes:
                if s in full_merged:
                    try:
                        sub_merged[s] = full_merged[s].loc[sub_dates]
                    except Exception:
                        skip = True
                        break
                else:
                    skip = True
                    break
            if skip:
                all_sample_results[label].append(None)
                continue

            fast, date_strs = _merged_to_fast(sub_merged, sub_dates, all_bank_codes)
            n_days = len(sub_dates)

            if start_date_str < PSBC_START:
                banks_init = list(FIVE_BANK_CODES)
            else:
                banks_init = list(SIX_BANK_CODES)

            entry_prices = {}
            for s in banks_init:
                if s in fast and fast[s]['close'][0] > 0:
                    low = fast[s]['low'][0]
                    high = fast[s]['high'][0]
                    if low <= 0 or high <= 0 or low >= high:
                        entry_prices[s] = round_price(fast[s]['close'][0])
                    else:
                        entry_prices[s] = round_price(rng.uniform(low, high))
                else:
                    entry_prices[s] = 0

            ret, max_dd, cap_used = run_backtest_with_random_entry(
                banks_init, sub_merged, fast, date_strs, n_days,
                entry_prices, h_days, PSBC_ORDINAL
            )

            if ret is not None:
                all_sample_results[label].append({
                    'return': ret,
                    'max_drawdown': max_dd,
                    'capital': cap_used,
                })
            else:
                all_sample_results[label].append(None)

        for batch_size in CONVERGENCE_BATCHES:
            if i + 1 == batch_size:
                for label in hold_labels:
                    results_so_far = [r for r in all_sample_results[label][:batch_size] if r is not None]
                    convergence_results[batch_size][label] = results_so_far

    elapsed = time.time() - t0
    print(f"\n  模拟完成! 总耗时: {elapsed:.1f}秒")

    print(f"\n{'=' * 90}")
    print(f"  盈利概率分析结果 (样本数={N_SAMPLES})")
    print(f"{'=' * 90}")
    print(f"{'持有期':>10} {'盈利概率':>10} {'中位收益率':>12} {'中位回撤':>10} {'平均收益':>10} {'样本数':>8}")
    print("-" * 90)

    for label in hold_labels:
        results = [r for r in all_sample_results[label] if r is not None]
        if not results:
            continue
        n = len(results)
        profitable = sum(1 for r in results if r['return'] > 0)
        prob = profitable / n * 100
        returns = [r['return'] for r in results]
        drawdowns = [r['max_drawdown'] for r in results]
        med_ret = np.median(returns)
        med_dd = np.median(drawdowns)
        avg_ret = np.mean(returns)
        print(f"{label:>10} {prob:>9.1f}% {med_ret:>+11.2f}% {med_dd:>9.2f}% {avg_ret:>+9.2f}% {n:>8}")

    print(f"\n{'=' * 90}")
    print(f"  收敛性验证")
    print(f"{'=' * 90}")

    header = f"{'样本数':>8}"
    for label in hold_labels:
        header += f" {label + '盈利概率':>14}"
    print(header)
    print("-" * (8 + 15 * len(hold_labels)))

    converged = True
    for batch_size in CONVERGENCE_BATCHES:
        line = f"{batch_size:>8}"
        for label in hold_labels:
            results = convergence_results[batch_size].get(label, [])
            if results:
                prob = sum(1 for r in results if r['return'] > 0) / len(results) * 100
                line += f" {prob:>13.1f}%"
            else:
                line += f" {'N/A':>14}"
        print(line)

    if len(CONVERGENCE_BATCHES) >= 2:
        print(f"\n  收敛判断 (2500 vs 2000 差异):")
        for label in hold_labels:
            r2500 = convergence_results[2500].get(label, [])
            r2000 = convergence_results[2000].get(label, [])
            if r2500 and r2000:
                p2500 = sum(1 for r in r2500 if r['return'] > 0) / len(r2500) * 100
                p2000 = sum(1 for r in r2000 if r['return'] > 0) / len(r2000) * 100
                diff = abs(p2500 - p2000)
                status = "[OK] 收敛" if diff < 2 else "[!!] 未收敛"
                print(f"    {label}: 差异={diff:.1f}% {status}")
                if diff >= 2:
                    converged = False

    if not converged:
        print(f"\n  [!] 部分持有期未收敛，建议增加样本数")
    else:
        print(f"\n  [OK] 所有持有期均已收敛")

    print(f"\n{'=' * 90}")
    print(f"  详细分布")
    print(f"{'=' * 90}")
    for label in hold_labels:
        results = [r for r in all_sample_results[label] if r is not None]
        if not results:
            continue
        returns = sorted([r['return'] for r in results])
        drawdowns = sorted([r['max_drawdown'] for r in results])
        n = len(returns)
        print(f"\n  {label} (n={n}):")
        print(f"    收益率: P5={returns[int(n*0.05)]:+.2f}%  P25={returns[int(n*0.25)]:+.2f}%  "
              f"P50={returns[int(n*0.50)]:+.2f}%  P75={returns[int(n*0.75)]:+.2f}%  "
              f"P95={returns[int(n*0.95)]:+.2f}%")
        print(f"    最大回撤: P5={drawdowns[int(n*0.05)]:.2f}%  P25={drawdowns[int(n*0.25)]:.2f}%  "
              f"P50={drawdowns[int(n*0.50)]:.2f}%  P75={drawdowns[int(n*0.75)]:.2f}%  "
              f"P95={drawdowns[int(n*0.95)]:.2f}%")


if __name__ == '__main__':
    main()
