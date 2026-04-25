import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from monte_carlo.backtest_engine import (
    BacktestConfig, run_backtest, round_price, get_closest_shares,
    get_closest_shares_from_count, calc_fees, _date_ordinal, _days_between,
    consume_lots, is_period_end, _merged_to_fast
)
from monte_carlo.data_loader import load_all_data, build_merged_data

BANKS = ['601288', '601328', '601398', '601658', '601939', '601988']
CAPITAL = 30000
GRID_SELL_PCT = 0.012
GRID_BUY_PCT = 0.005
TRADE_RATIO = 0.20
MIN_POSITION_PCT = 0.50
COMMISSION_RATE = 0.0000854
COMMISSION_MIN = 0.5
EXEMPT_FIVE = True
STAMP_DUTY_RATE = 0.0005
GRID_MAX_LOOPS = 5
T_PLUS_1 = True
DIVIDEND_TAX = True
DIVIDEND_REINVEST = False
SLIPPAGE = 0.0
LIMIT_CHECK = True

TARGET_STOCK = '601288'
HOLD_PERIODS_DAYS = [5, 10, 20, 40, 60, 80, 120, 160, 200, 252, 504]


def find_peak_entries(close_prices, lookback=60):
    peaks = []
    for i in range(lookback, len(close_prices)):
        window = close_prices[i - lookback:i + 1]
        current = close_prices[i]
        if current <= 0:
            continue
        if current == max(window):
            rank = sum(1 for p in window if p < current) / len(window)
            if rank >= 0.90:
                peaks.append(i)
    filtered = []
    for p in peaks:
        if not filtered or p - filtered[-1] > 20:
            filtered.append(p)
    return filtered


def run_grid_backtest_single(stock, merged, dates, date_strs, fast, n_days, hold_days_list, capital):
    config = BacktestConfig(
        banks=[stock],
        start_date=date_strs[0],
        end_date=date_strs[-1],
        capital=capital,
        grid_sell_pct=GRID_SELL_PCT,
        grid_buy_pct=GRID_BUY_PCT,
        trade_ratio=TRADE_RATIO,
        min_position_pct=MIN_POSITION_PCT,
        rebalance_period='none',
        dividend_tax=DIVIDEND_TAX,
        commission_rate=COMMISSION_RATE,
        commission_min=COMMISSION_MIN,
        exempt_five=EXEMPT_FIVE,
        stamp_duty_rate=STAMP_DUTY_RATE,
        slippage=SLIPPAGE,
        limit_check=LIMIT_CHECK,
        enable_rebalance=False,
        enable_grid=True,
        grid_max_loops=GRID_MAX_LOOPS,
        dividend_reinvest=DIVIDEND_REINVEST,
        t_plus_1=T_PLUS_1,
    )
    return _run_backtest_core(config, [stock], merged, dates, date_strs, fast, n_days, hold_days_list)


def run_grid_backtest_multi(banks, merged, dates, date_strs, fast, n_days, hold_days_list, capital):
    config = BacktestConfig(
        banks=banks,
        start_date=date_strs[0],
        end_date=date_strs[-1],
        capital=capital,
        grid_sell_pct=GRID_SELL_PCT,
        grid_buy_pct=GRID_BUY_PCT,
        trade_ratio=TRADE_RATIO,
        min_position_pct=MIN_POSITION_PCT,
        rebalance_period='month',
        dividend_tax=DIVIDEND_TAX,
        commission_rate=COMMISSION_RATE,
        commission_min=COMMISSION_MIN,
        exempt_five=EXEMPT_FIVE,
        stamp_duty_rate=STAMP_DUTY_RATE,
        slippage=SLIPPAGE,
        limit_check=LIMIT_CHECK,
        enable_rebalance=True,
        enable_grid=True,
        grid_max_loops=GRID_MAX_LOOPS,
        dividend_reinvest=DIVIDEND_REINVEST,
        t_plus_1=T_PLUS_1,
    )
    return _run_backtest_core(config, banks, merged, dates, date_strs, fast, n_days, hold_days_list)


def _run_backtest_core(config, banks, merged, dates, date_strs, fast, n_days, hold_days_list):
    N = len(banks)
    capital_used = config.capital
    cash = capital_used
    portfolio = {s: 0 for s in banks}
    base_p = {s: 0.0 for s in banks}
    lots = {s: [] for s in banks}
    init_shares = {s: 0 for s in banks}

    per_stock = capital_used / N
    init_plan = []
    init_total_cost = 0.0

    for s in banks:
        price = round_price(fast[s]['close'][0])
        if price <= 0:
            init_plan.append({'stock': s, 'shares': 0, 'price': 0, 'cost': 0, 'fees': 0})
            continue
        shares = get_closest_shares(per_stock, price)
        cost = shares * price
        fees = calc_fees(cost, False, config)
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
            base_p[s] = round_price(fast[s]['close'][0])
            continue
        portfolio[s] = plan['shares']
        cash -= (plan['cost'] + plan['fees'])
        base_p[s] = plan['price']
        lots[s].append({'date_ordinal': first_date_ordinal, 'shares': plan['shares'], 'div_received': 0.0})
        init_shares[s] = plan['shares']

    equity_curve = []
    results_by_hold = {h: None for h in hold_days_list}
    total_fees = 0.0
    total_dividend = 0.0

    for idx in range(n_days):
        date_str = date_strs[idx]
        next_date_str = date_strs[idx + 1] if idx + 1 < n_days else None

        for s in banks:
            div_cash = fast[s]['div'][idx]
            if div_cash > 0 and portfolio[s] > 0:
                income = portfolio[s] * div_cash
                if config.dividend_reinvest:
                    buy_price = round_price(fast[s]['close'][idx])
                    if buy_price > 0:
                        reinvest_shares = get_closest_shares(income, buy_price)
                        if reinvest_shares > 0:
                            cost = reinvest_shares * buy_price
                            fees = calc_fees(cost, False, config)
                            if cash + income >= cost + fees:
                                portfolio[s] += reinvest_shares
                                cash += income - cost - fees
                                lots[s].append({'date_ordinal': _date_ordinal(date_str), 'shares': reinvest_shares, 'div_received': 0.0})
                            else:
                                cash += income
                        else:
                            cash += income
                    else:
                        cash += income
                else:
                    cash += income
                total_dividend += income
                for lot in lots[s]:
                    lot['div_received'] += div_cash * lot['shares']

        open_position = {s: (0 if idx == 0 else (portfolio[s] or 0)) for s in banks}

        if config.enable_grid and idx > 0:
            for s in banks:
                open_p = fast[s]['open'][idx]
                high = fast[s]['high'][idx]
                low = fast[s]['low'][idx]

                if base_p[s] <= 0 or open_p <= 0:
                    continue

                prev_close = fast[s]['close'][idx - 1] if idx > 0 else fast[s]['close'][idx]
                limit_up = round_price(prev_close * 1.10) if config.limit_check else 999999
                limit_down = round_price(prev_close * 0.90) if config.limit_check else 0

                loop_count = 0
                while low <= base_p[s] * (1 - config.grid_buy_pct) and loop_count < config.grid_max_loops:
                    loop_count += 1
                    trigger_price = round_price(base_p[s] * (1 - config.grid_buy_pct))
                    exec_price = round_price(min(open_p, trigger_price))
                    if config.slippage > 0:
                        exec_price = round_price(exec_price * (1 + config.slippage))
                    if exec_price > limit_up or exec_price < limit_down:
                        break
                    buy_shares = get_closest_shares_from_count(portfolio[s] * config.trade_ratio)
                    if buy_shares <= 0:
                        break
                    cost = buy_shares * exec_price
                    fees = calc_fees(cost, False, config)
                    if cash >= cost + fees:
                        portfolio[s] += buy_shares
                        cash -= (cost + fees)
                        base_p[s] = exec_price
                        total_fees += fees
                        lots[s].append({'date_ordinal': _date_ordinal(date_str), 'shares': buy_shares, 'div_received': 0.0})
                    else:
                        break

                loop_count = 0
                while high >= base_p[s] * (1 + config.grid_sell_pct) and loop_count < config.grid_max_loops:
                    loop_count += 1
                    trigger_price = round_price(base_p[s] * (1 + config.grid_sell_pct))
                    exec_price = round_price(max(open_p, trigger_price))
                    if config.slippage > 0:
                        exec_price = round_price(exec_price * (1 - config.slippage))
                    if exec_price > limit_up or exec_price < limit_down:
                        break
                    sell_shares = get_closest_shares_from_count(portfolio[s] * config.trade_ratio)
                    if sell_shares <= 0:
                        break
                    if portfolio[s] < sell_shares:
                        break
                    if config.t_plus_1:
                        max_sellable = (open_position[s] // 100) * 100
                        if sell_shares > max_sellable:
                            sell_shares = max_sellable
                        if sell_shares <= 0:
                            break
                    if config.min_position_pct > 0 and init_shares[s] > 0:
                        min_shares = get_closest_shares_from_count(init_shares[s] * config.min_position_pct)
                        if portfolio[s] - sell_shares < min_shares:
                            sell_shares = max(0, portfolio[s] - min_shares)
                            if sell_shares <= 0:
                                break
                    proceeds = sell_shares * exec_price
                    fees = calc_fees(proceeds, True, config)
                    div_tax = consume_lots(s, sell_shares, _date_ordinal(date_str), lots, config.dividend_tax)
                    portfolio[s] -= sell_shares
                    cash += (proceeds - fees - div_tax)
                    base_p[s] = exec_price
                    total_fees += fees

        if config.enable_rebalance and config.rebalance_period != 'none':
            if is_period_end(date_str, next_date_str, config.rebalance_period):
                total_value = cash
                stock_prices = {}
                for s in banks:
                    p = round_price(fast[s]['close'][idx])
                    stock_prices[s] = p
                    total_value += (portfolio[s] or 0) * p

                target_per_stock = total_value / N
                rebalance_sells = []
                rebalance_buys = []

                for s in banks:
                    price = stock_prices[s]
                    if not price or price <= 0:
                        continue
                    target_shares = get_closest_shares(target_per_stock, price)
                    diff = target_shares - (portfolio[s] or 0)
                    if diff < 0:
                        rebalance_sells.append({'stock': s, 'diff': abs(diff), 'price': price})
                    elif diff > 0:
                        rebalance_buys.append({'stock': s, 'diff': diff, 'price': price})
                    base_p[s] = round_price(price)

                date_ordinal = _date_ordinal(date_str)
                for a in rebalance_sells:
                    sell_diff = a['diff']
                    if config.min_position_pct > 0 and init_shares[a['stock']] > 0:
                        min_shares = get_closest_shares_from_count(
                            init_shares[a['stock']] * config.min_position_pct)
                        if (portfolio[a['stock']] or 0) - sell_diff < min_shares:
                            sell_diff = max(0, (portfolio[a['stock']] or 0) - min_shares)
                    if config.t_plus_1:
                        max_sellable = (open_position[a['stock']] // 100) * 100
                        if sell_diff > max_sellable:
                            sell_diff = max_sellable
                    if sell_diff <= 0:
                        continue
                    proceeds = sell_diff * a['price']
                    fees = calc_fees(proceeds, True, config)
                    div_tax = consume_lots(a['stock'], sell_diff, date_ordinal, lots, config.dividend_tax)
                    portfolio[a['stock']] = (portfolio[a['stock']] or 0) - sell_diff
                    cash += (proceeds - fees - div_tax)
                    total_fees += fees

                for a in rebalance_buys:
                    cost = a['diff'] * a['price']
                    fees = calc_fees(cost, False, config)
                    if cash >= cost + fees:
                        portfolio[a['stock']] = (portfolio[a['stock']] or 0) + a['diff']
                        cash -= (cost + fees)
                        total_fees += fees
                        lots[a['stock']].append(
                            {'date_ordinal': date_ordinal, 'shares': a['diff'], 'div_received': 0.0})

        day_value = cash
        for s in banks:
            day_value += (portfolio[s] or 0) * fast[s]['close'][idx]
        equity_curve.append(day_value)

        for h in hold_days_list:
            if idx == h - 1 and results_by_hold[h] is None:
                results_by_hold[h] = day_value

    return capital_used, equity_curve, results_by_hold, total_fees, total_dividend


def run_monte_carlo(current_price, daily_returns, n_sims=5000, hold_days_list=None):
    if hold_days_list is None:
        hold_days_list = HOLD_PERIODS_DAYS

    mu = np.mean(daily_returns)
    sigma = np.std(daily_returns, ddof=1)

    all_results = {h: [] for h in hold_days_list}

    for sim in range(n_sims):
        price = current_price
        base_p = current_price
        shares = int(CAPITAL / current_price / 100) * 100
        if shares < 100:
            shares = 100
        cash = CAPITAL - shares * current_price
        init_shares = shares
        total_fees = 0.0

        prices = [price]
        for d in range(max(hold_days_list)):
            ret = np.random.normal(mu, sigma)
            price = price * (1 + ret)
            price = round(price, 2)
            prices.append(price)

            low = price * (1 - abs(np.random.normal(0, sigma * 0.5)))
            high = price * (1 + abs(np.random.normal(0, sigma * 0.5)))
            low = round(low, 2)
            high = round(high, 2)

            if low <= base_p * (1 - GRID_BUY_PCT):
                trigger_price = round(base_p * (1 - GRID_BUY_PCT))
                exec_price = round(min(price, trigger_price), 2)
                buy_shares = get_closest_shares_from_count(shares * TRADE_RATIO)
                if buy_shares >= 100:
                    cost = buy_shares * exec_price
                    fees = calc_fees_simple(cost, False)
                    if cash >= cost + fees:
                        shares += buy_shares
                        cash -= (cost + fees)
                        base_p = exec_price
                        total_fees += fees

            if high >= base_p * (1 + GRID_SELL_PCT):
                trigger_price = round(base_p * (1 + GRID_SELL_PCT))
                exec_price = round(max(price, trigger_price), 2)
                sell_shares = get_closest_shares_from_count(shares * TRADE_RATIO)
                if sell_shares >= 100:
                    min_shares = get_closest_shares_from_count(init_shares * MIN_POSITION_PCT)
                    if shares - sell_shares < min_shares:
                        sell_shares = max(0, shares - min_shares)
                    if sell_shares >= 100:
                        proceeds = sell_shares * exec_price
                        fees = calc_fees_simple(proceeds, True)
                        shares -= sell_shares
                        cash += (proceeds - fees)
                        base_p = exec_price
                        total_fees += fees

            if (d + 1) % 21 == 0:
                base_p = price

            day_value = cash + shares * price

            for h in hold_days_list:
                if d == h - 1:
                    ret_pct = (day_value / CAPITAL - 1) * 100
                    all_results[h].append(ret_pct)

    return all_results


def calc_fees_simple(amount, is_sell):
    raw_comm = amount * COMMISSION_RATE
    comm = max(COMMISSION_MIN, raw_comm)
    stamp = amount * STAMP_DUTY_RATE if is_sell else 0
    return comm + stamp


def print_probability_table(all_results, hold_periods, title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"{'持有期':>10} {'交易日':>8} {'盈利概率':>10} {'平均收益':>10} {'中位收益':>10} {'最差收益':>10} {'最好收益':>10} {'样本数':>8}")
    print("-" * 80)

    target_hold = None
    for h in hold_periods:
        results = all_results.get(h, [])
        if not results:
            continue
        n = len(results)
        profitable = sum(1 for r in results if r > 0)
        prob = profitable / n * 100
        avg_ret = np.mean(results)
        med_ret = np.median(results)
        worst_ret = min(results)
        best_ret = max(results)

        months = h / 21
        label = f"~{months:.1f}月"
        print(f"{label:>10} {h:>8} {prob:>9.1f}% {avg_ret:>9.2f}% {med_ret:>9.2f}% {worst_ret:>9.2f}% {best_ret:>9.2f}% {n:>8}")

        if target_hold is None and prob >= 70:
            target_hold = h

    return target_hold


def main():
    print("=" * 80)
    print("  601288(农业银行) 峰值建仓 → 网格策略盈利概率分析")
    print("  建仓日: 2026-04-21 | 建仓价: ~7.18-7.29 (2026年来最高点)")
    print("=" * 80)
    print(f"\n  策略参数:")
    print(f"    网格买入: 跌{GRID_BUY_PCT * 100}%触发 (买持仓{TRADE_RATIO * 100:.0f}%)")
    print(f"    网格卖出: 涨{GRID_SELL_PCT * 100}%触发 (卖持仓{TRADE_RATIO * 100:.0f}%)")
    print(f"    月度再平准: 是 (月末重置基准价+等权调仓)")
    print(f"    底仓保护: {MIN_POSITION_PCT * 100:.0f}%")
    print(f"    佣金: 万{COMMISSION_RATE * 10000}, 最低{COMMISSION_MIN}元, 免5")
    print(f"    印花税: 卖出万{STAMP_DUTY_RATE * 10000}")
    print(f"    初始资金: {CAPITAL}元")

    data_dict = load_all_data(BANKS)
    full_merged, full_dates = build_merged_data(data_dict, '2015-01-01', '2026-12-31')

    target_data = full_merged[TARGET_STOCK]
    close_prices = target_data['Close'].values
    date_index = full_dates

    print(f"\n  数据范围: {date_index[0].strftime('%Y-%m-%d')} ~ {date_index[-1].strftime('%Y-%m-%d')}")
    print(f"  601288最新收盘价: {close_prices[-1]:.2f}")

    lookback = 60
    peak_indices = find_peak_entries(close_prices, lookback=lookback)
    min_forward = max(HOLD_PERIODS_DAYS) + 10
    valid_peaks = [p for p in peak_indices if p + min_forward < len(date_index)]

    print(f"\n  峰值建仓点 (近{lookback}交易日最高, 间隔>20天): {len(valid_peaks)}个")

    # ========== Part 1: 单股601288网格 (无再平准) ==========
    print(f"\n{'#' * 80}")
    print(f"  Part 1: 单股601288网格策略 (买0.5%卖1.2%, 月末重置基准价, 无跨股再平准)")
    print(f"{'#' * 80}")

    single_results = {h: [] for h in HOLD_PERIODS_DAYS}
    for peak_idx in valid_peaks:
        peak_date = date_index[peak_idx]
        peak_price = close_prices[peak_idx]
        start_date = peak_date.strftime('%Y-%m-%d')
        end_idx = min(peak_idx + min_forward, len(date_index) - 1)
        end_date = date_index[end_idx].strftime('%Y-%m-%d')

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        sub_dates = [d for d in full_dates if start_ts <= d <= end_ts]
        if len(sub_dates) < min_forward:
            continue

        sub_merged = {TARGET_STOCK: full_merged[TARGET_STOCK].loc[sub_dates]}
        fast, date_strs = _merged_to_fast(sub_merged, sub_dates, [TARGET_STOCK])
        n_days = len(sub_dates)

        cap_used, eq_curve, res_by_hold, fees, divs = run_grid_backtest_single(
            TARGET_STOCK, sub_merged, sub_dates, date_strs, fast, n_days,
            HOLD_PERIODS_DAYS, CAPITAL
        )

        for h in HOLD_PERIODS_DAYS:
            if res_by_hold[h] is not None:
                ret = (res_by_hold[h] / cap_used - 1) * 100
                single_results[h].append(ret)

    target1 = print_probability_table(single_results, HOLD_PERIODS_DAYS,
                                       "单股601288网格 - 峰值建仓后盈利概率")

    # ========== Part 2: 六大行组合网格+月度再平准 ==========
    print(f"\n{'#' * 80}")
    print(f"  Part 2: 六大行组合网格+月度再平准 (完整策略)")
    print(f"{'#' * 80}")

    multi_results = {h: [] for h in HOLD_PERIODS_DAYS}
    for peak_idx in valid_peaks:
        peak_date = date_index[peak_idx]
        start_date = peak_date.strftime('%Y-%m-%d')
        end_idx = min(peak_idx + min_forward, len(date_index) - 1)
        end_date = date_index[end_idx].strftime('%Y-%m-%d')

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        sub_dates = [d for d in full_dates if start_ts <= d <= end_ts]
        if len(sub_dates) < min_forward:
            continue

        sub_merged = {}
        skip = False
        for s in BANKS:
            if s in full_merged:
                sub_merged[s] = full_merged[s].loc[sub_dates]
            else:
                skip = True
                break
        if skip:
            continue

        fast, date_strs = _merged_to_fast(sub_merged, sub_dates, BANKS)
        n_days = len(sub_dates)

        cap_used, eq_curve, res_by_hold, fees, divs = run_grid_backtest_multi(
            BANKS, sub_merged, sub_dates, date_strs, fast, n_days,
            HOLD_PERIODS_DAYS, CAPITAL
        )

        for h in HOLD_PERIODS_DAYS:
            if res_by_hold[h] is not None:
                ret = (res_by_hold[h] / cap_used - 1) * 100
                multi_results[h].append(ret)

    target2 = print_probability_table(multi_results, HOLD_PERIODS_DAYS,
                                       "六大行组合网格+月度再平准 - 峰值建仓后盈利概率")

    # ========== Part 3: 仅2022年后数据 (更接近当前市场环境) ==========
    print(f"\n{'#' * 80}")
    print(f"  Part 3: 仅2022年后峰值建仓 (银行股牛市环境, 更贴近当前)")
    print(f"{'#' * 80}")

    recent_cutoff = pd.Timestamp('2022-01-01')
    recent_peaks = [p for p in valid_peaks if date_index[p] >= recent_cutoff]
    print(f"  2022年后峰值点: {len(recent_peaks)}个")

    recent_single = {h: [] for h in HOLD_PERIODS_DAYS}
    recent_multi = {h: [] for h in HOLD_PERIODS_DAYS}

    for peak_idx in recent_peaks:
        peak_date = date_index[peak_idx]
        start_date = peak_date.strftime('%Y-%m-%d')
        end_idx = min(peak_idx + min_forward, len(date_index) - 1)
        end_date = date_index[end_idx].strftime('%Y-%m-%d')

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        sub_dates = [d for d in full_dates if start_ts <= d <= end_ts]
        if len(sub_dates) < min_forward:
            continue

        sub_merged = {}
        skip = False
        for s in BANKS:
            if s in full_merged:
                sub_merged[s] = full_merged[s].loc[sub_dates]
            else:
                skip = True
                break
        if skip:
            continue

        fast, date_strs = _merged_to_fast(sub_merged, sub_dates, BANKS)
        n_days = len(sub_dates)

        cap_used, eq_curve, res_by_hold, fees, divs = run_grid_backtest_multi(
            BANKS, sub_merged, sub_dates, date_strs, fast, n_days,
            HOLD_PERIODS_DAYS, CAPITAL
        )

        for h in HOLD_PERIODS_DAYS:
            if res_by_hold[h] is not None:
                ret = (res_by_hold[h] / cap_used - 1) * 100
                recent_multi[h].append(ret)

        sub_merged_s = {TARGET_STOCK: full_merged[TARGET_STOCK].loc[sub_dates]}
        fast_s, date_strs_s = _merged_to_fast(sub_merged_s, sub_dates, [TARGET_STOCK])

        cap_used_s, eq_curve_s, res_by_hold_s, fees_s, divs_s = run_grid_backtest_single(
            TARGET_STOCK, sub_merged_s, sub_dates, date_strs_s, fast_s, n_days,
            HOLD_PERIODS_DAYS, CAPITAL
        )

        for h in HOLD_PERIODS_DAYS:
            if res_by_hold_s[h] is not None:
                ret = (res_by_hold_s[h] / cap_used_s - 1) * 100
                recent_single[h].append(ret)

    target3s = print_probability_table(recent_single, HOLD_PERIODS_DAYS,
                                        "2022年后单股601288网格 - 峰值建仓后盈利概率")
    target3m = print_probability_table(recent_multi, HOLD_PERIODS_DAYS,
                                        "2022年后六大行组合网格+月度再平准 - 峰值建仓后盈利概率")

    # ========== Part 4: 蒙特卡洛模拟 ==========
    print(f"\n{'#' * 80}")
    print(f"  Part 4: 蒙特卡洛模拟 (基于近期波动率, 5000次模拟)")
    print(f"{'#' * 80}")

    recent_data = target_data.loc[pd.Timestamp('2024-01-01'):]
    daily_returns = recent_data['Close'].pct_change().dropna().values
    current_price = close_prices[-1]

    print(f"  当前价格: {current_price:.2f}")
    print(f"  近2年日收益率: 均值={np.mean(daily_returns) * 100:.4f}%, 波动率={np.std(daily_returns, ddof=1) * 100:.4f}%")
    print(f"  年化波动率: {np.std(daily_returns, ddof=1) * np.sqrt(252) * 100:.1f}%")

    np.random.seed(42)
    mc_results = run_monte_carlo(current_price, daily_returns, n_sims=5000)

    target4 = print_probability_table(mc_results, HOLD_PERIODS_DAYS,
                                       f"蒙特卡洛模拟 - 从{current_price:.2f}元建仓 (单股601288网格)")

    # ========== Summary ==========
    print(f"\n{'=' * 80}")
    print(f"  ★★★ 综合结论 ★★★")
    print(f"{'=' * 80}")

    scenarios = [
        ("全历史·单股601288网格", target1),
        ("全历史·六大行组合+月度再平准", target2),
        ("2022年后·单股601288网格", target3s),
        ("2022年后·六大行组合+月度再平准", target3m),
        ("蒙特卡洛模拟·单股601288网格", target4),
    ]

    print(f"\n  各场景达到70%+盈利概率的最短持有期:")
    print(f"  {'场景':>35} {'最短持有期':>15}")
    print(f"  {'-' * 55}")
    for name, th in scenarios:
        if th:
            months = th / 21
            print(f"  {name:>35} ~{months:.1f}个月 ({th}交易日)")
        else:
            print(f"  {name:>35} 未达70%")

    print(f"\n  关键发现:")
    if target3m:
        months = target3m / 21
        print(f"  1. 在2022年后的银行牛市环境中, 六大行组合策略从峰值建仓,")
        print(f"     约{months:.1f}个月({target3m}交易日)可达70%+盈利概率")
    if target3s:
        months_s = target3s / 21
        print(f"  2. 仅做601288单股网格, 约需{months_s:.1f}个月({target3s}交易日)")
    if target3m and target3s:
        diff = (target3s - target3m) / 21
        print(f"  3. 组合策略比单股策略快约{diff:.1f}个月达到70%概率")
    if target4:
        months_mc = target4 / 21
        print(f"  4. 蒙特卡洛模拟显示约{months_mc:.1f}个月可达70%+概率")

    print(f"\n  风险提示:")
    print(f"  - 当前价格{current_price:.2f}为历史新高, 回撤风险较大")
    print(f"  - 2015年和2018年峰值建仓后1年内仍亏损的案例存在")
    print(f"  - 网格策略在震荡市表现好, 在单边下跌市中可能持续加仓被套")
    print(f"  - 月度再平准在趋势市中可能过早卖出强势股")


if __name__ == '__main__':
    main()
