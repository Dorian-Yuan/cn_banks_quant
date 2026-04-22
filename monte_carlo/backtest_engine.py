from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from .data_loader import load_all_data, build_merged_data, load_bank_data


@dataclass
class BacktestConfig:
    banks: List[str]
    start_date: str
    end_date: str
    capital: float = 30000
    grid_sell_pct: float = 0.03
    grid_buy_pct: float = 0.02
    trade_ratio: float = 0.20
    min_position_pct: float = 0.50
    rebalance_period: str = 'month'
    dividend_tax: bool = True
    commission_rate: float = 0.0000854
    commission_min: float = 0.5
    exempt_five: bool = True
    stamp_duty_rate: float = 0.0005
    slippage: float = 0
    limit_check: bool = True
    enable_rebalance: bool = True
    enable_grid: bool = True
    grid_max_loops: int = 5
    dividend_reinvest: bool = False
    t_plus_1: bool = True


@dataclass
class BacktestResult:
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    final_value: float = 0.0
    total_fees: float = 0.0
    total_dividend: float = 0.0
    total_div_tax: float = 0.0
    total_trades: int = 0
    grid_buy_count: int = 0
    grid_sell_count: int = 0
    rebalance_count: int = 0
    trading_days: int = 0
    equity_curve: List[float] = field(default_factory=list)
    equity_dates: List[str] = field(default_factory=list)


def round_price(price):
    return round(price * 100) / 100


def get_closest_shares(target_amount, price):
    if price <= 0:
        return 0
    raw = target_amount / price
    low = int(raw // 100) * 100
    high = low + 100
    if low == 0:
        return high
    if abs(low * price - target_amount) <= abs(high * price - target_amount):
        return low
    return high


def get_closest_shares_from_count(target_count):
    rounded = round(target_count / 100) * 100
    if rounded < 100:
        return 0
    return int(rounded)


def calc_fees(amount, is_sell, config):
    raw_comm = amount * config.commission_rate
    if config.exempt_five:
        comm = max(config.commission_min, raw_comm)
    else:
        comm = max(5.0, raw_comm)
    stamp = amount * config.stamp_duty_rate if is_sell else 0
    return comm + stamp


_date_cache = {}


def _date_ordinal(date_str):
    if date_str in _date_cache:
        return _date_cache[date_str]
    y, m, d = int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10])
    val = y * 10000 + m * 100 + d
    _date_cache[date_str] = val
    return val


def _days_between(d1, d2):
    from datetime import date
    return (date(d2 // 10000, (d2 % 10000) // 100, d2 % 100) -
            date(d1 // 10000, (d1 % 10000) // 100, d1 % 100)).days


def consume_lots(stock, sell_shares, sell_date_ordinal, lots, dividend_tax):
    if not dividend_tax or sell_shares <= 0:
        return 0.0
    tax = 0.0
    remaining = sell_shares
    i = 0
    while i < len(lots[stock]) and remaining > 0:
        lot = lots[stock][i]
        consumed = min(remaining, lot['shares'])
        if consumed <= 0:
            i += 1
            continue
        hold_days = _days_between(lot['date_ordinal'], sell_date_ordinal)
        rate = 0.0
        if hold_days <= 30:
            rate = 0.20
        elif hold_days <= 365:
            rate = 0.10
        div_per_share = lot['div_received'] / lot['shares'] if lot['shares'] > 0 else 0
        tax += div_per_share * consumed * rate
        lot['shares'] -= consumed
        lot['div_received'] -= div_per_share * consumed
        remaining -= consumed
        if lot['shares'] <= 0:
            lots[stock].pop(i)
        else:
            i += 1
    return tax


def is_period_end(date_str, next_date_str, period):
    if not next_date_str:
        return True
    y, m = int(date_str[:4]), int(date_str[5:7])
    ny, nm = int(next_date_str[:4]), int(next_date_str[5:7])
    if period == 'month':
        return m != nm or y != ny
    elif period == 'quarter':
        q = (m - 1) // 3
        nq = (nm - 1) // 3
        return q != nq or y != ny
    elif period == 'year':
        return y != ny
    return False


def _merged_to_fast(merged, dates, banks):
    fast = {}
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]
    for s in banks:
        if s not in merged:
            continue
        sub = merged[s]
        n = len(dates)
        opens = np.empty(n)
        closes = np.empty(n)
        highs = np.empty(n)
        lows = np.empty(n)
        divs = np.empty(n)
        for i, d in enumerate(dates):
            row = sub.loc[d]
            opens[i] = float(row['Open']) if not np.isnan(row['Open']) else 0
            closes[i] = float(row['Close']) if not np.isnan(row['Close']) else 0
            highs[i] = float(row['High']) if not np.isnan(row['High']) else 0
            lows[i] = float(row['Low']) if not np.isnan(row['Low']) else 0
            divs[i] = float(row.get('DivCash', 0)) if not np.isnan(row.get('DivCash', 0)) else 0
        fast[s] = {'open': opens, 'close': closes, 'high': highs, 'low': lows, 'div': divs}
    return fast, date_strs


def run_backtest(config, preloaded_merged=None, preloaded_dates=None):
    banks = config.banks
    start = config.start_date
    end = config.end_date

    if preloaded_merged is not None and preloaded_dates is not None:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        dates = [d for d in preloaded_dates if start_ts <= d <= end_ts]
        if not dates or len(dates) < 2:
            return BacktestResult()
        merged = {}
        for s in banks:
            if s in preloaded_merged:
                merged[s] = preloaded_merged[s].loc[dates]
            else:
                return BacktestResult()
    else:
        data_dict = load_all_data(banks)
        merged, dates = build_merged_data(data_dict, start, end)
        if not dates or len(dates) < 2:
            return BacktestResult()

    fast, date_strs = _merged_to_fast(merged, dates, banks)
    n_days = len(dates)

    N = len(banks)
    capital = config.capital
    cash = capital
    portfolio = {s: 0 for s in banks}
    base_p = {s: 0.0 for s in banks}
    lots = {s: [] for s in banks}
    init_shares = {s: 0 for s in banks}

    total_fees = 0.0
    total_dividend = 0.0
    total_div_tax = 0.0
    total_trades = 0
    grid_buy_count = 0
    grid_sell_count = 0
    rebalance_count = 0

    per_stock = capital / N
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

    if init_total_cost > capital:
        capital = init_total_cost
        cash = capital

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
        total_fees += plan['fees']
        total_trades += 1
        lots[s].append({'date_ordinal': first_date_ordinal, 'shares': plan['shares'], 'div_received': 0.0})
        init_shares[s] = plan['shares']

    equity_curve = []

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
                                total_fees += fees
                                total_trades += 1
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

        if config.enable_grid:
            open_position = {s: (0 if idx == 0 else (portfolio[s] or 0)) for s in banks}
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
                        total_trades += 1
                        grid_buy_count += 1
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
                    total_div_tax += div_tax
                    total_trades += 1
                    grid_sell_count += 1

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
                    total_div_tax += div_tax
                    total_trades += 1

                for a in rebalance_buys:
                    cost = a['diff'] * a['price']
                    fees = calc_fees(cost, False, config)
                    if cash >= cost + fees:
                        portfolio[a['stock']] = (portfolio[a['stock']] or 0) + a['diff']
                        cash -= (cost + fees)
                        total_fees += fees
                        total_trades += 1
                        lots[a['stock']].append(
                            {'date_ordinal': date_ordinal, 'shares': a['diff'], 'div_received': 0.0})

                rebalance_count += 1

        day_value = cash
        for s in banks:
            day_value += (portfolio[s] or 0) * fast[s]['close'][idx]
        equity_curve.append(day_value)

    if not equity_curve:
        return BacktestResult()

    final_value = equity_curve[-1]
    total_return = (final_value / capital - 1) * 100
    trading_days = len(equity_curve)
    years = trading_days / 252
    annual_return = ((final_value / capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    peak = 0
    max_drawdown = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (v - peak) / peak if peak > 0 else 0
        if dd < max_drawdown:
            max_drawdown = dd

    daily_returns = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i - 1] > 0:
            daily_returns.append((equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1])
    if daily_returns:
        arr = np.array(daily_returns)
        avg_dr = arr.mean()
        std_dr = arr.std(ddof=1) if len(arr) > 1 else 0
        sharpe_ratio = (avg_dr / std_dr) * np.sqrt(252) if std_dr > 0 else 0
    else:
        sharpe_ratio = 0

    calmar_ratio = annual_return / abs(max_drawdown * 100) if max_drawdown < 0 and years > 0 else 0

    return BacktestResult(
        total_return=total_return,
        annual_return=annual_return,
        max_drawdown=max_drawdown * 100,
        sharpe_ratio=sharpe_ratio,
        calmar_ratio=calmar_ratio,
        final_value=final_value,
        total_fees=total_fees,
        total_dividend=total_dividend,
        total_div_tax=total_div_tax,
        total_trades=total_trades,
        grid_buy_count=grid_buy_count,
        grid_sell_count=grid_sell_count,
        rebalance_count=rebalance_count,
        trading_days=trading_days,
        equity_curve=equity_curve,
        equity_dates=date_strs,
    )


def run_buy_hold(config, preloaded_merged=None, preloaded_dates=None):
    banks = config.banks
    start = config.start_date
    end = config.end_date

    if preloaded_merged is not None and preloaded_dates is not None:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        dates = [d for d in preloaded_dates if start_ts <= d <= end_ts]
        if not dates or len(dates) < 2:
            return BacktestResult()
        merged = {}
        for s in banks:
            if s in preloaded_merged:
                merged[s] = preloaded_merged[s].loc[dates]
            else:
                return BacktestResult()
    else:
        data_dict = load_all_data(banks)
        merged, dates = build_merged_data(data_dict, start, end)
        if not dates or len(dates) < 2:
            return BacktestResult()

    fast, date_strs = _merged_to_fast(merged, dates, banks)
    n_days = len(dates)

    N = len(banks)
    capital = config.capital
    cash = capital
    portfolio = {s: 0 for s in banks}

    per_stock = capital / N
    init_total_cost = 0.0

    for s in banks:
        price = round_price(fast[s]['close'][0])
        if price <= 0:
            continue
        shares = get_closest_shares(per_stock, price)
        cost = shares * price
        fees = calc_fees(cost, False, config)
        portfolio[s] = shares
        cash -= (cost + fees)
        init_total_cost += cost + fees

    if init_total_cost > capital:
        capital = init_total_cost
    cash = capital - init_total_cost

    total_dividend = 0.0
    equity_curve = []

    for idx in range(n_days):
        for s in banks:
            div_cash = fast[s]['div'][idx]
            if div_cash > 0 and portfolio[s] > 0:
                income = portfolio[s] * div_cash
                cash += income
                total_dividend += income

        day_value = cash
        for s in banks:
            day_value += (portfolio[s] or 0) * fast[s]['close'][idx]
        equity_curve.append(day_value)

    if not equity_curve:
        return BacktestResult()

    final_value = equity_curve[-1]
    total_return = (final_value / capital - 1) * 100
    trading_days = len(equity_curve)
    years = trading_days / 252
    annual_return = ((final_value / capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    peak = 0
    max_drawdown = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (v - peak) / peak if peak > 0 else 0
        if dd < max_drawdown:
            max_drawdown = dd

    daily_returns = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i - 1] > 0:
            daily_returns.append((equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1])
    if daily_returns:
        arr = np.array(daily_returns)
        avg_dr = arr.mean()
        std_dr = arr.std(ddof=1) if len(arr) > 1 else 0
        sharpe_ratio = (avg_dr / std_dr) * np.sqrt(252) if std_dr > 0 else 0
    else:
        sharpe_ratio = 0

    calmar_ratio = annual_return / abs(max_drawdown * 100) if max_drawdown < 0 and years > 0 else 0

    return BacktestResult(
        total_return=total_return,
        annual_return=annual_return,
        max_drawdown=max_drawdown * 100,
        sharpe_ratio=sharpe_ratio,
        calmar_ratio=calmar_ratio,
        final_value=final_value,
        total_fees=0.0,
        total_dividend=total_dividend,
        total_div_tax=0.0,
        total_trades=0,
        grid_buy_count=0,
        grid_sell_count=0,
        rebalance_count=0,
        trading_days=trading_days,
        equity_curve=equity_curve,
        equity_dates=date_strs,
    )


def run_single_stock_grid(stock, config, preloaded_merged=None, preloaded_dates=None):
    single_config = BacktestConfig(
        banks=[stock],
        start_date=config.start_date,
        end_date=config.end_date,
        capital=config.capital,
        grid_sell_pct=config.grid_sell_pct,
        grid_buy_pct=config.grid_buy_pct,
        trade_ratio=config.trade_ratio,
        min_position_pct=config.min_position_pct,
        rebalance_period='none',
        dividend_tax=config.dividend_tax,
        commission_rate=config.commission_rate,
        commission_min=config.commission_min,
        exempt_five=config.exempt_five,
        stamp_duty_rate=config.stamp_duty_rate,
        slippage=config.slippage,
        limit_check=config.limit_check,
        enable_rebalance=False,
        enable_grid=True,
        grid_max_loops=config.grid_max_loops,
        dividend_reinvest=config.dividend_reinvest,
    )
    return run_backtest(single_config, preloaded_merged=preloaded_merged, preloaded_dates=preloaded_dates)
