import random
import time
from typing import List, Dict, Tuple
from scipy import stats
from .config import (
    CATEGORY_RANGES, CATEGORY_LABELS, RANDOM_SEED, PSBC_START,
    FIVE_BANK_CODES, SIX_BANK_CODES, BANK_INDEX_CODE, DATA_START,
)
from .config import get_banks_for_period
from .data_loader import get_all_trading_dates, load_index_data, load_bank_data, load_all_data, build_merged_data
from .backtest_engine import BacktestConfig, BacktestResult, run_backtest, run_buy_hold, run_single_stock_grid

_preloaded = {}


def preload_all_data():
    global _preloaded
    if _preloaded:
        return _preloaded
    all_banks = list(set(FIVE_BANK_CODES + SIX_BANK_CODES))
    data_dict = load_all_data(all_banks)
    merged, dates = build_merged_data(data_dict, "2015-01-01", "2099-12-31")
    _preloaded = {'merged': merged, 'dates': dates}
    return _preloaded


def _run_bt(config, preloaded=None):
    if preloaded:
        return run_backtest(config, preloaded_merged=preloaded['merged'], preloaded_dates=preloaded['dates'])
    return run_backtest(config)


def _run_bh(config, preloaded=None):
    if preloaded:
        return run_buy_hold(config, preloaded_merged=preloaded['merged'], preloaded_dates=preloaded['dates'])
    return run_buy_hold(config)


def _run_sg(stock, config, preloaded=None):
    if preloaded:
        return run_single_stock_grid(stock, config, preloaded_merged=preloaded['merged'], preloaded_dates=preloaded['dates'])
    return run_single_stock_grid(stock, config)


def generate_random_periods(n, category, trading_dates, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    min_days, max_days = CATEGORY_RANGES[category]
    n_dates = len(trading_dates)
    if n_dates < 2:
        return []

    results = []
    attempts = 0
    max_attempts = n * 10

    while len(results) < n and attempts < max_attempts:
        attempts += 1
        if max_days >= 99999:
            max_start = max(0, n_dates - min_days - 1)
            if max_start <= 0:
                continue
            start_idx = rng.randint(0, max_start)
            available_days = n_dates - start_idx - 1
            if available_days < min_days:
                continue
            days = rng.randint(min_days, available_days)
        else:
            max_start = max(0, n_dates - max_days - 1)
            if max_start <= 0:
                continue
            start_idx = rng.randint(0, max_start)
            days = rng.randint(min_days, max_days)
        end_idx = start_idx + days
        if end_idx >= n_dates:
            end_idx = n_dates - 1
        if end_idx <= start_idx:
            continue
        actual_days = end_idx - start_idx
        cat_min, cat_max = CATEGORY_RANGES[category]
        if actual_days < cat_min:
            continue
        results.append((trading_dates[start_idx], trading_dates[end_idx]))

    return results


def _make_base_config(banks, start, end, **overrides):
    defaults = dict(
        banks=banks, start_date=start, end_date=end,
        capital=30000, grid_sell_pct=0.03, grid_buy_pct=0.02,
        trade_ratio=0.20, min_position_pct=0.50, rebalance_period='month',
        dividend_tax=True, commission_rate=0.0000854, commission_min=0.5,
        exempt_five=True, stamp_duty_rate=0.0005, slippage=0,
        limit_check=True, enable_rebalance=True, enable_grid=True,
        grid_max_loops=5, dividend_reinvest=False,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _compute_stats(excess_returns):
    if not excess_returns:
        return {'win_rate': 0, 'avg_excess': 0, 'median_excess': 0, 'p_value': 1.0, 'n': 0}
    arr = [r for r in excess_returns if r is not None and not (r != r)]
    if not arr:
        return {'win_rate': 0, 'avg_excess': 0, 'median_excess': 0, 'p_value': 1.0, 'n': 0}
    wins = sum(1 for r in arr if r > 0)
    win_rate = wins / len(arr) * 100
    avg_excess = sum(arr) / len(arr)
    sorted_arr = sorted(arr)
    mid = len(sorted_arr) // 2
    median_excess = sorted_arr[mid] if len(sorted_arr) % 2 == 1 else (sorted_arr[mid - 1] + sorted_arr[mid]) / 2
    try:
        _, p_value = stats.wilcoxon(arr)
    except Exception:
        p_value = 1.0
    return {'win_rate': win_rate, 'avg_excess': avg_excess, 'median_excess': median_excess, 'p_value': p_value, 'n': len(arr)}


def _run_experiment_core(periods_by_category, strategy_a_fn, strategy_b_fn, label_a="策略A", label_b="策略B"):
    result = {}
    for cat, periods in periods_by_category.items():
        excess_returns = []
        for start, end in periods:
            try:
                ra = strategy_a_fn(start, end)
                rb = strategy_b_fn(start, end)
                if ra and rb and ra.trading_days > 0 and rb.trading_days > 0:
                    excess_returns.append(ra.annual_return - rb.annual_return)
            except Exception:
                continue
        result[cat] = {
            'stats': _compute_stats(excess_returns),
            'excess_returns': excess_returns,
            'label_a': label_a,
            'label_b': label_b,
        }
    return result


def _run_experiment_8_core(periods_by_category, strategy_fn, benchmark_fn):
    result = {}
    for cat, periods in periods_by_category.items():
        dd_wins = 0
        sharpe_wins = 0
        total = 0
        drawdown_diffs = []
        sharpe_diffs = []
        for start, end in periods:
            try:
                ra = strategy_fn(start, end)
                rb = benchmark_fn(start, end)
                if ra and rb and ra.trading_days > 0 and rb.trading_days > 0:
                    total += 1
                    if ra.max_drawdown > rb.max_drawdown:
                        dd_wins += 1
                    if ra.sharpe_ratio > rb.sharpe_ratio:
                        sharpe_wins += 1
                    drawdown_diffs.append(ra.max_drawdown - rb.max_drawdown)
                    sharpe_diffs.append(ra.sharpe_ratio - rb.sharpe_ratio)
            except Exception:
                continue
        result[cat] = {
            'dd_win_rate': dd_wins / total * 100 if total > 0 else 0,
            'sharpe_win_rate': sharpe_wins / total * 100 if total > 0 else 0,
            'n': total,
            'drawdown_diffs': drawdown_diffs,
            'sharpe_diffs': sharpe_diffs,
        }
    return result


def run_experiment_1(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED) for cat in CATEGORY_RANGES}

    def strategy_a(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end), preloaded)
    def benchmark(start, end):
        return _run_bh(_make_base_config(get_banks_for_period(start), start, end), preloaded)
    return _run_experiment_core(periods_by_category, strategy_a, benchmark, "网格再平准策略", "买入持有")


def run_experiment_2(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 2) for cat in CATEGORY_RANGES}

    def strategy_a(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, grid_sell_pct=0.03, grid_buy_pct=0.02), preloaded)
    def strategy_b(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, grid_sell_pct=0.02, grid_buy_pct=0.02), preloaded)
    return _run_experiment_core(periods_by_category, strategy_a, strategy_b, "卖3%买2%", "卖2%买2%")


def run_experiment_3(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 3) for cat in CATEGORY_RANGES}

    def strategy_a(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, trade_ratio=0.20), preloaded)
    def strategy_b1(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, trade_ratio=0.30), preloaded)
    def strategy_b2(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, trade_ratio=0.10), preloaded)
    return {'vs_30': _run_experiment_core(periods_by_category, strategy_a, strategy_b1, "单次20%", "单次30%"),
            'vs_10': _run_experiment_core(periods_by_category, strategy_a, strategy_b2, "单次20%", "单次10%")}


def run_experiment_4(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 4) for cat in CATEGORY_RANGES}
    result = {}
    for cat, periods in periods_by_category.items():
        month_wins = quarter_wins = year_wins = total = 0
        for start, end in periods:
            try:
                banks = get_banks_for_period(start)
                rm = _run_bt(_make_base_config(banks, start, end, rebalance_period='month'), preloaded)
                rq = _run_bt(_make_base_config(banks, start, end, rebalance_period='quarter'), preloaded)
                ry = _run_bt(_make_base_config(banks, start, end, rebalance_period='year'), preloaded)
                if rm and rq and ry and rm.trading_days > 0:
                    total += 1
                    best = max(rm.annual_return, rq.annual_return, ry.annual_return)
                    if rm.annual_return == best: month_wins += 1
                    if rq.annual_return == best: quarter_wins += 1
                    if ry.annual_return == best: year_wins += 1
            except Exception:
                continue
        result[cat] = {
            'month_win_rate': month_wins / total * 100 if total > 0 else 0,
            'quarter_win_rate': quarter_wins / total * 100 if total > 0 else 0,
            'year_win_rate': year_wins / total * 100 if total > 0 else 0,
            'n': total,
        }
    return result


def run_experiment_5(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 5) for cat in CATEGORY_RANGES}

    def strategy_a(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, grid_sell_pct=0.03, grid_buy_pct=0.02), preloaded)
    def strategy_b1(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, grid_sell_pct=0.025, grid_buy_pct=0.025), preloaded)
    def strategy_b2(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, grid_sell_pct=0.02, grid_buy_pct=0.03), preloaded)
    return {'vs_symmetric': _run_experiment_core(periods_by_category, strategy_a, strategy_b1, "卖3%买2%", "卖2.5%买2.5%"),
            'vs_early_stop': _run_experiment_core(periods_by_category, strategy_a, strategy_b2, "卖3%买2%", "卖2%买3%")}


def run_experiment_6(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 6) for cat in CATEGORY_RANGES}

    def strategy_a(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, min_position_pct=0.50), preloaded)
    def strategy_b1(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, min_position_pct=0.30), preloaded)
    def strategy_b2(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, min_position_pct=0.70), preloaded)
    return {'vs_30': _run_experiment_core(periods_by_category, strategy_a, strategy_b1, "底仓50%", "底仓30%"),
            'vs_70': _run_experiment_core(periods_by_category, strategy_a, strategy_b2, "底仓50%", "底仓70%")}


def run_experiment_7(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 7) for cat in CATEGORY_RANGES}

    def strategy_a(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end), preloaded)
    def strategy_b(start, end):
        cfg = _make_base_config(["601398"], start, end, enable_rebalance=False)
        return _run_sg("601398", cfg, preloaded)
    return _run_experiment_core(periods_by_category, strategy_a, strategy_b, "多标的再平准", "单标的纯网格(工行)")


def run_experiment_8(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 8) for cat in CATEGORY_RANGES}

    def strategy_a(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end), preloaded)
    def benchmark(start, end):
        return _run_bh(_make_base_config(get_banks_for_period(start), start, end), preloaded)
    return _run_experiment_8_core(periods_by_category, strategy_a, benchmark)


def run_experiment_A(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    capitals = [10000, 30000, 100000, 500000]
    result = {}
    for cap in capitals:
        periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 100 + cap) for cat in CATEGORY_RANGES}
        def make_strategy(capital_val):
            def strategy(start, end):
                return _run_bt(_make_base_config(get_banks_for_period(start), start, end, capital=capital_val), preloaded)
            return strategy
        def benchmark(start, end):
            return _run_bh(_make_base_config(get_banks_for_period(start), start, end, capital=cap), preloaded)
        result[cap] = _run_experiment_core(periods_by_category, make_strategy(cap), benchmark, f"策略({cap}元)", "买入持有")
    return result


def run_experiment_B(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    combos = {
        "2只(工行+建行)": ["601398", "601939"],
        "4只(工建农中)": ["601398", "601939", "601288", "601988"],
        "5只(前5大行)": FIVE_BANK_CODES,
        "6只(全部)": SIX_BANK_CODES,
    }
    result = {}
    for name, bank_list in combos.items():
        periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 200 + hash(name)) for cat in CATEGORY_RANGES}
        cat_result = {}
        for cat, periods in periods_by_category.items():
            annual_returns = []
            max_dds = []
            for start, end in periods:
                try:
                    r = _run_bt(_make_base_config(bank_list, start, end), preloaded)
                    if r and r.trading_days > 0:
                        annual_returns.append(r.annual_return)
                        max_dds.append(r.max_drawdown)
                except Exception:
                    continue
            sorted_ar = sorted(annual_returns) if annual_returns else [0]
            mid = len(sorted_ar) // 2
            median_ar = sorted_ar[mid] if len(sorted_ar) % 2 == 1 else (sorted_ar[mid - 1] + sorted_ar[mid]) / 2
            sorted_dd = sorted(max_dds) if max_dds else [0]
            mid = len(sorted_dd) // 2
            median_dd = sorted_dd[mid] if len(sorted_dd) % 2 == 1 else (sorted_dd[mid - 1] + sorted_dd[mid]) / 2
            cat_result[cat] = {'median_annual_return': median_ar, 'median_max_drawdown': median_dd, 'n': len(annual_returns), 'annual_returns': annual_returns}
        result[name] = cat_result
    return result


def run_experiment_C(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    index_df = load_index_data()
    index_dict = {}
    for _, row in index_df.iterrows():
        d = row['Date'].strftime("%Y-%m-%d")
        index_dict[d] = float(row['Close'])

    periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 300) for cat in CATEGORY_RANGES}
    result = {}
    for cat, periods in periods_by_category.items():
        bull_wins = bull_total = bear_wins = bear_total = flat_wins = flat_total = 0
        for start, end in periods:
            try:
                banks = get_banks_for_period(start)
                cfg = _make_base_config(banks, start, end)
                ra = _run_bt(cfg, preloaded)
                rb = _run_bh(cfg, preloaded)
                if not ra or not rb or ra.trading_days == 0:
                    continue
                idx_start = index_dict.get(start)
                idx_end = index_dict.get(end)
                idx_change = (idx_end / idx_start - 1) * 100 if idx_start and idx_end and idx_start > 0 else 0
                won = ra.annual_return > rb.annual_return
                if idx_change > 20:
                    bull_total += 1
                    if won: bull_wins += 1
                elif idx_change < -20:
                    bear_total += 1
                    if won: bear_wins += 1
                else:
                    flat_total += 1
                    if won: flat_wins += 1
            except Exception:
                continue
        result[cat] = {
            'bull': {'win_rate': bull_wins / bull_total * 100 if bull_total > 0 else 0, 'n': bull_total},
            'bear': {'win_rate': bear_wins / bear_total * 100 if bear_total > 0 else 0, 'n': bear_total},
            'flat': {'win_rate': flat_wins / flat_total * 100 if flat_total > 0 else 0, 'n': flat_total},
        }
    return result


def run_experiment_D(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 400) for cat in CATEGORY_RANGES}

    def strategy_cash(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, dividend_reinvest=False), preloaded)
    def strategy_reinvest(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, dividend_reinvest=True), preloaded)
    return _run_experiment_core(periods_by_category, strategy_cash, strategy_reinvest, "分红落袋", "分红再投资")


def run_experiment_E(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    cost_configs = {
        "免5(0.5起)": {'commission_rate': 0.0000854, 'commission_min': 0.5, 'exempt_five': True},
        "不免5(5起)": {'commission_rate': 0.0000854, 'commission_min': 5.0, 'exempt_five': False},
        "高佣金(万3,5起)": {'commission_rate': 0.0003, 'commission_min': 5.0, 'exempt_five': False},
    }
    result = {}
    for name, cost_params in cost_configs.items():
        periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 500 + hash(name)) for cat in CATEGORY_RANGES}
        def make_strategy(params):
            def strategy(start, end):
                return _run_bt(_make_base_config(get_banks_for_period(start), start, end, **params), preloaded)
            return strategy
        def benchmark(start, end):
            return _run_bh(_make_base_config(get_banks_for_period(start), start, end, **cost_params), preloaded)
        result[name] = _run_experiment_core(periods_by_category, make_strategy(cost_params), benchmark, name, "买入持有")
    return result


def run_experiment_F(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 600) for cat in CATEGORY_RANGES}

    def strategy_rebalance_only(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, enable_grid=False), preloaded)
    def strategy_full(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end), preloaded)
    return _run_experiment_core(periods_by_category, strategy_rebalance_only, strategy_full, "纯再平准(无网格)", "再平准+网格")


def run_experiment_G(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 700) for cat in CATEGORY_RANGES}

    def strategy_a(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, grid_max_loops=5), preloaded)
    def strategy_b1(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, grid_max_loops=1), preloaded)
    def strategy_b2(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, grid_max_loops=10), preloaded)
    return {'vs_1': _run_experiment_core(periods_by_category, strategy_a, strategy_b1, "循环5次", "循环1次"),
            'vs_10': _run_experiment_core(periods_by_category, strategy_a, strategy_b2, "循环5次", "循环10次")}


def run_experiment_H(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    grid_combos = {
        "卖1%买0.5%": (0.01, 0.005),
        "卖2%买1%": (0.02, 0.01),
        "卖3%买2%": (0.03, 0.02),
        "卖5%买3%": (0.05, 0.03),
        "卖8%买5%": (0.08, 0.05),
    }
    result = {}
    for i, (name, (sell_pct, buy_pct)) in enumerate(grid_combos.items()):
        periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 800 + i) for cat in CATEGORY_RANGES}
        cat_result = {}
        for cat, periods in periods_by_category.items():
            annual_returns = []
            bh_returns = []
            for start, end in periods:
                try:
                    banks = get_banks_for_period(start)
                    r = _run_bt(_make_base_config(banks, start, end, grid_sell_pct=sell_pct, grid_buy_pct=buy_pct), preloaded)
                    rb = _run_bh(_make_base_config(banks, start, end, grid_sell_pct=sell_pct, grid_buy_pct=buy_pct), preloaded)
                    if r and rb and r.trading_days > 0 and rb.trading_days > 0:
                        annual_returns.append(r.annual_return)
                        bh_returns.append(rb.annual_return)
                except Exception:
                    continue
            excess = [a - b for a, b in zip(annual_returns, bh_returns)]
            sorted_ar = sorted(annual_returns) if annual_returns else [0]
            mid = len(sorted_ar) // 2
            median_ar = sorted_ar[mid] if len(sorted_ar) % 2 == 1 else (sorted_ar[mid - 1] + sorted_ar[mid]) / 2
            cat_result[cat] = {
                'median_annual_return': median_ar,
                'n': len(annual_returns),
                'annual_returns': annual_returns,
                'win_rate_vs_bh': sum(1 for e in excess if e > 0) / len(excess) * 100 if excess else 0,
                'median_excess': sorted(excess)[len(excess) // 2] if excess else 0,
                'excess_returns': excess,
            }
        result[name] = cat_result
    return result


def run_experiment_I(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 900) for cat in CATEGORY_RANGES}

    def strategy_a(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, slippage=0), preloaded)
    def strategy_b1(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, slippage=0.001), preloaded)
    def strategy_b2(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, slippage=0.002), preloaded)
    return {'vs_01pct': _run_experiment_core(periods_by_category, strategy_a, strategy_b1, "无滑点", "0.1%滑点"),
            'vs_02pct': _run_experiment_core(periods_by_category, strategy_a, strategy_b2, "无滑点", "0.2%滑点")}


def run_experiment_J(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 1000) for cat in CATEGORY_RANGES}

    def strategy_monthly(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, rebalance_period='month'), preloaded)
    def strategy_none(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, rebalance_period='none'), preloaded)
    return _run_experiment_core(periods_by_category, strategy_monthly, strategy_none, "月度再平准+网格", "永不再平准+网格")


def run_experiment_K(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    sample_sizes = [200, 500, 1000, 2000]
    cat = 'medium_long'
    result = {}
    for n in sample_sizes:
        periods = generate_random_periods(n, cat, trading_dates, seed=RANDOM_SEED + 1100)
        excess_returns = []
        for start, end in periods:
            try:
                banks = get_banks_for_period(start)
                ra = _run_bt(_make_base_config(banks, start, end), preloaded)
                rb = _run_bh(_make_base_config(banks, start, end), preloaded)
                if ra and rb and ra.trading_days > 0 and rb.trading_days > 0:
                    excess_returns.append(ra.annual_return - rb.annual_return)
            except Exception:
                continue
        stats = _compute_stats(excess_returns)
        import numpy as np
        variance = float(np.var(excess_returns)) if excess_returns else 0
        result[n] = {
            'stats': stats,
            'excess_returns': excess_returns,
            'variance': variance,
            'category': cat,
        }
    return result


def run_experiment_L(trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    stress_periods = [
        ("2015年股灾", "2015-06-12", "2015-08-26"),
        ("2018年熊市", "2018-01-24", "2019-01-04"),
        ("2020年新冠", "2020-01-02", "2020-03-31"),
        ("2022年回调", "2022-01-04", "2022-11-01"),
    ]
    result = {}
    for name, start, end in stress_periods:
        try:
            banks = get_banks_for_period(start)
            ra = _run_bt(_make_base_config(banks, start, end), preloaded)
            rb = _run_bh(_make_base_config(banks, start, end), preloaded)
            if ra and rb and ra.trading_days > 0 and rb.trading_days > 0:
                equity_curve = ra.equity_curve
                peak = equity_curve[0]
                max_dd = 0
                recovery_idx = None
                for idx, v in enumerate(equity_curve):
                    if v > peak:
                        peak = v
                    dd = (v - peak) / peak
                    if dd < max_dd:
                        max_dd = dd
                        recovery_idx = None
                    elif recovery_idx is None and v >= peak:
                        recovery_idx = idx
                result[name] = {
                    'strategy_annual': ra.annual_return,
                    'bh_annual': rb.annual_return,
                    'strategy_dd': ra.max_drawdown,
                    'bh_dd': rb.max_drawdown,
                    'strategy_sharpe': ra.sharpe_ratio,
                    'bh_sharpe': rb.sharpe_ratio,
                    'strategy_trades': ra.total_trades,
                    'strategy_fees': ra.total_fees,
                    'trading_days': ra.trading_days,
                    'max_dd_depth': max_dd * 100,
                    'recovery_days': recovery_idx if recovery_idx else ra.trading_days,
                    'excess': ra.annual_return - rb.annual_return,
                }
            else:
                result[name] = {'error': '回测数据不足'}
        except Exception as e:
            result[name] = {'error': str(e)}
    return result


def run_all_experiments(n_samples=1000):
    print("加载交易日历...")
    trading_dates = get_all_trading_dates()
    print(f"交易日历加载完成: {len(trading_dates)} 个交易日 ({trading_dates[0]} ~ {trading_dates[-1]})")

    print("预加载全部数据...")
    preloaded = preload_all_data()
    print(f"数据预加载完成: {len(preloaded['dates'])} 个交易日")

    all_results = {}
    experiments = [
        ("实验1: 基础超额测试", run_experiment_1),
        ("实验2: 网格步长对比", run_experiment_2),
        ("实验3: 单次调仓力度对比", run_experiment_3),
        ("实验4: 再平准频率对比", run_experiment_4),
        ("实验5: 网格对称性对比", run_experiment_5),
        ("实验6: 最低底仓限制对比", run_experiment_6),
        ("实验7: 多标的vs单标的", run_experiment_7),
        ("实验8: 抗回撤与风险控制", run_experiment_8),
        ("补充A: 初始资金敏感度", run_experiment_A),
        ("补充B: 标的数量敏感度", run_experiment_B),
        ("补充C: 市场环境分环境", run_experiment_C),
        ("补充D: 分红处理方式对比", run_experiment_D),
        ("补充E: 交易成本敏感度", run_experiment_E),
        ("补充F: 纯再平准(无网格)", run_experiment_F),
        ("补充G: 网格最大循环次数", run_experiment_G),
        ("补充H: 大范围网格步长扫描", run_experiment_H),
        ("补充I: 滑点影响测试", run_experiment_I),
        ("补充J: 永不再平准(纯网格)", run_experiment_J),
        ("补充K: 蒙特卡洛收敛性验证", run_experiment_K),
        ("补充L: 极端行情压力测试", run_experiment_L),
    ]

    for name, fn in experiments:
        print(f"\n{'='*60}")
        print(f"开始运行: {name}")
        t0 = time.time()
        try:
            if "补充L" in name:
                result = fn(trading_dates=trading_dates, preloaded=preloaded)
            else:
                result = fn(n_samples=n_samples, trading_dates=trading_dates, preloaded=preloaded)
            all_results[name] = result
            elapsed = time.time() - t0
            print(f"完成: {name} ({elapsed:.1f}秒)")
        except Exception as e:
            print(f"错误: {name} - {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = None

    return all_results


DENSE_GRID_SELL = 0.01
DENSE_GRID_BUY = 0.005


def run_experiment_M(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 1200) for cat in CATEGORY_RANGES}

    def dense_grid(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, grid_sell_pct=DENSE_GRID_SELL, grid_buy_pct=DENSE_GRID_BUY), preloaded)

    result = {}
    for label, sell, buy in [("vs_2_1", 0.02, 0.01), ("vs_3_2", 0.03, 0.02), ("vs_5_3", 0.05, 0.03)]:
        def other_grid(start, end, s=sell, b=buy):
            return _run_bt(_make_base_config(get_banks_for_period(start), start, end, grid_sell_pct=s, grid_buy_pct=b), preloaded)
        result[label] = _run_experiment_core(periods_by_category, dense_grid, other_grid, f"卖1%买0.5%", f"卖{sell*100:.0f}%买{buy*100:.0f}%")
    return result


def run_experiment_N(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    ratios = [0.10, 0.20, 0.30, 0.40]
    result = {}
    for i, tr in enumerate(ratios):
        periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 1300 + i) for cat in CATEGORY_RANGES}
        cat_result = {}
        for cat, periods in periods_by_category.items():
            annual_returns = []
            max_dds = []
            trade_counts = []
            for start, end in periods:
                try:
                    r = _run_bt(_make_base_config(get_banks_for_period(start), start, end, grid_sell_pct=DENSE_GRID_SELL, grid_buy_pct=DENSE_GRID_BUY, trade_ratio=tr), preloaded)
                    if r and r.trading_days > 0:
                        annual_returns.append(r.annual_return)
                        max_dds.append(r.max_drawdown)
                        trade_counts.append(r.total_trades)
                except Exception:
                    continue
            sorted_ar = sorted(annual_returns) if annual_returns else [0]
            mid = len(sorted_ar) // 2
            median_ar = sorted_ar[mid] if len(sorted_ar) % 2 == 1 else (sorted_ar[mid - 1] + sorted_ar[mid]) / 2
            sorted_dd = sorted(max_dds) if max_dds else [0]
            mid = len(sorted_dd) // 2
            median_dd = sorted_dd[mid] if len(sorted_dd) % 2 == 1 else (sorted_dd[mid - 1] + sorted_dd[mid]) / 2
            avg_trades = sum(trade_counts) / len(trade_counts) if trade_counts else 0
            cat_result[cat] = {'median_annual_return': median_ar, 'median_max_drawdown': median_dd, 'n': len(annual_returns), 'avg_trades': avg_trades}
        result[f"单次{int(tr*100)}%"] = cat_result
    return result


def run_experiment_O(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    positions = [0.30, 0.50, 0.70, 0.80]
    result = {}
    for i, mp in enumerate(positions):
        periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 1400 + i) for cat in CATEGORY_RANGES}
        cat_result = {}
        for cat, periods in periods_by_category.items():
            annual_returns = []
            max_dds = []
            for start, end in periods:
                try:
                    r = _run_bt(_make_base_config(get_banks_for_period(start), start, end, grid_sell_pct=DENSE_GRID_SELL, grid_buy_pct=DENSE_GRID_BUY, min_position_pct=mp), preloaded)
                    if r and r.trading_days > 0:
                        annual_returns.append(r.annual_return)
                        max_dds.append(r.max_drawdown)
                except Exception:
                    continue
            sorted_ar = sorted(annual_returns) if annual_returns else [0]
            mid = len(sorted_ar) // 2
            median_ar = sorted_ar[mid] if len(sorted_ar) % 2 == 1 else (sorted_ar[mid - 1] + sorted_ar[mid]) / 2
            sorted_dd = sorted(max_dds) if max_dds else [0]
            mid = len(sorted_dd) // 2
            median_dd = sorted_dd[mid] if len(sorted_dd) % 2 == 1 else (sorted_dd[mid - 1] + sorted_dd[mid]) / 2
            cat_result[cat] = {'median_annual_return': median_ar, 'median_max_drawdown': median_dd, 'n': len(annual_returns)}
        result[f"底仓{int(mp*100)}%"] = cat_result
    return result


def run_experiment_P(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 1500) for cat in CATEGORY_RANGES}
    result = {}
    for cat, periods in periods_by_category.items():
        wins = {1: 0, 5: 0, 10: 0, 999: 0}
        total = 0
        annual_by_loops = {1: [], 5: [], 10: [], 999: []}
        for start, end in periods:
            try:
                banks = get_banks_for_period(start)
                results = {}
                for loops in [1, 5, 10, 999]:
                    r = _run_bt(_make_base_config(banks, start, end, grid_sell_pct=DENSE_GRID_SELL, grid_buy_pct=DENSE_GRID_BUY, grid_max_loops=loops), preloaded)
                    if r and r.trading_days > 0:
                        results[loops] = r
                        annual_by_loops[loops].append(r.annual_return)
                if len(results) == 4:
                    total += 1
                    best = max(results.values(), key=lambda x: x.annual_return)
                    for loops, r in results.items():
                        if r.annual_return == best.annual_return:
                            wins[loops] += 1
            except Exception:
                continue
        cat_result = {'n': total}
        for loops in [1, 5, 10, 999]:
            cat_result[f'loops_{loops}_win_rate'] = wins[loops] / total * 100 if total > 0 else 0
            arr = annual_by_loops[loops]
            if arr:
                sorted_arr = sorted(arr)
                mid = len(sorted_arr) // 2
                cat_result[f'loops_{loops}_median_annual'] = sorted_arr[mid] if len(sorted_arr) % 2 == 1 else (sorted_arr[mid - 1] + sorted_arr[mid]) / 2
            else:
                cat_result[f'loops_{loops}_median_annual'] = 0
        result[cat] = cat_result
    return result


def run_experiment_Q(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    cost_configs = {
        "免5(0.5起)": {'commission_rate': 0.0000854, 'commission_min': 0.5, 'exempt_five': True},
        "不免5(5起)": {'commission_rate': 0.0000854, 'commission_min': 5.0, 'exempt_five': False},
        "高佣金(万3,5起)": {'commission_rate': 0.0003, 'commission_min': 5.0, 'exempt_five': False},
    }
    result = {}
    for i, (name, cost_params) in enumerate(cost_configs.items()):
        periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 1600 + i) for cat in CATEGORY_RANGES}
        cat_result = {}
        for cat, periods in periods_by_category.items():
            annual_returns = []
            fee_ratios = []
            for start, end in periods:
                try:
                    r = _run_bt(_make_base_config(get_banks_for_period(start), start, end, grid_sell_pct=DENSE_GRID_SELL, grid_buy_pct=DENSE_GRID_BUY, **cost_params), preloaded)
                    if r and r.trading_days > 0:
                        annual_returns.append(r.annual_return)
                        total_gain = r.final_value - 30000
                        fee_ratio = r.total_fees / total_gain * 100 if total_gain > 0 else 0
                        fee_ratios.append(fee_ratio)
                except Exception:
                    continue
            sorted_ar = sorted(annual_returns) if annual_returns else [0]
            mid = len(sorted_ar) // 2
            median_ar = sorted_ar[mid] if len(sorted_ar) % 2 == 1 else (sorted_ar[mid - 1] + sorted_ar[mid]) / 2
            avg_fee_ratio = sum(fee_ratios) / len(fee_ratios) if fee_ratios else 0
            cat_result[cat] = {'median_annual_return': median_ar, 'n': len(annual_returns), 'avg_fee_ratio': avg_fee_ratio, 'annual_returns': annual_returns}
        result[name] = cat_result
    return result


def run_experiment_R(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    capitals = [10000, 30000, 100000, 500000]
    result = {}
    for i, cap in enumerate(capitals):
        periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 1700 + i) for cat in CATEGORY_RANGES}
        cat_result = {}
        for cat, periods in periods_by_category.items():
            annual_returns = []
            fee_ratios = []
            for start, end in periods:
                try:
                    r = _run_bt(_make_base_config(get_banks_for_period(start), start, end, grid_sell_pct=DENSE_GRID_SELL, grid_buy_pct=DENSE_GRID_BUY, capital=cap), preloaded)
                    if r and r.trading_days > 0:
                        annual_returns.append(r.annual_return)
                        total_gain = r.final_value - cap
                        fee_ratio = r.total_fees / total_gain * 100 if total_gain > 0 else 0
                        fee_ratios.append(fee_ratio)
                except Exception:
                    continue
            sorted_ar = sorted(annual_returns) if annual_returns else [0]
            mid = len(sorted_ar) // 2
            median_ar = sorted_ar[mid] if len(sorted_ar) % 2 == 1 else (sorted_ar[mid - 1] + sorted_ar[mid]) / 2
            avg_fee_ratio = sum(fee_ratios) / len(fee_ratios) if fee_ratios else 0
            cat_result[cat] = {'median_annual_return': median_ar, 'n': len(annual_returns), 'avg_fee_ratio': avg_fee_ratio}
        result[f"{cap}元"] = cat_result
    return result


def run_experiment_S(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    slippages = [0, 0.0005, 0.001, 0.002]
    result = {}
    for i, slip in enumerate(slippages):
        periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 1800 + i) for cat in CATEGORY_RANGES}
        cat_result = {}
        for cat, periods in periods_by_category.items():
            annual_returns = []
            for start, end in periods:
                try:
                    r = _run_bt(_make_base_config(get_banks_for_period(start), start, end, grid_sell_pct=DENSE_GRID_SELL, grid_buy_pct=DENSE_GRID_BUY, slippage=slip), preloaded)
                    if r and r.trading_days > 0:
                        annual_returns.append(r.annual_return)
                except Exception:
                    continue
            sorted_ar = sorted(annual_returns) if annual_returns else [0]
            mid = len(sorted_ar) // 2
            median_ar = sorted_ar[mid] if len(sorted_ar) % 2 == 1 else (sorted_ar[mid - 1] + sorted_ar[mid]) / 2
            cat_result[cat] = {'median_annual_return': median_ar, 'n': len(annual_returns), 'annual_returns': annual_returns}
        result[f"滑点{slip*100:.2f}%"] = cat_result
    return result


def run_experiment_T(n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 1900) for cat in CATEGORY_RANGES}
    result = {}
    for cat, periods in periods_by_category.items():
        freq_results = {}
        for freq in ['month', 'quarter', 'year', 'none']:
            annual_returns = []
            for start, end in periods:
                try:
                    r = _run_bt(_make_base_config(get_banks_for_period(start), start, end, grid_sell_pct=DENSE_GRID_SELL, grid_buy_pct=DENSE_GRID_BUY, rebalance_period=freq), preloaded)
                    if r and r.trading_days > 0:
                        annual_returns.append(r.annual_return)
                except Exception:
                    continue
            sorted_ar = sorted(annual_returns) if annual_returns else [0]
            mid = len(sorted_ar) // 2
            median_ar = sorted_ar[mid] if len(sorted_ar) % 2 == 1 else (sorted_ar[mid - 1] + sorted_ar[mid]) / 2
            freq_results[freq] = {'median_annual_return': median_ar, 'n': len(annual_returns), 'annual_returns': annual_returns}
        result[cat] = freq_results
    return result


def run_experiment_U(trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    stress_periods = [
        ("2015年股灾", "2015-06-12", "2015-08-26"),
        ("2018年熊市", "2018-01-24", "2019-01-04"),
        ("2020年新冠", "2020-01-02", "2020-03-31"),
        ("2022年回调", "2022-01-04", "2022-11-01"),
        ("2024年9月暴涨", "2024-09-24", "2024-10-08"),
    ]
    result = {}
    for name, start, end in stress_periods:
        try:
            banks = get_banks_for_period(start)
            r_dense = _run_bt(_make_base_config(banks, start, end, grid_sell_pct=DENSE_GRID_SELL, grid_buy_pct=DENSE_GRID_BUY), preloaded)
            r_default = _run_bt(_make_base_config(banks, start, end, grid_sell_pct=0.03, grid_buy_pct=0.02), preloaded)
            if r_dense and r_default and r_dense.trading_days > 0 and r_default.trading_days > 0:
                result[name] = {
                    'dense_annual': r_dense.annual_return,
                    'default_annual': r_default.annual_return,
                    'dense_dd': r_dense.max_drawdown,
                    'default_dd': r_default.max_drawdown,
                    'dense_trades': r_dense.total_trades,
                    'default_trades': r_default.total_trades,
                    'dense_fees': r_dense.total_fees,
                    'default_fees': r_default.total_fees,
                    'dense_sharpe': r_dense.sharpe_ratio,
                    'default_sharpe': r_default.sharpe_ratio,
                    'trading_days': r_dense.trading_days,
                    'excess': r_dense.annual_return - r_default.annual_return,
                }
            else:
                result[name] = {'error': '回测数据不足'}
        except Exception as e:
            result[name] = {'error': str(e)}
    return result


def run_experiment_V(optimal_params, n_samples=1000, trading_dates=None, preloaded=None):
    if trading_dates is None:
        trading_dates = get_all_trading_dates()
    periods_by_category = {cat: generate_random_periods(n_samples, cat, trading_dates, seed=RANDOM_SEED + 2000) for cat in CATEGORY_RANGES}

    def dense_optimal(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end, grid_sell_pct=DENSE_GRID_SELL, grid_buy_pct=DENSE_GRID_BUY, **optimal_params), preloaded)
    def default_strategy(start, end):
        return _run_bt(_make_base_config(get_banks_for_period(start), start, end), preloaded)
    return _run_experiment_core(periods_by_category, dense_optimal, default_strategy, "极密网格(最优参数)", "默认策略(卖3%买2%)")


def run_dense_grid_experiments(n_samples=1000):
    print("加载交易日历...")
    trading_dates = get_all_trading_dates()
    print(f"交易日历加载完成: {len(trading_dates)} 个交易日 ({trading_dates[0]} ~ {trading_dates[-1]})")

    print("预加载全部数据...")
    preloaded = preload_all_data()
    print(f"数据预加载完成: {len(preloaded['dates'])} 个交易日")

    all_results = {}
    experiments = [
        ("测试M: 极密vs其他网格配对", run_experiment_M),
        ("测试N: 仓位敏感度", run_experiment_N),
        ("测试O: 底仓敏感度", run_experiment_O),
        ("测试P: 日内触发次数限制", run_experiment_P),
        ("测试Q: 手续费敏感度", run_experiment_Q),
        ("测试R: 初始资金敏感度", run_experiment_R),
        ("测试S: 滑点敏感度", run_experiment_S),
        ("测试T: 再平准频率", run_experiment_T),
        ("测试U: 极端行情压力测试", run_experiment_U),
    ]

    for name, fn in experiments:
        print(f"\n{'='*60}")
        print(f"开始运行: {name}")
        t0 = time.time()
        try:
            if "测试U" in name:
                result = fn(trading_dates=trading_dates, preloaded=preloaded)
            else:
                result = fn(n_samples=n_samples, trading_dates=trading_dates, preloaded=preloaded)
            all_results[name] = result
            elapsed = time.time() - t0
            print(f"完成: {name} ({elapsed:.1f}秒)")
        except Exception as e:
            print(f"错误: {name} - {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = None

    return all_results


def run_all_supplement_experiments(n_samples=1000):
    print("加载交易日历...")
    trading_dates = get_all_trading_dates()
    print(f"交易日历加载完成: {len(trading_dates)} 个交易日 ({trading_dates[0]} ~ {trading_dates[-1]})")

    print("预加载全部数据...")
    preloaded = preload_all_data()
    print(f"数据预加载完成: {len(preloaded['dates'])} 个交易日")

    all_results = {}
    experiments = [
        ("补充F: 纯再平准(无网格)", run_experiment_F),
        ("补充G: 网格最大循环次数", run_experiment_G),
        ("补充H: 大范围网格步长扫描", run_experiment_H),
        ("补充I: 滑点影响测试", run_experiment_I),
        ("补充J: 永不再平准(纯网格)", run_experiment_J),
        ("补充K: 蒙特卡洛收敛性验证", run_experiment_K),
        ("补充L: 极端行情压力测试", run_experiment_L),
    ]

    for name, fn in experiments:
        print(f"\n{'='*60}")
        print(f"开始运行: {name}")
        t0 = time.time()
        try:
            if "补充L" in name:
                result = fn(trading_dates=trading_dates, preloaded=preloaded)
            else:
                result = fn(n_samples=n_samples, trading_dates=trading_dates, preloaded=preloaded)
            all_results[name] = result
            elapsed = time.time() - t0
            print(f"完成: {name} ({elapsed:.1f}秒)")
        except Exception as e:
            print(f"错误: {name} - {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = None

    return all_results
