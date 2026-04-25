import math
import multiprocessing
import os
import time
from collections import defaultdict

import numpy as np
from scipy import stats

from .backtest_engine import BacktestConfig, run_backtest
from .config import (CATEGORY_LABELS, CATEGORY_RANGES, FIVE_BANK_CODES,
                     SIX_BANK_CODES, BANK_INDEX_CODE)
from .data_loader import get_all_trading_dates, load_index_data
from .monte_carlo_runner import generate_random_periods, preload_all_data

CATEGORY_ORDER = ['short', 'medium', 'medium_long', 'long']

PARAM_A_SELL = 0.012
PARAM_A_BUY = 0.005
PARAM_B_SELL = 0.0175
PARAM_B_BUY = 0.0075

_worker_preloaded = None


def _worker_init():
    global _worker_preloaded
    _worker_preloaded = preload_all_data()


def _get_banks(start_date):
    if start_date < "2019-12-10":
        return FIVE_BANK_CODES
    return SIX_BANK_CODES


def _make_config(banks, start, end, **overrides):
    defaults = dict(
        banks=banks, start_date=start, end_date=end,
        capital=30000, grid_sell_pct=0.012, grid_buy_pct=0.005,
        trade_ratio=0.30, min_position_pct=0.50, rebalance_period='month',
        dividend_tax=True, commission_rate=0.0000854, commission_min=0.5,
        exempt_five=True, stamp_duty_rate=0.0005, slippage=0,
        limit_check=True, enable_rebalance=True, enable_grid=True,
        grid_max_loops=999, dividend_reinvest=False,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _run_single_task(config_dict):
    config = BacktestConfig(**config_dict)
    try:
        r = run_backtest(config, preloaded_merged=_worker_preloaded['merged'],
                         preloaded_dates=_worker_preloaded['dates'])
        if r and r.trading_days > 0 and math.isfinite(r.annual_return) and math.isfinite(r.max_drawdown):
            return {'annual_return': r.annual_return, 'max_drawdown': r.max_drawdown,
                    'trading_days': r.trading_days, 'total_trades': r.total_trades}
    except Exception:
        pass
    return None


def _compute_pair_stats(excess_returns):
    if not excess_returns:
        return {'n': 0, 'win_rate': 0, 'median_excess': 0, 'p_value': 1.0}
    arr = [e for e in excess_returns if e is not None and math.isfinite(e)]
    if not arr:
        return {'n': 0, 'win_rate': 0, 'median_excess': 0, 'p_value': 1.0}
    wins = sum(1 for e in arr if e > 0)
    win_rate = wins / len(arr) * 100
    sorted_arr = sorted(arr)
    mid = len(sorted_arr) // 2
    median_excess = sorted_arr[mid] if len(sorted_arr) % 2 == 1 else (sorted_arr[mid - 1] + sorted_arr[mid]) / 2
    p_value = 1.0
    if len(arr) >= 10:
        try:
            _, p_value = stats.wilcoxon(arr, alternative='greater')
        except Exception:
            p_value = 1.0
    return {'n': len(arr), 'win_rate': win_rate, 'median_excess': median_excess, 'p_value': p_value}


def _run_paired_comparison(periods_by_category, make_config_a, make_config_b, n_workers):
    all_tasks = []
    task_meta = []
    for cat in CATEGORY_ORDER:
        for idx, (start, end) in enumerate(periods_by_category[cat]):
            cfg_a = make_config_a(start, end)
            cfg_b = make_config_b(start, end)
            da = {k: getattr(cfg_a, k) for k in cfg_a.__dataclass_fields__}
            db = {k: getattr(cfg_b, k) for k in cfg_b.__dataclass_fields__}
            all_tasks.append(da)
            task_meta.append((cat, idx, 'A'))
            all_tasks.append(db)
            task_meta.append((cat, idx, 'B'))

    with multiprocessing.Pool(processes=n_workers, initializer=_worker_init) as pool:
        raw_results = list(pool.imap_unordered(_run_single_task, all_tasks, chunksize=100))

    organized = defaultdict(lambda: defaultdict(dict))
    n_invalid = 0
    for i, result in enumerate(raw_results):
        cat, idx, strategy = task_meta[i]
        if result is None:
            n_invalid += 1
            continue
        organized[cat][idx][strategy] = result

    result = {}
    for cat in CATEGORY_ORDER:
        a_returns = []
        b_returns = []
        excess_returns = []
        a_dds = []
        b_dds = []
        for idx in sorted(organized[cat].keys()):
            d = organized[cat][idx]
            if 'A' in d and 'B' in d:
                a_returns.append(d['A']['annual_return'])
                b_returns.append(d['B']['annual_return'])
                a_dds.append(d['A']['max_drawdown'])
                b_dds.append(d['B']['max_drawdown'])
                excess_returns.append(d['A']['annual_return'] - d['B']['annual_return'])

        pair_stats = _compute_pair_stats(excess_returns)

        def _median(arr):
            if not arr:
                return 0
            s = sorted(arr)
            m = len(s) // 2
            return s[m] if len(s) % 2 == 1 else (s[m - 1] + s[m]) / 2

        result[cat] = {
            'n': pair_stats['n'],
            'a_median_ar': _median(a_returns),
            'b_median_ar': _median(b_returns),
            'a_median_dd': _median(a_dds),
            'b_median_dd': _median(b_dds),
            'a_returns': a_returns,
            'b_returns': b_returns,
            'excess_returns': excess_returns,
            'win_rate': pair_stats['win_rate'],
            'median_excess': pair_stats['median_excess'],
            'p_value': pair_stats['p_value'],
        }

    return result


def run_base_comparison(n_samples, trading_dates, n_workers):
    print("  基础对比: 卖1.2%买0.5% vs 卖1.75%买0.75%")
    periods_by_category = {}
    for cat in CATEGORY_RANGES:
        periods_by_category[cat] = generate_random_periods(
            n_samples, cat, trading_dates, seed=42 + 3000)

    def make_a(start, end):
        return _make_config(_get_banks(start), start, end,
                           grid_sell_pct=PARAM_A_SELL, grid_buy_pct=PARAM_A_BUY)

    def make_b(start, end):
        return _make_config(_get_banks(start), start, end,
                           grid_sell_pct=PARAM_B_SELL, grid_buy_pct=PARAM_B_BUY)

    return _run_paired_comparison(periods_by_category, make_a, make_b, n_workers)


def run_capital_comparison(n_samples, trading_dates, n_workers):
    print("  资金维度对比")
    capitals = [10000, 30000, 100000, 500000]
    result = {}
    for i, cap in enumerate(capitals):
        periods_by_category = {}
        for cat in CATEGORY_RANGES:
            periods_by_category[cat] = generate_random_periods(
                n_samples, cat, trading_dates, seed=42 + 3100 + i)

        def make_a(start, end, c=cap):
            return _make_config(_get_banks(start), start, end,
                               grid_sell_pct=PARAM_A_SELL, grid_buy_pct=PARAM_A_BUY, capital=c)

        def make_b(start, end, c=cap):
            return _make_config(_get_banks(start), start, end,
                               grid_sell_pct=PARAM_B_SELL, grid_buy_pct=PARAM_B_BUY, capital=c)

        result[str(cap)] = _run_paired_comparison(periods_by_category, make_a, make_b, n_workers)
    return result


def run_trade_ratio_comparison(n_samples, trading_dates, n_workers):
    print("  调仓比例对比")
    ratios = [0.10, 0.20, 0.30, 0.40]
    result = {}
    for i, tr in enumerate(ratios):
        periods_by_category = {}
        for cat in CATEGORY_RANGES:
            periods_by_category[cat] = generate_random_periods(
                n_samples, cat, trading_dates, seed=42 + 3200 + i)

        def make_a(start, end, r=tr):
            return _make_config(_get_banks(start), start, end,
                               grid_sell_pct=PARAM_A_SELL, grid_buy_pct=PARAM_A_BUY, trade_ratio=r)

        def make_b(start, end, r=tr):
            return _make_config(_get_banks(start), start, end,
                               grid_sell_pct=PARAM_B_SELL, grid_buy_pct=PARAM_B_BUY, trade_ratio=r)

        result[f"{int(tr*100)}%"] = _run_paired_comparison(periods_by_category, make_a, make_b, n_workers)
    return result


def run_min_position_comparison(n_samples, trading_dates, n_workers):
    print("  底仓比例对比")
    positions = [0.30, 0.50, 0.70, 0.80]
    result = {}
    for i, mp in enumerate(positions):
        periods_by_category = {}
        for cat in CATEGORY_RANGES:
            periods_by_category[cat] = generate_random_periods(
                n_samples, cat, trading_dates, seed=42 + 3300 + i)

        def make_a(start, end, p=mp):
            return _make_config(_get_banks(start), start, end,
                               grid_sell_pct=PARAM_A_SELL, grid_buy_pct=PARAM_A_BUY, min_position_pct=p)

        def make_b(start, end, p=mp):
            return _make_config(_get_banks(start), start, end,
                               grid_sell_pct=PARAM_B_SELL, grid_buy_pct=PARAM_B_BUY, min_position_pct=p)

        result[f"{int(mp*100)}%"] = _run_paired_comparison(periods_by_category, make_a, make_b, n_workers)
    return result


def run_rebalance_comparison(n_samples, trading_dates, n_workers):
    print("  再平准频率对比")
    freqs = [('月度', 'month'), ('季度', 'quarter'), ('年度', 'year'), ('无', 'none')]
    result = {}
    for i, (label, freq) in enumerate(freqs):
        periods_by_category = {}
        for cat in CATEGORY_RANGES:
            periods_by_category[cat] = generate_random_periods(
                n_samples, cat, trading_dates, seed=42 + 3400 + i)

        def make_a(start, end, f=freq):
            overrides = {'grid_sell_pct': PARAM_A_SELL, 'grid_buy_pct': PARAM_A_BUY}
            if f == 'none':
                overrides['enable_rebalance'] = False
            else:
                overrides['rebalance_period'] = f
            return _make_config(_get_banks(start), start, end, **overrides)

        def make_b(start, end, f=freq):
            overrides = {'grid_sell_pct': PARAM_B_SELL, 'grid_buy_pct': PARAM_B_BUY}
            if f == 'none':
                overrides['enable_rebalance'] = False
            else:
                overrides['rebalance_period'] = f
            return _make_config(_get_banks(start), start, end, **overrides)

        result[label] = _run_paired_comparison(periods_by_category, make_a, make_b, n_workers)
    return result


def run_market_env_comparison(n_samples, trading_dates, n_workers):
    print("  市场环境对比")
    index_df = load_index_data()
    index_dict = {}
    for _, row in index_df.iterrows():
        d = row['Date'].strftime("%Y-%m-%d")
        index_dict[d] = float(row['Close'])

    periods_by_category = {}
    for cat in CATEGORY_RANGES:
        periods_by_category[cat] = generate_random_periods(
            n_samples, cat, trading_dates, seed=42 + 3500)

    all_tasks = []
    task_meta = []
    for cat in CATEGORY_ORDER:
        for idx, (start, end) in enumerate(periods_by_category[cat]):
            banks = _get_banks(start)
            idx_start = index_dict.get(start)
            idx_end = index_dict.get(end)
            idx_change = (idx_end / idx_start - 1) * 100 if idx_start and idx_end and idx_start > 0 else 0
            if idx_change > 20:
                env = '牛市'
            elif idx_change < -20:
                env = '熊市'
            else:
                env = '震荡'

            cfg_a = _make_config(banks, start, end, grid_sell_pct=PARAM_A_SELL, grid_buy_pct=PARAM_A_BUY)
            cfg_b = _make_config(banks, start, end, grid_sell_pct=PARAM_B_SELL, grid_buy_pct=PARAM_B_BUY)
            da = {k: getattr(cfg_a, k) for k in cfg_a.__dataclass_fields__}
            db = {k: getattr(cfg_b, k) for k in cfg_b.__dataclass_fields__}
            all_tasks.append(da)
            task_meta.append((cat, idx, 'A', env))
            all_tasks.append(db)
            task_meta.append((cat, idx, 'B', env))

    with multiprocessing.Pool(processes=n_workers, initializer=_worker_init) as pool:
        raw_results = list(pool.imap_unordered(_run_single_task, all_tasks, chunksize=100))

    organized = defaultdict(lambda: defaultdict(dict))
    env_map = {}
    for i, result in enumerate(raw_results):
        cat, idx, strategy, env = task_meta[i]
        if result is None:
            continue
        organized[cat][idx][strategy] = result
        env_map[(cat, idx)] = env

    result = {}
    for env in ['牛市', '熊市', '震荡']:
        env_result = {}
        for cat in CATEGORY_ORDER:
            excess_returns = []
            a_returns = []
            b_returns = []
            for idx in sorted(organized[cat].keys()):
                if env_map.get((cat, idx)) != env:
                    continue
                d = organized[cat][idx]
                if 'A' in d and 'B' in d:
                    a_returns.append(d['A']['annual_return'])
                    b_returns.append(d['B']['annual_return'])
                    excess_returns.append(d['A']['annual_return'] - d['B']['annual_return'])

            pair_stats = _compute_pair_stats(excess_returns)

            def _median(arr):
                if not arr:
                    return 0
                s = sorted(arr)
                m = len(s) // 2
                return s[m] if len(s) % 2 == 1 else (s[m - 1] + s[m]) / 2

            env_result[cat] = {
                'n': pair_stats['n'],
                'a_median_ar': _median(a_returns),
                'b_median_ar': _median(b_returns),
                'a_returns': a_returns,
                'b_returns': b_returns,
                'excess_returns': excess_returns,
                'win_rate': pair_stats['win_rate'],
                'median_excess': pair_stats['median_excess'],
                'p_value': pair_stats['p_value'],
            }
        result[env] = env_result
    return result


def run_bank_count_comparison(n_samples, trading_dates, n_workers):
    print("  标的数量对比")
    combos = {
        "4只(工建农中)": ["601398", "601939", "601288", "601988"],
        "5只(前5大行)": FIVE_BANK_CODES,
        "6只(全部)": SIX_BANK_CODES,
    }
    result = {}
    for i, (name, bank_list) in enumerate(combos.items()):
        periods_by_category = {}
        for cat in CATEGORY_RANGES:
            periods_by_category[cat] = generate_random_periods(
                n_samples, cat, trading_dates, seed=42 + 3600 + i)

        def make_a(start, end, bl=bank_list):
            return _make_config(bl, start, end,
                               grid_sell_pct=PARAM_A_SELL, grid_buy_pct=PARAM_A_BUY)

        def make_b(start, end, bl=bank_list):
            return _make_config(bl, start, end,
                               grid_sell_pct=PARAM_B_SELL, grid_buy_pct=PARAM_B_BUY)

        result[name] = _run_paired_comparison(periods_by_category, make_a, make_b, n_workers)
    return result


def run_fee_comparison(n_samples, trading_dates, n_workers):
    print("  手续费档位对比")
    fee_configs = {
        "免5(0.5起)": {'commission_rate': 0.0000854, 'commission_min': 0.5, 'exempt_five': True},
        "不免5(5起)": {'commission_rate': 0.0000854, 'commission_min': 5.0, 'exempt_five': False},
        "高佣金(万3)": {'commission_rate': 0.0003, 'commission_min': 5.0, 'exempt_five': False},
    }
    result = {}
    for i, (name, fee_params) in enumerate(fee_configs.items()):
        periods_by_category = {}
        for cat in CATEGORY_RANGES:
            periods_by_category[cat] = generate_random_periods(
                n_samples, cat, trading_dates, seed=42 + 3700 + i)

        def make_a(start, end, fp=fee_params):
            return _make_config(_get_banks(start), start, end,
                               grid_sell_pct=PARAM_A_SELL, grid_buy_pct=PARAM_A_BUY, **fp)

        def make_b(start, end, fp=fee_params):
            return _make_config(_get_banks(start), start, end,
                               grid_sell_pct=PARAM_B_SELL, grid_buy_pct=PARAM_B_BUY, **fp)

        result[name] = _run_paired_comparison(periods_by_category, make_a, make_b, n_workers)
    return result


def run_convergence_test(trading_dates, n_workers):
    print("  收敛性验证")
    sample_sizes = [200, 500, 1000, 2000]
    cat = 'medium_long'
    result = {}
    for n in sample_sizes:
        periods = generate_random_periods(n, cat, trading_dates, seed=42 + 3800 + n)
        full_periods = {}
        for c in CATEGORY_ORDER:
            if c == cat:
                full_periods[c] = periods
            else:
                full_periods[c] = generate_random_periods(max(10, n // 5), c, trading_dates, seed=42 + 3800 + n + hash(c))

        def make_a(start, end):
            return _make_config(_get_banks(start), start, end,
                               grid_sell_pct=PARAM_A_SELL, grid_buy_pct=PARAM_A_BUY)

        def make_b(start, end):
            return _make_config(_get_banks(start), start, end,
                               grid_sell_pct=PARAM_B_SELL, grid_buy_pct=PARAM_B_BUY)

        r = _run_paired_comparison(full_periods, make_a, make_b, n_workers)
        result[str(n)] = r.get(cat, {})
    return result


def run_all_comparisons(n_samples=1000, n_workers=None):
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 8)

    print("=" * 60)
    print("卖1.2%买0.5% vs 卖1.75%买0.75% 多条件蒙特卡洛对比")
    print(f"样本量: {n_samples}, 并行进程: {n_workers}")
    print("=" * 60)

    trading_dates = get_all_trading_dates()

    all_results = {}

    t0 = time.time()
    print("\n[1/9] 基础对比")
    all_results['base'] = run_base_comparison(n_samples, trading_dates, n_workers)
    print(f"  完成: {time.time()-t0:.1f}秒")

    t1 = time.time()
    print("\n[2/9] 资金维度")
    all_results['capital'] = run_capital_comparison(n_samples, trading_dates, n_workers)
    print(f"  完成: {time.time()-t1:.1f}秒")

    t2 = time.time()
    print("\n[3/9] 调仓比例")
    all_results['trade_ratio'] = run_trade_ratio_comparison(n_samples, trading_dates, n_workers)
    print(f"  完成: {time.time()-t2:.1f}秒")

    t3 = time.time()
    print("\n[4/9] 底仓比例")
    all_results['min_position'] = run_min_position_comparison(n_samples, trading_dates, n_workers)
    print(f"  完成: {time.time()-t3:.1f}秒")

    t4 = time.time()
    print("\n[5/9] 再平准频率")
    all_results['rebalance'] = run_rebalance_comparison(n_samples, trading_dates, n_workers)
    print(f"  完成: {time.time()-t4:.1f}秒")

    t5 = time.time()
    print("\n[6/9] 市场环境")
    all_results['market_env'] = run_market_env_comparison(n_samples, trading_dates, n_workers)
    print(f"  完成: {time.time()-t5:.1f}秒")

    t6 = time.time()
    print("\n[7/9] 标的数量")
    all_results['bank_count'] = run_bank_count_comparison(n_samples, trading_dates, n_workers)
    print(f"  完成: {time.time()-t6:.1f}秒")

    t7 = time.time()
    print("\n[8/9] 手续费档位")
    all_results['fee'] = run_fee_comparison(n_samples, trading_dates, n_workers)
    print(f"  完成: {time.time()-t7:.1f}秒")

    t8 = time.time()
    print("\n[9/9] 收敛性验证")
    all_results['convergence'] = run_convergence_test(trading_dates, n_workers)
    print(f"  完成: {time.time()-t8:.1f}秒")

    total = time.time() - t0
    print(f"\n全部完成: {total:.1f}秒")

    return all_results
