import math
import multiprocessing
import os
import time
from collections import defaultdict

import numpy as np
from scipy import stats

from .backtest_engine import BacktestConfig, run_backtest, run_buy_hold
from .config import (CATEGORY_LABELS, CATEGORY_RANGES, FIVE_BANK_CODES,
                     SIX_BANK_CODES)
from .data_loader import get_all_trading_dates
from .monte_carlo_runner import generate_random_periods, preload_all_data

CATEGORY_ORDER = ['short', 'medium', 'medium_long', 'long']
STRATEGY_LABELS = {
    'A': '卖1.2%买0.5%+月度再平准',
    'B': '卖1.0%买0.5%+月度再平准',
    'C': '卖3.0%买2.0%+月度再平准',
    'D': '卖1.2%买0.5%+永不再平准',
    'E': '纯再平准(无网格)',
    'F': '买入持有',
}

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
        capital=30000, grid_sell_pct=0.03, grid_buy_pct=0.02,
        trade_ratio=0.30, min_position_pct=0.50, rebalance_period='month',
        dividend_tax=True, commission_rate=0.0000854, commission_min=0.5,
        exempt_five=True, stamp_duty_rate=0.0005, slippage=0,
        limit_check=True, enable_rebalance=True, enable_grid=True,
        grid_max_loops=999, dividend_reinvest=False,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _run_single_backtest_task(config_dict):
    is_bh = config_dict.pop('_is_buyhold', False)
    config = BacktestConfig(**config_dict)
    try:
        if is_bh:
            r = run_buy_hold(config, preloaded_merged=_worker_preloaded['merged'],
                             preloaded_dates=_worker_preloaded['dates'])
        else:
            r = run_backtest(config, preloaded_merged=_worker_preloaded['merged'],
                             preloaded_dates=_worker_preloaded['dates'])
        if r and r.trading_days > 0 and math.isfinite(r.annual_return) and math.isfinite(r.max_drawdown):
            return {'annual_return': r.annual_return, 'max_drawdown': r.max_drawdown,
                    'trading_days': r.trading_days, 'total_trades': r.total_trades}
    except Exception:
        pass
    return None


def _build_strategy_configs(banks, start, end):
    return {
        'A': _make_config(banks, start, end, grid_sell_pct=0.012, grid_buy_pct=0.005),
        'B': _make_config(banks, start, end, grid_sell_pct=0.010, grid_buy_pct=0.005),
        'C': _make_config(banks, start, end, grid_sell_pct=0.030, grid_buy_pct=0.020),
        'D': _make_config(banks, start, end, grid_sell_pct=0.012, grid_buy_pct=0.005,
                          enable_rebalance=False),
        'E': _make_config(banks, start, end, enable_grid=False),
        'F': _make_config(banks, start, end, enable_grid=False, enable_rebalance=False),
    }


def run_ablation_study(n_samples=1000, n_workers=None):
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 8)

    print("=" * 60)
    print("测试一：策略模块剥离与极密步长对比")
    print(f"样本量: {n_samples}, 并行进程: {n_workers}")
    print("=" * 60)

    trading_dates = get_all_trading_dates()
    periods_by_category = {}
    for cat, (min_d, max_d) in CATEGORY_RANGES.items():
        periods_by_category[cat] = generate_random_periods(
            n_samples, cat, trading_dates, seed=42 + 2100)

    all_tasks = []
    task_meta = []
    for cat in CATEGORY_ORDER:
        for idx, (start, end) in enumerate(periods_by_category[cat]):
            banks = _get_banks(start)
            configs = _build_strategy_configs(banks, start, end)
            for strategy_key, config in configs.items():
                if strategy_key == 'F':
                    task_dict = {
                        'banks': banks, 'start_date': start, 'end_date': end,
                        'capital': 30000, 'grid_sell_pct': 0.03, 'grid_buy_pct': 0.02,
                        'trade_ratio': 0.30, 'min_position_pct': 0.50,
                        'rebalance_period': 'month', 'dividend_tax': True,
                        'commission_rate': 0.0000854, 'commission_min': 0.5,
                        'exempt_five': True, 'stamp_duty_rate': 0.0005,
                        'slippage': 0, 'limit_check': True,
                        'enable_rebalance': False, 'enable_grid': False,
                        'grid_max_loops': 999, 'dividend_reinvest': False,
                        '_is_buyhold': True,
                    }
                    all_tasks.append(task_dict)
                else:
                    d = {k: getattr(config, k) for k in config.__dataclass_fields__}
                    d['_is_buyhold'] = False
                    all_tasks.append(d)
                task_meta.append((cat, idx, strategy_key))

    print(f"总任务数: {len(all_tasks)}")

    t0 = time.time()
    with multiprocessing.Pool(processes=n_workers, initializer=_worker_init) as pool:
        raw_results = list(pool.imap_unordered(_run_single_backtest_task, all_tasks, chunksize=50))

    elapsed = time.time() - t0
    print(f"回测完成: {elapsed:.1f}秒")

    organized = defaultdict(lambda: defaultdict(list))
    bh_data = defaultdict(lambda: defaultdict(list))
    n_invalid = 0
    for i, result in enumerate(raw_results):
        cat, idx, strategy_key = task_meta[i]
        if result is None:
            n_invalid += 1
            continue
        if strategy_key == 'F':
            bh_data[cat][idx] = result
        else:
            organized[cat][strategy_key].append((idx, result))

    print(f"无效结果: {n_invalid}")

    final_results = {}
    for cat in CATEGORY_ORDER:
        cat_result = {}
        for strategy_key in ['A', 'B', 'C', 'D', 'E']:
            items = organized[cat].get(strategy_key, [])
            if not items:
                cat_result[strategy_key] = {'n': 0}
                continue
            annual_returns = []
            max_dds = []
            excess_returns = []
            for idx, r in items:
                annual_returns.append(r['annual_return'])
                max_dds.append(r['max_drawdown'])
                if idx in bh_data[cat]:
                    excess_returns.append(r['annual_return'] - bh_data[cat][idx]['annual_return'])

            sorted_ar = sorted(annual_returns)
            mid = len(sorted_ar) // 2
            median_ar = sorted_ar[mid] if len(sorted_ar) % 2 == 1 else (sorted_ar[mid - 1] + sorted_ar[mid]) / 2
            sorted_dd = sorted(max_dds)
            mid = len(sorted_dd) // 2
            median_dd = sorted_dd[mid] if len(sorted_dd) % 2 == 1 else (sorted_dd[mid - 1] + sorted_dd[mid]) / 2

            win_rate = sum(1 for e in excess_returns if e > 0) / len(excess_returns) * 100 if excess_returns else 0
            sorted_ex = sorted(excess_returns)
            mid = len(sorted_ex) // 2
            median_excess = sorted_ex[mid] if len(sorted_ex) % 2 == 1 else (sorted_ex[mid - 1] + sorted_ex[mid]) / 2

            p_value = 1.0
            if len(excess_returns) >= 10:
                try:
                    _, p_value = stats.wilcoxon(excess_returns, alternative='greater')
                except Exception:
                    p_value = 1.0

            q1 = sorted_ar[len(sorted_ar) // 4] if len(sorted_ar) >= 4 else sorted_ar[0]
            q3 = sorted_ar[3 * len(sorted_ar) // 4] if len(sorted_ar) >= 4 else sorted_ar[-1]
            min_ar = sorted_ar[0]
            max_ar = sorted_ar[-1]

            cat_result[strategy_key] = {
                'n': len(annual_returns),
                'median_annual_return': median_ar,
                'median_max_drawdown': median_dd,
                'win_rate_vs_bh': win_rate,
                'median_excess': median_excess,
                'p_value': p_value,
                'annual_returns': annual_returns,
                'max_drawdowns': max_dds,
                'excess_returns': excess_returns,
                'boxplot': [min_ar, q1, median_ar, q3, max_ar],
            }

        bh_items = list(bh_data[cat].values())
        if bh_items:
            bh_annual = [r['annual_return'] for r in bh_items]
            bh_dds = [r['max_drawdown'] for r in bh_items]
            sorted_bh = sorted(bh_annual)
            mid = len(sorted_bh) // 2
            median_bh = sorted_bh[mid] if len(sorted_bh) % 2 == 1 else (sorted_bh[mid - 1] + sorted_bh[mid]) / 2
            sorted_bhdd = sorted(bh_dds)
            mid = len(sorted_bhdd) // 2
            median_bhdd = sorted_bhdd[mid] if len(sorted_bhdd) % 2 == 1 else (sorted_bhdd[mid - 1] + sorted_bhdd[mid]) / 2
            q1 = sorted_bh[len(sorted_bh) // 4] if len(sorted_bh) >= 4 else sorted_bh[0]
            q3 = sorted_bh[3 * len(sorted_bh) // 4] if len(sorted_bh) >= 4 else sorted_bh[-1]
            cat_result['F'] = {
                'n': len(bh_annual),
                'median_annual_return': median_bh,
                'median_max_drawdown': median_bhdd,
                'annual_returns': bh_annual,
                'boxplot': [sorted_bh[0], q1, median_bh, q3, sorted_bh[-1]],
            }

        final_results[cat] = cat_result

    return final_results
