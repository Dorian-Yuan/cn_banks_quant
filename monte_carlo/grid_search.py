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


def _build_param_combos():
    sell_steps = [0.010, 0.0125, 0.015, 0.0175, 0.020]
    buy_steps = [0.005, 0.0075, 0.010, 0.0125, 0.015]
    combos = []
    for s in sell_steps:
        for b in buy_steps:
            combos.append((s, b))
    combos.append((0.012, 0.005))
    combos.append((0.012, 0.004))
    return combos


def run_grid_search(n_samples=500, n_workers=None):
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 8)

    print("=" * 60)
    print("测试二：极密参数二维网格寻优")
    print(f"样本量: {n_samples}, 并行进程: {n_workers}")
    print("=" * 60)

    combos = _build_param_combos()
    print(f"参数组合数: {len(combos)}")

    trading_dates = get_all_trading_dates()
    periods_by_category = {}
    for cat, (min_d, max_d) in CATEGORY_RANGES.items():
        periods_by_category[cat] = generate_random_periods(
            n_samples, cat, trading_dates, seed=42 + 2200)

    all_tasks = []
    task_meta = []
    for cat in CATEGORY_ORDER:
        for idx, (start, end) in enumerate(periods_by_category[cat]):
            banks = _get_banks(start)
            for combo_idx, (sell_pct, buy_pct) in enumerate(combos):
                config = _make_config(banks, start, end, grid_sell_pct=sell_pct, grid_buy_pct=buy_pct)
                config_dict = {k: getattr(config, k) for k in config.__dataclass_fields__}
                config_dict['_is_buyhold'] = False
                all_tasks.append(config_dict)
                task_meta.append((cat, idx, combo_idx))

            bh_config = _make_config(banks, start, end, enable_grid=False, enable_rebalance=False)
            bh_dict = {k: getattr(bh_config, k) for k in bh_config.__dataclass_fields__}
            bh_dict['_is_buyhold'] = True
            all_tasks.append(bh_dict)
            task_meta.append((cat, idx, -1))

    print(f"总任务数: {len(all_tasks)}")

    t0 = time.time()
    with multiprocessing.Pool(processes=n_workers, initializer=_worker_init) as pool:
        raw_results = list(pool.imap_unordered(_run_single_backtest_task, all_tasks, chunksize=50))

    elapsed = time.time() - t0
    print(f"回测完成: {elapsed:.1f}秒")

    organized = defaultdict(lambda: {'bh': {}, 'combos': defaultdict(list)})
    n_invalid = 0
    for i, result in enumerate(raw_results):
        cat, idx, combo_idx = task_meta[i]
        if result is None:
            n_invalid += 1
            continue
        if combo_idx == -1:
            organized[cat]['bh'][idx] = result
        else:
            organized[cat]['combos'][combo_idx].append((idx, result))

    print(f"无效结果: {n_invalid}")

    final_results = {}
    for cat in CATEGORY_ORDER:
        cat_result = {}
        bh_data = organized[cat]['bh']

        for combo_idx, (sell_pct, buy_pct) in enumerate(combos):
            items = organized[cat]['combos'].get(combo_idx, [])
            if not items:
                cat_result[f"卖{sell_pct*100:.2f}%买{buy_pct*100:.2f}%"] = {'n': 0}
                continue

            annual_returns = []
            excess_returns = []
            for idx, r in items:
                annual_returns.append(r['annual_return'])
                if idx in bh_data:
                    excess_returns.append(r['annual_return'] - bh_data[idx]['annual_return'])

            sorted_ar = sorted(annual_returns)
            mid = len(sorted_ar) // 2
            median_ar = sorted_ar[mid] if len(sorted_ar) % 2 == 1 else (sorted_ar[mid - 1] + sorted_ar[mid]) / 2

            win_rate = sum(1 for e in excess_returns if e > 0) / len(excess_returns) * 100 if excess_returns else 0

            key = f"卖{sell_pct*100:.2f}%买{buy_pct*100:.2f}%"
            cat_result[key] = {
                'n': len(annual_returns),
                'median_annual_return': median_ar,
                'win_rate_vs_bh': win_rate,
                'sell_pct': sell_pct,
                'buy_pct': buy_pct,
                'annual_returns': annual_returns,
                'excess_returns': excess_returns,
            }

        final_results[cat] = cat_result

    long_results = final_results.get('long', {})
    sorted_combos = sorted(long_results.items(), key=lambda x: x[1].get('median_annual_return', 0), reverse=True)
    top3 = sorted_combos[:3]

    top3_significance = {}
    if len(top3) >= 2:
        for i in range(len(top3)):
            for j in range(i + 1, len(top3)):
                name_i, data_i = top3[i]
                name_j, data_j = top3[j]
                excess_i = data_i.get('excess_returns', [])
                excess_j = data_j.get('excess_returns', [])
                min_len = min(len(excess_i), len(excess_j))
                if min_len >= 10:
                    paired = [excess_i[k] - excess_j[k] for k in range(min_len)]
                    try:
                        _, p_value = stats.wilcoxon(paired)
                    except Exception:
                        p_value = 1.0
                    top3_significance[f"{name_i} vs {name_j}"] = {
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                    }

    final_results['_top3'] = [(name, data) for name, data in top3]
    final_results['_top3_significance'] = top3_significance
    final_results['_combos'] = [(s, b) for s, b in combos]

    return final_results
