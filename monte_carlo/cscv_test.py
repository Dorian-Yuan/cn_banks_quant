import math
import os
import time
from itertools import combinations

import numpy as np

from .backtest_engine import BacktestConfig, run_backtest
from .config import FIVE_BANK_CODES, SIX_BANK_CODES
from .monte_carlo_runner import preload_all_data

BUY_STEPS = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]
SELL_STEPS = [0.008, 0.010, 0.012, 0.014, 0.016, 0.018, 0.020]


def build_return_matrix(n_workers=None):
    print("构建收益率矩阵: %d组参数" % (len(BUY_STEPS) * len(SELL_STEPS)))

    banks_6 = SIX_BANK_CODES
    start = "2015-01-05"
    end = "2026-04-21"

    preloaded = preload_all_data()

    configs = []
    param_labels = []
    base_idx = -1

    for sell_pct in SELL_STEPS:
        for buy_pct in BUY_STEPS:
            label = "卖%.1f%%买%.1f%%" % (sell_pct * 100, buy_pct * 100)
            param_labels.append(label)
            if abs(sell_pct - 0.012) < 0.0001 and abs(buy_pct - 0.005) < 0.0001:
                base_idx = len(param_labels) - 1

            config = BacktestConfig(
                banks=banks_6, start_date=start, end_date=end,
                capital=30000, grid_sell_pct=sell_pct, grid_buy_pct=buy_pct,
                trade_ratio=0.30, min_position_pct=0.50, rebalance_period='month',
                dividend_tax=True, commission_rate=0.0000854, commission_min=0.5,
                exempt_five=True, stamp_duty_rate=0.0005, slippage=0,
                limit_check=True, enable_rebalance=True, enable_grid=True,
                grid_max_loops=999, dividend_reinvest=False,
            )
            configs.append(config)

    M = len(configs)
    print("  总参数组合: %d, 基准参数索引: %d(%s)" % (M, base_idx, param_labels[base_idx] if base_idx >= 0 else 'N/A'))

    all_results = []
    for i, config in enumerate(configs):
        if i % 10 == 0:
            print("  回测进度: %d/%d" % (i, M))
        try:
            r = run_backtest(config, preloaded_merged=preloaded['merged'],
                             preloaded_dates=preloaded['dates'])
            if r and r.trading_days > 0 and len(r.equity_curve) > 10:
                all_results.append(r)
            else:
                all_results.append(None)
        except Exception as e:
            print("  回测#%d异常: %s" % (i, e))
            all_results.append(None)

    valid_count = sum(1 for r in all_results if r is not None)
    print("  有效回测: %d/%d" % (valid_count, M))

    all_dates = None
    for r in all_results:
        if r is not None and r.equity_dates:
            cur_dates = set(r.equity_dates)
            if all_dates is None:
                all_dates = cur_dates
            else:
                all_dates = all_dates & cur_dates

    if all_dates is None:
        raise ValueError("无法获取交易日期交集")

    sorted_dates = sorted(all_dates)
    date_to_idx = {d: i for i, d in enumerate(sorted_dates)}
    T = len(sorted_dates)
    print("  交易日交集: %d天" % T)

    R = np.full((T, M), np.nan)
    for j, r in enumerate(all_results):
        if r is None:
            continue
        ec = r.equity_curve
        ed = r.equity_dates
        daily_returns = [0.0]
        for k in range(1, len(ec)):
            if ec[k - 1] > 0:
                daily_returns.append((ec[k] - ec[k - 1]) / ec[k - 1])
            else:
                daily_returns.append(0.0)
        for k in range(len(ed)):
            if ed[k] in date_to_idx:
                idx = date_to_idx[ed[k]]
                if k < len(daily_returns):
                    R[idx, j] = daily_returns[k]

    for j in range(M):
        col = R[:, j]
        valid_mask = ~np.isnan(col)
        if valid_mask.sum() > 1:
            first_valid = np.where(valid_mask)[0][0]
            for i in range(first_valid, T):
                if np.isnan(R[i, j]):
                    R[i, j] = 0

    print("  收益率矩阵: shape=%s" % str(R.shape))

    return R, param_labels, base_idx, sorted_dates


def compute_cscv_with_base(R, S, base_idx, param_labels):
    T, M = R.shape
    block_size = T // S
    remainder = T % S

    block_indices = []
    start = 0
    for i in range(S):
        size = block_size + (1 if i < remainder else 0)
        block_indices.append(list(range(start, start + size)))
        start += size

    half = S // 2
    all_combos = list(combinations(range(S), half))
    n_combos = len(all_combos)
    print("  CSCV: %d区块, C(%d,%d)=%d种组合" % (S, S, half, n_combos))

    block_means = np.zeros((S, M))
    block_vars = np.zeros((S, M))
    for s in range(S):
        idx = block_indices[s]
        block_data = R[idx, :]
        block_means[s, :] = np.mean(block_data, axis=0)
        block_vars[s, :] = np.var(block_data, axis=0, ddof=1)

    is_block_arrays = []
    oos_block_arrays = []
    for is_blocks in all_combos:
        oos_blocks = tuple(b for b in range(S) if b not in is_blocks)
        is_block_arrays.append(np.array(list(is_blocks)))
        oos_block_arrays.append(np.array(list(oos_blocks)))

    is_sharpes = np.zeros((n_combos, M))
    oos_sharpes = np.zeros((n_combos, M))

    for c_idx in range(n_combos):
        is_mean = np.sum(block_means[is_block_arrays[c_idx], :], axis=0)
        is_var = np.sum(block_vars[is_block_arrays[c_idx], :], axis=0)
        is_sharpes[c_idx, :] = np.where(is_var > 0, is_mean / np.sqrt(is_var), 0)

        oos_mean = np.sum(block_means[oos_block_arrays[c_idx], :], axis=0)
        oos_var = np.sum(block_vars[oos_block_arrays[c_idx], :], axis=0)
        oos_sharpes[c_idx, :] = np.where(oos_var > 0, oos_mean / np.sqrt(oos_var), 0)

    is_best_indices = np.argmax(is_sharpes, axis=1)
    is_best_sharpes = is_sharpes[np.arange(n_combos), is_best_indices]
    oos_sharpes_of_best = oos_sharpes[np.arange(n_combos), is_best_indices]

    oos_rank_matrix = np.zeros((n_combos, M), dtype=int)
    for c_idx in range(n_combos):
        oos_rank_matrix[c_idx, :] = np.argsort(np.argsort(-oos_sharpes[c_idx, :])) + 1

    oos_ranks = oos_rank_matrix[np.arange(n_combos), is_best_indices]

    relative_ranks = (oos_ranks - 1) / (M - 1) if M > 1 else oos_ranks.astype(float)
    pbo = float(np.mean(relative_ranks > 0.5))

    logit_ranks = np.log(relative_ranks / (1 - relative_ranks + 1e-10) + 1e-10)

    is_rank_matrix = np.zeros((n_combos, M), dtype=int)
    for c_idx in range(n_combos):
        is_rank_matrix[c_idx, :] = np.argsort(np.argsort(-is_sharpes[c_idx, :])) + 1

    base_tracking = None
    if base_idx >= 0:
        is_ranks_of_base = is_rank_matrix[:, base_idx].astype(float)
        oos_ranks_of_base = oos_rank_matrix[:, base_idx].astype(float)
        base_tracking = {
            'is_avg_rank': float(np.mean(is_ranks_of_base)),
            'oos_avg_rank': float(np.mean(oos_ranks_of_base)),
            'is_rank_dist': is_ranks_of_base.tolist(),
            'oos_rank_dist': oos_ranks_of_base.tolist(),
            'is_top1_pct': float(np.mean(is_ranks_of_base <= 1) * 100),
            'oos_top5_pct': float(np.mean(oos_ranks_of_base <= 5) * 100),
            'is_median_rank': float(np.median(is_ranks_of_base)),
            'oos_median_rank': float(np.median(oos_ranks_of_base)),
        }

    best_param_counts = {}
    for idx in is_best_indices:
        label = param_labels[idx] if idx < len(param_labels) else str(idx)
        best_param_counts[label] = best_param_counts.get(label, 0) + 1
    sorted_best = sorted(best_param_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    result = {
        'pbo': pbo,
        'S': S,
        'M': M,
        'n_combos': n_combos,
        'relative_ranks': relative_ranks.tolist(),
        'logit_ranks': logit_ranks.tolist(),
        'is_best_sharpes': is_best_sharpes.tolist(),
        'oos_sharpes_of_best': oos_sharpes_of_best.tolist(),
        'oos_ranks': oos_ranks.tolist(),
        'base_tracking': base_tracking,
        'base_idx': base_idx,
        'base_label': param_labels[base_idx] if base_idx >= 0 and base_idx < len(param_labels) else 'N/A',
        'top_is_params': sorted_best,
        'param_labels': param_labels,
    }

    print("  PBO = %.2f%% (%d/%d)" % (pbo * 100, int(np.sum(relative_ranks > 0.5)), n_combos))
    if base_tracking:
        print("  基准参数(%s): IS中位排名%.1f, OOS中位排名%.1f" % (
            result['base_label'], base_tracking['is_median_rank'], base_tracking['oos_median_rank']))

    return result


def run_cscv(S=16, n_workers=None):
    print("=" * 60)
    print("CSCV 过拟合测试")
    print("=" * 60)

    t0 = time.time()
    R, param_labels, base_idx, dates = build_return_matrix(n_workers=n_workers)
    t1 = time.time()
    print("  矩阵构建: %.1f秒" % (t1 - t0))

    result = compute_cscv_with_base(R, S, base_idx, param_labels)
    t2 = time.time()
    print("  CSCV计算: %.1f秒" % (t2 - t1))
    print("  总耗时: %.1f秒" % (t2 - t0))

    result['elapsed'] = t2 - t0
    result['dates_range'] = "%s ~ %s" % (dates[0], dates[-1]) if dates else ''
    result['trading_days'] = len(dates)

    return result
