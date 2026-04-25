import sys
import os
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monte_carlo.ablation_study import run_ablation_study
from monte_carlo.grid_search import run_grid_search
from monte_carlo.generate_ablation_report import generate_ablation_report


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    parser = argparse.ArgumentParser(description="策略剥离对比与极密参数寻优")
    parser.add_argument('-n', '--samples', type=int, default=1000, help='测试一每期限样本数')
    parser.add_argument('-n2', '--samples2', type=int, default=500, help='测试二每期限样本数')
    parser.add_argument('-w', '--workers', type=int, default=None, help='并行进程数')
    parser.add_argument('--quick', action='store_true', help='快速模式(每期限10样本)')
    parser.add_argument('--test1-only', action='store_true', help='仅运行测试一')
    parser.add_argument('--test2-only', action='store_true', help='仅运行测试二')
    args = parser.parse_args()

    n1 = 10 if args.quick else args.samples
    n2 = 10 if args.quick else args.samples2
    n_workers = args.workers

    print("=" * 60)
    print("策略剥离对比与极密参数寻优")
    print(f"测试一样本量: {n1}, 测试二样本量: {n2}, 并行进程: {n_workers or '自动'}")
    print("=" * 60)

    t0 = time.time()
    ablation_results = None
    grid_results = None

    if not args.test2_only:
        print("\n" + "=" * 60)
        print("开始测试一：策略模块剥离与极密步长对比")
        print("=" * 60)
        t1 = time.time()
        ablation_results = run_ablation_study(n_samples=n1, n_workers=n_workers)
        print(f"测试一完成: {time.time()-t1:.1f}秒")

    if not args.test1_only:
        print("\n" + "=" * 60)
        print("开始测试二：极密参数二维网格寻优")
        print("=" * 60)
        t2 = time.time()
        grid_results = run_grid_search(n_samples=n2, n_workers=n_workers)
        print(f"测试二完成: {time.time()-t2:.1f}秒")

    total_elapsed = time.time() - t0
    print(f"\n总耗时: {total_elapsed:.1f}秒")

    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "research", "ablation_grid_results.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    save_data = {}
    if ablation_results:
        save_data['ablation'] = ablation_results
    if grid_results:
        save_data['grid_search'] = grid_results
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(save_data), f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {json_path}")

    report_path = generate_ablation_report(
        ablation_results or {}, grid_results or {},
        n_samples1=n1, n_samples2=n2,
        output_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "research", "ablation_grid_search_report.html"))
    print(f"报告已生成: {report_path}")

    try:
        import urllib.request
        url = f"https://api.day.app/XbLpUuV9b4wh2ZExePySME/ablation_done?group=quant&title=AblationDone&body=completed+{int(total_elapsed)}s"
        urllib.request.urlopen(url)
        print("BARK通知已发送")
    except Exception as e:
        print(f"BARK通知失败: {e}")


if __name__ == '__main__':
    main()
