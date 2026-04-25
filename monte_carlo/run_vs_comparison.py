import sys
import os
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monte_carlo.vs_comparison import run_all_comparisons
from monte_carlo.generate_vs_report import generate_vs_report


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
    parser = argparse.ArgumentParser(description="卖1.2%买0.5% vs 卖1.75%买0.75% 多条件对比")
    parser.add_argument('-n', '--samples', type=int, default=1000, help='每条件每期限样本数')
    parser.add_argument('-w', '--workers', type=int, default=None, help='并行进程数')
    parser.add_argument('--quick', action='store_true', help='快速模式(每条件10样本)')
    args = parser.parse_args()

    n_samples = 10 if args.quick else args.samples
    n_workers = args.workers

    t0 = time.time()
    all_results = run_all_comparisons(n_samples=n_samples, n_workers=n_workers)
    total_elapsed = time.time() - t0
    print(f"\n总耗时: {total_elapsed:.1f}秒")

    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "research", "vs_comparison_results.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    save_data = make_serializable(all_results)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {json_path}")

    report_path = generate_vs_report(
        all_results, n_samples=n_samples,
        output_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "research", "vs_comparison_report.html"))
    print(f"报告已生成: {report_path}")

    try:
        import urllib.request
        url = f"https://api.day.app/XbLpUuV9b4wh2ZExePySME/vs_comparison_done?group=quant&title=VSComparison&body=completed+{int(total_elapsed)}s"
        urllib.request.urlopen(url)
        print("BARK通知已发送")
    except Exception as e:
        print(f"BARK通知失败: {e}")


if __name__ == '__main__':
    main()
