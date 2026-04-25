import sys
import os
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monte_carlo.cscv_test import run_cscv
from monte_carlo.generate_cscv_report import generate_cscv_report


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
    parser = argparse.ArgumentParser(description="CSCV过拟合测试")
    parser.add_argument('-s', '--blocks', type=int, default=16, help='CSCV区块数(默认16)')
    parser.add_argument('-w', '--workers', type=int, default=None, help='并行进程数')
    parser.add_argument('--quick', action='store_true', help='快速模式(S=8)')
    args = parser.parse_args()

    S = 8 if args.quick else args.blocks
    n_workers = args.workers

    t0 = time.time()
    result = run_cscv(S=S, n_workers=n_workers)
    total_elapsed = time.time() - t0
    print("\n总耗时: %.1f秒" % total_elapsed)

    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "research", "cscv_results.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    save_data = make_serializable(result)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print("结果已保存: %s" % json_path)

    report_path = generate_cscv_report(
        result,
        output_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "research", "cscv_report.html"))
    print("报告已生成: %s" % report_path)

    try:
        import urllib.request
        pbo = result.get('pbo', 0)
        url = "https://api.day.app/XbLpUuV9b4wh2ZExePySME/cscv_done?group=quant&title=CSCV_Test&body=PBO=%.1f%%+completed+%ds" % (pbo * 100, int(total_elapsed))
        urllib.request.urlopen(url)
        print("BARK通知已发送")
    except Exception as e:
        print("BARK通知失败: %s" % e)


if __name__ == '__main__':
    main()
