import sys
import argparse
import time
from .monte_carlo_runner import run_all_experiments, run_all_supplement_experiments
from .generate_report import generate_html_report


def main():
    parser = argparse.ArgumentParser(description="蒙特卡洛随机回测系统")
    parser.add_argument('-n', '--samples', type=int, default=1000, help='每个期限的随机样本数（默认1000）')
    parser.add_argument('-o', '--output', type=str, default=None, help='HTML报告输出路径')
    parser.add_argument('--quick', action='store_true', help='快速模式（每期限10个样本）')
    parser.add_argument('--supplement', action='store_true', help='仅运行补充实验F~L')
    args = parser.parse_args()

    n_samples = 10 if args.quick else args.samples

    print("=" * 60)
    print("蒙特卡洛随机回测系统")
    print(f"每期限样本数: {n_samples}")
    if args.supplement:
        print("模式: 仅补充实验(F~L)")
    print("=" * 60)

    t0 = time.time()
    if args.supplement:
        all_results = run_all_supplement_experiments(n_samples=n_samples)
    else:
        all_results = run_all_experiments(n_samples=n_samples)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"全部实验完成，总耗时: {elapsed:.1f}秒")

    output_path = generate_html_report(all_results, output_path=args.output)
    print(f"报告已生成: {output_path}")


if __name__ == '__main__':
    main()
