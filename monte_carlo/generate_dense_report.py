import os
import json
from .config import CATEGORY_LABELS

CATEGORY_ORDER = ['short', 'medium', 'medium_long', 'long']


def _fmt(v, d=2):
    if v is None:
        return '-'
    return f'{v:.{d}f}'


def _win_rate_color(wr):
    if wr >= 70:
        return 'rgba(46,204,113,0.4)'
    elif wr >= 55:
        return 'rgba(46,204,113,0.2)'
    elif wr >= 45:
        return 'rgba(241,196,15,0.3)'
    elif wr >= 30:
        return 'rgba(231,76,60,0.2)'
    else:
        return 'rgba(231,76,60,0.4)'


def _render_M(result):
    rows = []
    for sub_key, sub_label in [('vs_2_1', 'vs 卖2%买1%'), ('vs_3_2', 'vs 卖3%买2%'), ('vs_5_3', 'vs 卖5%买3%')]:
        sub = result.get(sub_key, {})
        rows.append(f'<tr><td rowspan="4">{sub_label}</td>')
        for i, cat in enumerate(CATEGORY_ORDER):
            s = sub.get(cat, {}).get('stats', {})
            wr = s.get('win_rate', 0)
            me = s.get('median_excess', 0)
            ae = s.get('avg_excess', 0)
            pv = s.get('p_value', 1)
            sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
            color = _win_rate_color(wr)
            prefix = '</tr><tr>' if i > 0 else ''
            rows.append(f'{prefix}<td>{CATEGORY_LABELS[cat]}</td><td style="background:{color}">{wr:.1f}%</td><td>{me:.2f}%</td><td>{ae:.2f}%</td><td>{pv:.4f}</td><td>{sig}</td>')
        rows.append('</tr>')
    return f'''<table class="detail-table">
        <thead><tr><th>对比</th><th>期限</th><th>极密胜率</th><th>中位超额</th><th>平均超额</th><th>p值</th><th>显著性</th></tr></thead>
        <tbody>{"".join(rows)}</tbody></table>'''


def _render_N(result):
    rows = []
    for name in result:
        for cat in CATEGORY_ORDER:
            r = result[name].get(cat, {})
            rows.append(f'<tr><td>{name}</td><td>{CATEGORY_LABELS[cat]}</td><td>{r.get("n",0)}</td><td>{_fmt(r.get("median_annual_return",0))}%</td><td>{_fmt(r.get("median_max_drawdown",0))}%</td><td>{r.get("avg_trades",0):.0f}</td></tr>')
    return f'''<table class="detail-table">
        <thead><tr><th>仓位</th><th>期限</th><th>样本</th><th>中位年化</th><th>中位回撤</th><th>平均交易次数</th></tr></thead>
        <tbody>{"".join(rows)}</tbody></table>'''


def _render_O(result):
    rows = []
    for name in result:
        for cat in CATEGORY_ORDER:
            r = result[name].get(cat, {})
            rows.append(f'<tr><td>{name}</td><td>{CATEGORY_LABELS[cat]}</td><td>{r.get("n",0)}</td><td>{_fmt(r.get("median_annual_return",0))}%</td><td>{_fmt(r.get("median_max_drawdown",0))}%</td></tr>')
    return f'''<table class="detail-table">
        <thead><tr><th>底仓</th><th>期限</th><th>样本</th><th>中位年化</th><th>中位回撤</th></tr></thead>
        <tbody>{"".join(rows)}</tbody></table>'''


def _render_P(result):
    rows = []
    for cat in CATEGORY_ORDER:
        r = result.get(cat, {})
        row = f'<tr><td>{CATEGORY_LABELS[cat]}</td><td>{r.get("n",0)}</td>'
        for loops in [1, 5, 10, 999]:
            wr = r.get(f'loops_{loops}_win_rate', 0)
            mar = r.get(f'loops_{loops}_median_annual', 0)
            label = '无限制' if loops == 999 else f'{loops}次'
            row += f'<td style="background:{_win_rate_color(wr)}">{label}: {wr:.1f}% (年化{_fmt(mar)}%)</td>'
        row += '</tr>'
        rows.append(row)
    return f'''<table class="detail-table">
        <thead><tr><th>期限</th><th>样本</th><th>1次/日</th><th>5次/日</th><th>10次/日</th><th>无限制</th></tr></thead>
        <tbody>{"".join(rows)}</tbody></table>
        <p class="note">grid_max_loops分别应用于买入和卖出循环。1次/日=每天每银行最多买1次+卖1次。</p>'''


def _render_Q(result):
    rows = []
    for name in result:
        for cat in CATEGORY_ORDER:
            r = result[name].get(cat, {})
            rows.append(f'<tr><td>{name}</td><td>{CATEGORY_LABELS[cat]}</td><td>{r.get("n",0)}</td><td>{_fmt(r.get("median_annual_return",0))}%</td><td>{_fmt(r.get("avg_fee_ratio",0))}%</td></tr>')
    return f'''<table class="detail-table">
        <thead><tr><th>费率</th><th>期限</th><th>样本</th><th>中位年化</th><th>手续费/收益</th></tr></thead>
        <tbody>{"".join(rows)}</tbody></table>'''


def _render_R(result):
    rows = []
    for name in result:
        for cat in CATEGORY_ORDER:
            r = result[name].get(cat, {})
            rows.append(f'<tr><td>{name}</td><td>{CATEGORY_LABELS[cat]}</td><td>{r.get("n",0)}</td><td>{_fmt(r.get("median_annual_return",0))}%</td><td>{_fmt(r.get("avg_fee_ratio",0))}%</td></tr>')
    return f'''<table class="detail-table">
        <thead><tr><th>资金</th><th>期限</th><th>样本</th><th>中位年化</th><th>手续费/收益</th></tr></thead>
        <tbody>{"".join(rows)}</tbody></table>'''


def _render_S(result):
    rows = []
    for name in result:
        for cat in CATEGORY_ORDER:
            r = result[name].get(cat, {})
            rows.append(f'<tr><td>{name}</td><td>{CATEGORY_LABELS[cat]}</td><td>{r.get("n",0)}</td><td>{_fmt(r.get("median_annual_return",0))}%</td></tr>')
    return f'''<table class="detail-table">
        <thead><tr><th>滑点</th><th>期限</th><th>样本</th><th>中位年化</th></tr></thead>
        <tbody>{"".join(rows)}</tbody></table>'''


def _render_T(result):
    rows = []
    freq_labels = {'month': '月度', 'quarter': '季度', 'year': '年度', 'none': '永不再平准'}
    for cat in CATEGORY_ORDER:
        cat_data = result.get(cat, {})
        row = f'<tr><td>{CATEGORY_LABELS[cat]}</td>'
        for freq in ['month', 'quarter', 'year', 'none']:
            r = cat_data.get(freq, {})
            row += f'<td>{_fmt(r.get("median_annual_return",0))}%</td>'
        row += '</tr>'
        rows.append(row)
    return f'''<table class="detail-table">
        <thead><tr><th>期限</th><th>月度</th><th>季度</th><th>年度</th><th>永不再平准</th></tr></thead>
        <tbody>{"".join(rows)}</tbody></table>'''


def _render_U(result):
    rows = []
    for name in result:
        r = result[name]
        if 'error' in r:
            rows.append(f'<tr><td>{name}</td><td colspan="9">{r["error"]}</td></tr>')
        else:
            ex = r.get('excess', 0)
            color = '#2ecc71' if ex > 0 else '#e74c3c'
            rows.append(f'<tr><td>{name}</td><td>{r.get("trading_days",0)}</td>'
                        f'<td>{_fmt(r.get("dense_annual",0))}%</td><td>{_fmt(r.get("default_annual",0))}%</td>'
                        f'<td style="color:{color};font-weight:bold">{_fmt(ex)}%</td>'
                        f'<td>{_fmt(r.get("dense_dd",0))}%</td><td>{_fmt(r.get("default_dd",0))}%</td>'
                        f'<td>{r.get("dense_trades",0)}</td><td>{r.get("default_trades",0)}</td>'
                        f'<td>{_fmt(r.get("dense_fees",0),0)}元</td><td>{_fmt(r.get("default_fees",0),0)}元</td></tr>')
    return f'''<table class="detail-table">
        <thead><tr><th>时段</th><th>交易日</th><th>极密年化</th><th>默认年化</th><th>超额</th><th>极密回撤</th><th>默认回撤</th><th>极密交易</th><th>默认交易</th><th>极密费用</th><th>默认费用</th></tr></thead>
        <tbody>{"".join(rows)}</tbody></table>'''


def _render_V(result):
    rows = []
    for cat in CATEGORY_ORDER:
        s = result.get(cat, {}).get('stats', {})
        wr = s.get('win_rate', 0)
        me = s.get('median_excess', 0)
        ae = s.get('avg_excess', 0)
        pv = s.get('p_value', 1)
        sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
        color = _win_rate_color(wr)
        rows.append(f'<tr><td>{CATEGORY_LABELS[cat]}</td><td style="background:{color}">{wr:.1f}%</td><td>{me:.2f}%</td><td>{ae:.2f}%</td><td>{pv:.4f}</td><td>{sig}</td></tr>')
    return f'''<table class="detail-table">
        <thead><tr><th>期限</th><th>极密胜率</th><th>中位超额</th><th>平均超额</th><th>p值</th><th>显著性</th></tr></thead>
        <tbody>{"".join(rows)}</tbody></table>'''


def _generate_experiment_section(name, result):
    if result is None:
        return f'<div class="card"><h2>{name}</h2><p>实验运行失败</p></div>'

    html = f'<div class="card"><h2>{name}</h2>'

    if "测试M" in name:
        html += _render_M(result)
    elif "测试N" in name:
        html += _render_N(result)
    elif "测试O" in name:
        html += _render_O(result)
    elif "测试P" in name:
        html += _render_P(result)
    elif "测试Q" in name:
        html += _render_Q(result)
    elif "测试R" in name:
        html += _render_R(result)
    elif "测试S" in name:
        html += _render_S(result)
    elif "测试T" in name:
        html += _render_T(result)
    elif "测试U" in name:
        html += _render_U(result)
    elif "测试V" in name:
        html += _render_V(result)
    else:
        html += f'<p>未识别的实验类型</p>'

    html += '</div>'
    return html


def generate_dense_grid_report(all_results, output_path=None):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research", "dense_grid_report.html")

    sections = []
    for name, result in all_results.items():
        sections.append(_generate_experiment_section(name, result))

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>极密网格(卖1%买0.5%)专项测试报告</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
.header {{ text-align: center; padding: 30px 0; border-bottom: 2px solid #3498db; margin-bottom: 30px; }}
.header h1 {{ color: #f1c40f; font-size: 2em; margin-bottom: 10px; }}
.header .meta {{ display: flex; justify-content: center; gap: 20px; color: #8899aa; font-size: 0.9em; }}
.card {{ background: #16213e; border-radius: 8px; padding: 24px; margin-bottom: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
.card h2 {{ color: #3498db; margin-bottom: 16px; border-bottom: 1px solid #2c3e50; padding-bottom: 8px; }}
.card h3 {{ color: #f1c40f; margin: 16px 0 8px; }}
.detail-table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.9em; }}
.detail-table th {{ background: #0f3460; color: #f1c40f; padding: 10px 12px; text-align: center; white-space: nowrap; }}
.detail-table td {{ padding: 8px 12px; text-align: center; border-bottom: 1px solid #2c3e50; }}
.detail-table tr:hover {{ background: rgba(52,152,219,0.1); }}
.note {{ color: #8899aa; font-style: italic; margin-top: 8px; }}
.conclusion {{ border: 2px solid #f1c40f; }}
.conclusion ul {{ padding-left: 24px; line-height: 1.8; }}
.conclusion li {{ margin-bottom: 4px; }}
@media (max-width: 768px) {{
    .header h1 {{ font-size: 1.5em; }}
    th, td {{ padding: 6px 8px; font-size: 0.8em; }}
}}
</style>
</head>
<body>
<div class="header">
    <h1>极密网格(卖1%买0.5%)专项测试报告</h1>
    <div class="meta">
        <span>蒙特卡洛随机回测</span>
        <span>每期限1000样本</span>
        <span>数据范围: 2015-01至今</span>
        <span>手续费: 万0.854(0.5起)+印花税万5+免5</span>
    </div>
</div>
{"".join(sections)}
</body>
</html>'''

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return output_path
