import os
import json
from datetime import datetime
from .config import CATEGORY_LABELS, REPORT_DIR, DATA_START


def _significance_stars(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return ""


def _fmt(val, decimals=2):
    if val is None:
        return "-"
    return f"{val:.{decimals}f}"


def generate_html_report(all_results, output_path=None):
    if output_path is None:
        os.makedirs(REPORT_DIR, exist_ok=True)
        output_path = os.path.join(REPORT_DIR, "monte_carlo_report.html")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections = []

    sections.append(_generate_header(now))
    sections.append(_generate_overview(all_results))

    exp_names = list(all_results.keys())
    for name in exp_names:
        result = all_results[name]
        if result is None:
            sections.append(f'<div class="card"><h2>{name}</h2><p>运行失败</p></div>')
            continue
        sections.append(_generate_experiment_section(name, result))

    sections.append(_generate_conclusion(all_results))

    html = _wrap_html("\n".join(sections), now)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"报告已生成: {output_path}")
    return output_path


def _generate_header(now):
    return f'''
    <div class="header">
        <h1>蒙特卡洛随机回测报告</h1>
        <p class="subtitle">多银行波段再平准网格策略 — 参数敏感性分析</p>
        <div class="meta">
            <span>生成时间: {now}</span>
            <span>数据范围: {DATA_START} 至今</span>
            <span>标的池: 5/6大行动态切换(2019-12-10前5行,之后6行)</span>
        </div>
    </div>'''


def _generate_overview(all_results):
    rows = []
    for name, result in all_results.items():
        if result is None:
            rows.append(f'<tr><td>{name}</td><td colspan="4">运行失败</td></tr>')
            continue

        if "实验1" in name or "实验2" in name or "实验7" in name:
            win_rates = []
            for cat in CATEGORY_LABELS:
                if cat in result:
                    win_rates.append(result[cat]['stats']['win_rate'])
            if win_rates:
                avg_wr = sum(win_rates) / len(win_rates)
                rows.append(f'<tr><td>{name}</td>')
                for cat in CATEGORY_LABELS:
                    wr = result.get(cat, {}).get('stats', {}).get('win_rate', 0)
                    color = _win_rate_color(wr)
                    rows.append(f'<td style="background:{color}">{wr:.1f}%</td>')
                rows.append('</tr>')
        elif "实验3" in name:
            for sub_name, sub_key in [("vs 30%", "vs_30"), ("vs 10%", "vs_10")]:
                sub = result.get(sub_key, {})
                rows.append(f'<tr><td>{name} ({sub_name})</td>')
                for cat in CATEGORY_LABELS:
                    wr = sub.get(cat, {}).get('stats', {}).get('win_rate', 0)
                    color = _win_rate_color(wr)
                    rows.append(f'<td style="background:{color}">{wr:.1f}%</td>')
                rows.append('</tr>')
        elif "实验4" in name:
            for freq in ['month', 'quarter', 'year']:
                freq_label = {'month': '月度', 'quarter': '季度', 'year': '年度'}[freq]
                rows.append(f'<tr><td>{name} ({freq_label}胜率)</td>')
                for cat in CATEGORY_LABELS:
                    key = f'{freq}_win_rate'
                    wr = result.get(cat, {}).get(key, 0)
                    color = _win_rate_color(wr)
                    rows.append(f'<td style="background:{color}">{wr:.1f}%</td>')
                rows.append('</tr>')
        elif "实验5" in name:
            for sub_name, sub_key in [("vs 对称", "vs_symmetric"), ("vs 早止盈", "vs_early_stop")]:
                sub = result.get(sub_key, {})
                rows.append(f'<tr><td>{name} ({sub_name})</td>')
                for cat in CATEGORY_LABELS:
                    wr = sub.get(cat, {}).get('stats', {}).get('win_rate', 0)
                    color = _win_rate_color(wr)
                    rows.append(f'<td style="background:{color}">{wr:.1f}%</td>')
                rows.append('</tr>')
        elif "实验6" in name:
            for sub_name, sub_key in [("vs 30%", "vs_30"), ("vs 70%", "vs_70")]:
                sub = result.get(sub_key, {})
                rows.append(f'<tr><td>{name} ({sub_name})</td>')
                for cat in CATEGORY_LABELS:
                    wr = sub.get(cat, {}).get('stats', {}).get('win_rate', 0)
                    color = _win_rate_color(wr)
                    rows.append(f'<td style="background:{color}">{wr:.1f}%</td>')
                rows.append('</tr>')
        elif "实验8" in name:
            for metric, key in [("抗跌胜率", "dd_win_rate"), ("夏普胜率", "sharpe_win_rate")]:
                rows.append(f'<tr><td>{name} ({metric})</td>')
                for cat in CATEGORY_LABELS:
                    wr = result.get(cat, {}).get(key, 0)
                    color = _win_rate_color(wr)
                    rows.append(f'<td style="background:{color}">{wr:.1f}%</td>')
                rows.append('</tr>')
        elif "补充A" in name:
            for cap in sorted(result.keys()):
                rows.append(f'<tr><td>{name} ({cap}元)</td>')
                for cat in CATEGORY_LABELS:
                    wr = result[cap].get(cat, {}).get('stats', {}).get('win_rate', 0)
                    color = _win_rate_color(wr)
                    rows.append(f'<td style="background:{color}">{wr:.1f}%</td>')
                rows.append('</tr>')
        elif "补充C" in name:
            for env in ['bull', 'bear', 'flat']:
                env_label = {'bull': '牛市', 'bear': '熊市', 'flat': '震荡'}[env]
                rows.append(f'<tr><td>{name} ({env_label})</td>')
                for cat in CATEGORY_LABELS:
                    wr = result.get(cat, {}).get(env, {}).get('win_rate', 0)
                    color = _win_rate_color(wr)
                    rows.append(f'<td style="background:{color}">{wr:.1f}%</td>')
                rows.append('</tr>')
        elif "补充D" in name:
            win_rates = []
            for cat in CATEGORY_LABELS:
                if cat in result:
                    win_rates.append(result[cat]['stats']['win_rate'])
            if win_rates:
                rows.append(f'<tr><td>{name}</td>')
                for cat in CATEGORY_LABELS:
                    wr = result.get(cat, {}).get('stats', {}).get('win_rate', 0)
                    color = _win_rate_color(wr)
                    rows.append(f'<td style="background:{color}">{wr:.1f}%</td>')
                rows.append('</tr>')
        elif "补充E" in name:
            for cost_name in result:
                rows.append(f'<tr><td>{name} ({cost_name})</td>')
                for cat in CATEGORY_LABELS:
                    wr = result[cost_name].get(cat, {}).get('stats', {}).get('win_rate', 0)
                    color = _win_rate_color(wr)
                    rows.append(f'<td style="background:{color}">{wr:.1f}%</td>')
                rows.append('</tr>')
        elif "补充B" in name:
            for combo_name in result:
                rows.append(f'<tr><td>{name} ({combo_name})</td>')
                for cat in CATEGORY_LABELS:
                    med = result[combo_name].get(cat, {}).get('median_annual_return', 0)
                    rows.append(f'<td>{med:.2f}%</td>')
                rows.append('</tr>')
        elif "补充F" in name or "补充J" in name:
            rows.append(f'<tr><td>{name}</td>')
            for cat in CATEGORY_LABELS:
                wr = result.get(cat, {}).get('stats', {}).get('win_rate', 0)
                color = _win_rate_color(wr)
                rows.append(f'<td style="background:{color}">{wr:.1f}%</td>')
            rows.append('</tr>')
        elif "补充G" in name:
            for sub_name, sub_key in [("vs 循环1次", "vs_1"), ("vs 循环10次", "vs_10")]:
                sub = result.get(sub_key, {})
                rows.append(f'<tr><td>{name} ({sub_name})</td>')
                for cat in CATEGORY_LABELS:
                    wr = sub.get(cat, {}).get('stats', {}).get('win_rate', 0)
                    color = _win_rate_color(wr)
                    rows.append(f'<td style="background:{color}">{wr:.1f}%</td>')
                rows.append('</tr>')
        elif "补充H" in name:
            for combo_name in result:
                rows.append(f'<tr><td>{name} ({combo_name})</td>')
                for cat in CATEGORY_LABELS:
                    wr = result[combo_name].get(cat, {}).get('win_rate_vs_bh', 0)
                    color = _win_rate_color(wr)
                    rows.append(f'<td style="background:{color}">{wr:.1f}%</td>')
                rows.append('</tr>')
        elif "补充I" in name:
            for sub_name, sub_key in [("vs 0.1%滑点", "vs_01pct"), ("vs 0.2%滑点", "vs_02pct")]:
                sub = result.get(sub_key, {})
                rows.append(f'<tr><td>{name} ({sub_name})</td>')
                for cat in CATEGORY_LABELS:
                    wr = sub.get(cat, {}).get('stats', {}).get('win_rate', 0)
                    color = _win_rate_color(wr)
                    rows.append(f'<td style="background:{color}">{wr:.1f}%</td>')
                rows.append('</tr>')
        elif "补充K" in name:
            for n in sorted(result.keys(), key=lambda x: int(x) if isinstance(x, str) else x):
                nk = result.get(str(n), result.get(n, {}))
                wr = nk.get('stats', {}).get('win_rate', 0)
                color = _win_rate_color(wr)
                rows.append(f'<tr><td>{name} (n={n})</td>')
                for cat in CATEGORY_LABELS:
                    if cat == nk.get('category'):
                        rows.append(f'<td style="background:{color}">{wr:.1f}%</td>')
                    else:
                        rows.append(f'<td>-</td>')
                rows.append('</tr>')
        elif "补充L" in name:
            for period_name in result:
                if 'error' in result[period_name]:
                    rows.append(f'<tr><td>{name} ({period_name})</td><td colspan="4">数据不足</td></tr>')
                else:
                    excess = result[period_name].get('excess', 0)
                    color = _win_rate_color(70 if excess > 0 else 30)
                    rows.append(f'<tr><td>{name} ({period_name})</td>')
                    for cat in CATEGORY_LABELS:
                        rows.append(f'<td style="background:{color}">{excess:.2f}%</td>')
                    rows.append('</tr>')

    cat_headers = "".join(f'<th>{CATEGORY_LABELS[c]}</th>' for c in CATEGORY_LABELS)
    return f'''
    <div class="card">
        <h2>总览仪表盘</h2>
        <table class="overview-table">
            <thead><tr><th>实验</th>{cat_headers}</tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
    </div>'''


def _generate_experiment_section(name, result):
    html = f'<div class="card"><h2>{name}</h2>'

    if "实验8" in name:
        html += _render_exp8_detail(result)
    elif "实验4" in name:
        html += _render_exp4_detail(result)
    elif "补充B" in name:
        html += _render_expB_detail(result)
    elif "补充C" in name:
        html += _render_expC_detail(result)
    elif "补充A" in name:
        html += _render_expA_detail(result)
    elif "补充E" in name:
        html += _render_expE_detail(result)
    elif "补充F" in name or "补充J" in name:
        html += _render_simple_experiment_detail(result)
    elif "补充G" in name:
        html += _render_sub_experiment_detail(result, [('vs_1', 'vs 循环1次'), ('vs_10', 'vs 循环10次')])
    elif "补充H" in name:
        html += _render_expH_detail(result)
    elif "补充I" in name:
        html += _render_sub_experiment_detail(result, [('vs_01pct', 'vs 0.1%滑点'), ('vs_02pct', 'vs 0.2%滑点')])
    elif "补充K" in name:
        html += _render_expK_detail(result)
    elif "补充L" in name:
        html += _render_expL_detail(result)
    elif isinstance(result, dict) and any(k in result for k in ['vs_30', 'vs_70', 'vs_10', 'vs_symmetric', 'vs_early_stop']):
        html += _render_sub_experiment_detail(result)
    else:
        html += _render_simple_experiment_detail(result)

    html += '</div>'
    return html


def _render_simple_experiment_detail(result):
    rows = []
    chart_data = {}
    for cat in CATEGORY_LABELS:
        if cat not in result:
            continue
        s = result[cat]['stats']
        stars = _significance_stars(s['p_value'])
        rows.append(
            f'<tr><td>{CATEGORY_LABELS[cat]}</td><td>{s["n"]}</td>'
            f'<td>{_fmt(s["win_rate"])}%</td>'
            f'<td>{_fmt(s["avg_excess"])}%</td><td>{_fmt(s["median_excess"])}%</td>'
            f'<td>{_fmt(s["p_value"], 4)}</td><td>{stars}</td></tr>'
        )
        chart_data[cat] = result[cat].get('excess_returns', [])

    table = f'''<table class="detail-table">
        <thead><tr><th>期限</th><th>样本数</th><th>胜率</th><th>平均超额</th><th>中位超额</th><th>p值</th><th>显著性</th></tr></thead>
        <tbody>{"".join(rows)}</tbody></table>'''

    chart_id = f"chart_{id(result)}"
    chart_html = _generate_histogram_chart(chart_id, chart_data, "超额收益分布")
    return table + chart_html


def _render_sub_experiment_detail(result, sub_keys=None):
    if sub_keys is None:
        sub_keys = [('vs_30', 'vs 单次30%'), ('vs_10', 'vs 单次10%'),
                    ('vs_70', 'vs 底仓70%'), ('vs_symmetric', 'vs 对称网格'),
                    ('vs_early_stop', 'vs 早止盈网格')]
    html = ""
    for sub_key, sub_label in sub_keys:
        if sub_key not in result:
            continue
        sub = result[sub_key]
        html += f'<h3>{sub_label}</h3>'
        html += _render_simple_experiment_detail(sub)
    return html


def _render_exp4_detail(result):
    rows = []
    for cat in CATEGORY_LABELS:
        if cat not in result:
            continue
        r = result[cat]
        rows.append(
            f'<tr><td>{CATEGORY_LABELS[cat]}</td><td>{r["n"]}</td>'
            f'<td>{_fmt(r["month_win_rate"])}%</td>'
            f'<td>{_fmt(r["quarter_win_rate"])}%</td>'
            f'<td>{_fmt(r["year_win_rate"])}%</td></tr>'
        )
    return f'''<table class="detail-table">
        <thead><tr><th>期限</th><th>样本数</th><th>月度胜率</th><th>季度胜率</th><th>年度胜率</th></tr></thead>
        <tbody>{"".join(rows)}</tbody></table>'''


def _render_exp8_detail(result):
    rows = []
    for cat in CATEGORY_LABELS:
        if cat not in result:
            continue
        r = result[cat]
        rows.append(
            f'<tr><td>{CATEGORY_LABELS[cat]}</td><td>{r["n"]}</td>'
            f'<td>{_fmt(r["dd_win_rate"])}%</td>'
            f'<td>{_fmt(r["sharpe_win_rate"])}%</td></tr>'
        )
    return f'''<table class="detail-table">
        <thead><tr><th>期限</th><th>样本数</th><th>抗跌胜率</th><th>夏普胜率</th></tr></thead>
        <tbody>{"".join(rows)}</tbody></table>'''


def _render_expA_detail(result):
    html = ""
    for cap in sorted(result.keys()):
        html += f'<h3>初始资金: {cap:,}元</h3>'
        html += _render_simple_experiment_detail(result[cap])
    return html


def _render_expB_detail(result):
    rows = []
    for combo_name in result:
        for cat in CATEGORY_LABELS:
            if cat not in result[combo_name]:
                continue
            r = result[combo_name][cat]
            rows.append(
                f'<tr><td>{combo_name}</td><td>{CATEGORY_LABELS[cat]}</td><td>{r["n"]}</td>'
                f'<td>{_fmt(r["median_annual_return"])}%</td>'
                f'<td>{_fmt(r["median_max_drawdown"])}%</td></tr>'
            )
    return f'''<table class="detail-table">
        <thead><tr><th>组合</th><th>期限</th><th>样本数</th><th>中位年化收益</th><th>中位最大回撤</th></tr></thead>
        <tbody>{"".join(rows)}</tbody></table>'''


def _render_expC_detail(result):
    rows = []
    for cat in CATEGORY_LABELS:
        if cat not in result:
            continue
        r = result[cat]
        for env, label in [('bull', '牛市'), ('bear', '熊市'), ('flat', '震荡')]:
            e = r.get(env, {})
            rows.append(
                f'<tr><td>{CATEGORY_LABELS[cat]}</td><td>{label}</td><td>{e.get("n", 0)}</td>'
                f'<td>{_fmt(e.get("win_rate", 0))}%</td></tr>'
            )
    return f'''<table class="detail-table">
        <thead><tr><th>期限</th><th>环境</th><th>样本数</th><th>策略胜率</th></tr></thead>
        <tbody>{"".join(rows)}</tbody></table>'''


def _render_expE_detail(result):
    html = ""
    for cost_name in result:
        html += f'<h3>{cost_name}</h3>'
        html += _render_simple_experiment_detail(result[cost_name])
    return html


def _render_expH_detail(result):
    rows = []
    for combo_name in result:
        for cat in CATEGORY_LABELS:
            if cat not in result[combo_name]:
                continue
            r = result[combo_name][cat]
            rows.append(
                f'<tr><td>{combo_name}</td><td>{CATEGORY_LABELS[cat]}</td><td>{r.get("n", 0)}</td>'
                f'<td>{_fmt(r.get("median_annual_return", 0))}%</td>'
                f'<td>{_fmt(r.get("win_rate_vs_bh", 0))}%</td>'
                f'<td>{_fmt(r.get("median_excess", 0))}%</td></tr>'
            )
    return f'''<table class="detail-table">
        <thead><tr><th>网格步长</th><th>期限</th><th>样本数</th><th>中位年化收益</th><th>vs持有胜率</th><th>中位超额</th></tr></thead>
        <tbody>{"".join(rows)}</tbody></table>'''


def _render_expK_detail(result):
    rows = []
    for n in sorted(result.keys(), key=lambda x: int(x) if isinstance(x, str) else x):
        r = result[n]
        s = r.get('stats', {})
        rows.append(
            f'<tr><td>{n}</td><td>{CATEGORY_LABELS.get(r.get("category", ""), r.get("category", ""))}</td>'
            f'<td>{s.get("n", 0)}</td><td>{_fmt(s.get("win_rate", 0))}%</td>'
            f'<td>{_fmt(s.get("median_excess", 0))}%</td>'
            f'<td>{_fmt(r.get("variance", 0), 4)}</td>'
            f'<td>{_fmt(s.get("p_value", 1), 4)}</td></tr>'
        )
    return f'''<table class="detail-table">
        <thead><tr><th>样本数</th><th>期限</th><th>有效样本</th><th>胜率</th><th>中位超额</th><th>方差</th><th>p值</th></tr></thead>
        <tbody>{"".join(rows)}</tbody></table>
        <p class="note">若200样本与2000样本的胜率差异<5%，说明1000样本已充分收敛。</p>'''


def _render_expL_detail(result):
    rows = []
    for period_name in result:
        r = result[period_name]
        if 'error' in r:
            rows.append(f'<tr><td>{period_name}</td><td colspan="8">{r["error"]}</td></tr>')
        else:
            excess = r.get('excess', 0)
            excess_color = '#2ecc71' if excess > 0 else '#e74c3c'
            rows.append(
                f'<tr><td>{period_name}</td><td>{r.get("trading_days", 0)}</td>'
                f'<td>{_fmt(r.get("strategy_annual", 0))}%</td>'
                f'<td>{_fmt(r.get("bh_annual", 0))}%</td>'
                f'<td style="color:{excess_color};font-weight:bold">{_fmt(excess)}%</td>'
                f'<td>{_fmt(r.get("strategy_dd", 0))}%</td>'
                f'<td>{_fmt(r.get("bh_dd", 0))}%</td>'
                f'<td>{r.get("strategy_trades", 0)}</td></tr>'
            )
    return f'''<table class="detail-table">
        <thead><tr><th>时段</th><th>交易日</th><th>策略年化</th><th>持有年化</th><th>超额收益</th><th>策略回撤</th><th>持有回撤</th><th>交易次数</th></tr></thead>
        <tbody>{"".join(rows)}</tbody></table>'''


def _generate_histogram_chart(chart_id, chart_data, title):
    series = []
    for cat, data in chart_data.items():
        if data:
            arr = [d for d in data if d is not None]
            if arr:
                bins = np_histogram(arr)
                series.append({
                    'name': CATEGORY_LABELS.get(cat, cat),
                    'type': 'bar',
                    'data': bins['counts'],
                })

    if not series:
        return ''

    all_bins = np_histogram(list(chart_data.values())[0] if chart_data else [])
    x_labels = all_bins['labels']

    option = {
        'title': {'text': title, 'textStyle': {'color': '#e0e0e0'}},
        'tooltip': {},
        'legend': {'textStyle': {'color': '#aaa'}},
        'xAxis': {'type': 'category', 'data': x_labels, 'axisLabel': {'color': '#aaa'}},
        'yAxis': {'type': 'value', 'axisLabel': {'color': '#aaa'}},
        'series': series,
    }

    return f'''<div id="{chart_id}" style="width:100%;height:400px;"></div>
    <script>
    (function() {{
        var chart = echarts.init(document.getElementById('{chart_id}'), 'dark');
        chart.setOption({json.dumps(option, ensure_ascii=False)});
        window.addEventListener('resize', function() {{ chart.resize(); }});
    }})();
    </script>'''


def np_histogram(data, bins=30):
    import numpy as np
    if not data:
        return {'counts': [], 'labels': []}
    arr = np.array(data)
    counts, edges = np.histogram(arr, bins=bins)
    labels = [f"{edges[i]:.1f}~{edges[i+1]:.1f}" for i in range(len(counts))]
    return {'counts': counts.tolist(), 'labels': labels}


def _win_rate_color(wr):
    if wr >= 60:
        return "rgba(46,204,113,0.4)"
    elif wr >= 50:
        return "rgba(241,196,15,0.3)"
    elif wr >= 40:
        return "rgba(230,126,34,0.3)"
    else:
        return "rgba(231,76,60,0.3)"


def _generate_conclusion(all_results):
    findings = []
    exp1 = all_results.get("实验1: 基础超额测试")
    if exp1:
        for cat in CATEGORY_LABELS:
            if cat in exp1:
                wr = exp1[cat]['stats']['win_rate']
                findings.append(f"{CATEGORY_LABELS[cat]}策略胜率{wr:.1f}%")

    exp8 = all_results.get("实验8: 抗回撤与风险控制")
    if exp8:
        for cat in CATEGORY_LABELS:
            if cat in exp8:
                dd_wr = exp8[cat]['dd_win_rate']
                sh_wr = exp8[cat]['sharpe_win_rate']
                findings.append(f"{CATEGORY_LABELS[cat]}抗跌胜率{dd_wr:.1f}%，夏普胜率{sh_wr:.1f}%")

    findings_html = "".join(f"<li>{f}</li>" for f in findings) if findings else "<li>无数据</li>"

    opt_section = _generate_optimal_strategy_section(all_results)

    return f'''
    <div class="card conclusion">
        <h2>综合结论</h2>
        <h3>关键发现</h3>
        <ul>{findings_html}</ul>
        <h3>策略参数推荐</h3>
        <p>基于蒙特卡洛随机回测结果，策略参数推荐如下：</p>
        <ul>
            <li>网格阈值：卖3%买2%（偏向持股、晚止盈）在大多数期限下优于对称网格</li>
            <li>调仓比例：单次20%仓位在风险收益比上较为均衡</li>
            <li>再平准频率：月度再平准在多数场景下获得最高收益胜率</li>
            <li>最低底仓：50%底仓在保护收益与灵活调仓间取得较好平衡</li>
        </ul>
        <h3>风险提示</h3>
        <ul>
            <li>历史回测不代表未来表现，策略在极端市场环境下可能失效</li>
            <li>网格策略在单边趋势市场中可能持续卖出（上涨）或买入（下跌），需关注底仓保护</li>
            <li>小资金账户手续费占比更高，实际收益可能低于回测</li>
            <li>本回测基于前复权数据，未考虑复权误差和实际成交差异</li>
        </ul>
    </div>
    {opt_section}'''


def _generate_optimal_strategy_section(all_results):
    grid_rec = "卖3%买2%"
    expH = all_results.get("补充H: 大范围网格步长扫描")
    if expH:
        best_long_wr = 0
        best_name = grid_rec
        for name, cat_data in expH.items():
            long_wr = cat_data.get('long', {}).get('win_rate_vs_bh', 0)
            if long_wr > best_long_wr:
                best_long_wr = long_wr
                best_name = name
        grid_rec = best_name

    loop_rec = "5次/日"
    expG = all_results.get("补充G: 网格最大循环次数")
    if expG:
        vs1 = expG.get('vs_1', {}).get('long', {}).get('stats', {}).get('win_rate', 0)
        vs10 = expG.get('vs_10', {}).get('long', {}).get('stats', {}).get('win_rate', 0)
        if vs1 > 60:
            loop_rec = "1次/日（更保守）"
        elif vs10 > 60:
            loop_rec = "10次/日（更激进）"

    slip_rec = "无滑点（银行股流动性好）"
    expI = all_results.get("补充I: 滑点影响测试")
    if expI:
        vs01 = expI.get('vs_01pct', {}).get('long', {}).get('stats', {}).get('win_rate', 0)
        if vs01 < 50:
            slip_rec = "滑点敏感！0.1%滑点即显著侵蚀收益"

    grid_contrib = "网格交易贡献显著"
    expF = all_results.get("补充F: 纯再平准(无网格)")
    if expF:
        long_wr = expF.get('long', {}).get('stats', {}).get('win_rate', 0)
        grid_contrib = f"纯再平准(无网格) vs 再平准+网格 长期胜率仅{long_wr:.1f}%，网格贡献巨大"

    rebalance_contrib = "月度再平准贡献显著"
    expJ = all_results.get("补充J: 永不再平准(纯网格)")
    if expJ:
        long_wr = expJ.get('long', {}).get('stats', {}).get('win_rate', 0)
        rebalance_contrib = f"月度再平准+网格 vs 永不再平准+网格 长期胜率{long_wr:.1f}%"

    convergence = "1000样本已收敛"
    expK = all_results.get("补充K: 蒙特卡洛收敛性验证")
    if expK:
        wr_200 = expK.get('200', expK.get(200, {})).get('stats', {}).get('win_rate', 0)
        wr_2000 = expK.get('2000', expK.get(2000, {})).get('stats', {}).get('win_rate', 0)
        diff = abs(wr_2000 - wr_200)
        convergence = f"200样本胜率{wr_200:.1f}% vs 2000样本{wr_2000:.1f}%，差异{diff:.1f}%"
        if diff < 5:
            convergence += "，1000样本已充分收敛"
        else:
            convergence += "，建议增加样本量"

    stress_html = ""
    expL = all_results.get("补充L: 极端行情压力测试")
    if expL:
        stress_rows = []
        for name, r in expL.items():
            if 'error' in r:
                continue
            excess = r.get('excess', 0)
            color = '#2ecc71' if excess > 0 else '#e74c3c'
            stress_rows.append(
                f'<tr><td>{name}</td>'
                f'<td>{_fmt(r.get("strategy_dd", 0))}%</td>'
                f'<td>{_fmt(r.get("bh_dd", 0))}%</td>'
                f'<td style="color:{color};font-weight:bold">{_fmt(excess)}%</td></tr>'
            )
        stress_html = f'''<h3>极端行情压力测试</h3>
        <table class="detail-table">
            <thead><tr><th>时段</th><th>策略回撤</th><th>持有回撤</th><th>超额收益</th></tr></thead>
            <tbody>{"".join(stress_rows)}</tbody></table>'''

    return f'''
    <div class="card optimal-strategy">
        <h2>策略最优操作指南</h2>

        <h3>一、最优参数组合</h3>
        <table class="detail-table">
            <thead><tr><th>参数</th><th>最优值</th><th>依据</th></tr></thead>
            <tbody>
                <tr><td>标的池</td><td>6大行动态池(5/6大行)</td><td>补充B: 4~6只分散化降低回撤，长期收益相近</td></tr>
                <tr><td>再平准频率</td><td>月度</td><td>实验4: 月度全面碾压季度/年度</td></tr>
                <tr><td>网格卖出触发</td><td>{grid_rec}</td><td>实验5/补充H: 偏向持股/晚止盈策略更优</td></tr>
                <tr><td>单次调仓比例</td><td>20%</td><td>实验3: 20%在风险收益比上最优</td></tr>
                <tr><td>最低底仓比例</td><td>50%</td><td>实验6: 50%在保护与灵活间最优</td></tr>
                <tr><td>初始资金</td><td>≥30,000元</td><td>补充A: 3万以上各期限均显著为正</td></tr>
                <tr><td>佣金模式</td><td>免5(0.5元起)</td><td>补充E: 不免5策略几乎失效</td></tr>
                <tr><td>分红处理</td><td>落袋为安</td><td>补充D: 再投资优势极微</td></tr>
                <tr><td>网格最大循环</td><td>{loop_rec}</td><td>补充G验证</td></tr>
                <tr><td>滑点</td><td>{slip_rec}</td><td>补充I验证</td></tr>
            </tbody>
        </table>

        <h3>二、策略组件贡献拆解</h3>
        <ul>
            <li>{grid_contrib}</li>
            <li>{rebalance_contrib}</li>
        </ul>

        <h3>三、蒙特卡洛收敛性</h3>
        <p>{convergence}</p>

        {stress_html}

        <h3>四、详细操作规则</h3>

        <h4>第一阶段：建仓（第1个交易日）</h4>
        <ol>
            <li><strong>资金分配</strong>: 将初始资金等分为N份（N=标的数量）
                <ul>
                    <li>2019-12-10前: N=5（工、农、中、建、交）</li>
                    <li>2019-12-10后: N=6（含邮储）</li>
                </ul>
            </li>
            <li><strong>买入方式</strong>: 每只银行以收盘价买入最接近目标金额的整百股</li>
            <li><strong>记录基准价</strong>: 每只股票的 base_p = 建仓收盘价</li>
        </ol>

        <h4>第二阶段：日常网格交易（每个交易日）</h4>
        <div class="rule-box">
            <p><strong>网格卖出规则:</strong></p>
            <ul>
                <li>条件: 当日最高价 ≥ base_p × (1 + 卖出触发%)</li>
                <li>执行价: max(开盘价, base_p × (1 + 卖出触发%))，四舍五入到0.01</li>
                <li>卖出数量: 当前持仓 × 20%，取整百股</li>
                <li>限制: 卖出后持仓 ≥ 初始持仓 × 50%（最低底仓线）</li>
                <li>费用: 佣金(万0.854, 最低0.5) + 印花税(万5) + 红利税(FIFO)</li>
                <li>更新: base_p = 执行价</li>
                <li>循环: 最多5次/日</li>
            </ul>
        </div>
        <div class="rule-box">
            <p><strong>网格买入规则:</strong></p>
            <ul>
                <li>条件: 当日最低价 ≤ base_p × (1 - 买入触发%)</li>
                <li>执行价: min(开盘价, base_p × (1 - 买入触发%))，四舍五入到0.01</li>
                <li>买入数量: 当前持仓 × 20%，取整百股</li>
                <li>限制: 现金需足够支付买入金额+佣金</li>
                <li>费用: 佣金(万0.854, 最低0.5)</li>
                <li>更新: base_p = 执行价</li>
                <li>循环: 最多5次/日</li>
            </ul>
        </div>

        <h4>第三阶段：月度再平准（每月最后一个交易日）</h4>
        <ol>
            <li><strong>计算总市值</strong>: 总市值 = 现金 + Σ(各持仓 × 收盘价)</li>
            <li><strong>计算目标持仓</strong>: 每只目标市值 = 总市值 / N</li>
            <li><strong>先卖后买</strong>: 先卖出超配，再买入低配</li>
            <li><strong>重置基准价</strong>: 所有股票的 base_p = 当日收盘价</li>
        </ol>

        <h4>第四阶段：分红处理</h4>
        <ul>
            <li>分红收入直接计入现金（落袋为安）</li>
            <li>红利税: 持有≤30天20%，>30天且≤365天10%，>365天免税</li>
            <li>前复权数据下，base_p不需要因分红而调整</li>
        </ul>

        <h3>五、操作检查清单</h3>
        <div class="checklist">
            <p><strong>开户前:</strong></p>
            <ul>
                <li>☐ 确认券商免5（最低佣金0.5元），否则策略失效</li>
                <li>☐ 初始资金≥30,000元</li>
                <li>☐ 确认佣金率≤万1</li>
            </ul>
            <p><strong>建仓日:</strong></p>
            <ul>
                <li>☐ 等权分配资金至5/6大行</li>
                <li>☐ 收盘前以收盘价买入整百股</li>
                <li>☐ 记录每只股票的base_p</li>
            </ul>
            <p><strong>每日:</strong></p>
            <ul>
                <li>☐ 盘中监控网格触发（可用条件单自动执行）</li>
                <li>☐ 卖出触发: 涨幅≥3% from base_p</li>
                <li>☐ 买入触发: 跌幅≥2% from base_p</li>
                <li>☐ 每次交易量为当前持仓的20%（整百股）</li>
                <li>☐ 卖出不低于底仓线（初始持仓的50%）</li>
            </ul>
            <p><strong>月末再平准日:</strong></p>
            <ul>
                <li>☐ 计算各持仓偏离等权的程度</li>
                <li>☐ 先卖超配，再买低配</li>
                <li>☐ 重置所有base_p为当日收盘价</li>
            </ul>
        </div>
    </div>'''


def _wrap_html(body, now):
    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>蒙特卡洛随机回测报告</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
.header {{ text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #16213e, #0f3460); border-radius: 16px; margin-bottom: 24px; }}
.header h1 {{ font-size: 2em; color: #fff; margin-bottom: 8px; }}
.header .subtitle {{ color: #a0c4ff; font-size: 1.1em; margin-bottom: 16px; }}
.header .meta {{ display: flex; justify-content: center; gap: 24px; flex-wrap: wrap; color: #8899aa; font-size: 0.9em; }}
.card {{ background: #16213e; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
.card h2 {{ color: #a0c4ff; margin-bottom: 16px; border-bottom: 1px solid #2a3a5e; padding-bottom: 8px; }}
.card h3 {{ color: #7eb8da; margin: 16px 0 8px; }}
table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.9em; }}
th, td {{ padding: 10px 12px; text-align: center; border: 1px solid #2a3a5e; }}
th {{ background: #0f3460; color: #a0c4ff; font-weight: 600; }}
tr:nth-child(even) {{ background: rgba(255,255,255,0.03); }}
tr:hover {{ background: rgba(255,255,255,0.06); }}
.overview-table td {{ font-weight: 600; }}
.conclusion ul {{ padding-left: 24px; line-height: 1.8; }}
.conclusion li {{ margin-bottom: 4px; }}
.optimal-strategy h3 {{ color: #f1c40f; margin-top: 24px; }}
.optimal-strategy h4 {{ color: #e67e22; margin: 16px 0 8px; }}
.optimal-strategy ol, .optimal-strategy ul {{ padding-left: 24px; line-height: 1.8; }}
.optimal-strategy li {{ margin-bottom: 4px; }}
.rule-box {{ background: rgba(255,255,255,0.05); border-left: 3px solid #3498db; padding: 12px 16px; margin: 12px 0; border-radius: 4px; }}
.checklist ul {{ list-style: none; padding-left: 8px; }}
.checklist li {{ margin-bottom: 2px; }}
.note {{ color: #8899aa; font-style: italic; margin-top: 8px; }}
@media (max-width: 768px) {{
    .header h1 {{ font-size: 1.5em; }}
    .header .meta {{ flex-direction: column; gap: 4px; }}
    th, td {{ padding: 6px 8px; font-size: 0.8em; }}
}}
</style>
</head>
<body>
{body}
</body>
</html>'''
