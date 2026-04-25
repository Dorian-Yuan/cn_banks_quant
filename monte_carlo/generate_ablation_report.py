import json
import os

CATEGORY_ORDER = ['short', 'medium', 'medium_long', 'long']
CATEGORY_LABELS = {'short': '短期(0~6个月)', 'medium': '中期(7~12个月)',
                   'medium_long': '中长期(12~24个月)', 'long': '长期(>24个月)'}
STRATEGY_LABELS = {
    'A': '卖1.2%买0.5%+月度再平准',
    'B': '卖1.0%买0.5%+月度再平准',
    'C': '卖3.0%买2.0%+月度再平准',
    'D': '卖1.2%买0.5%+永不再平准',
    'E': '纯再平准(无网格)',
    'F': '买入持有',
}


def _fmt(v, d=2):
    if v is None:
        return '-'
    return f'{v:.{d}f}'


def _sig_label(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def _wr_color(wr):
    if wr >= 70:
        return 'rgba(46,204,113,0.4)'
    if wr >= 55:
        return 'rgba(46,204,113,0.2)'
    if wr >= 45:
        return 'rgba(241,196,15,0.3)'
    if wr >= 30:
        return 'rgba(231,76,60,0.2)'
    return 'rgba(231,76,60,0.4)'


def _build_ablation_table(ablation_results):
    rows = []
    for strategy_key in ['A', 'B', 'C', 'D', 'E', 'F']:
        label = STRATEGY_LABELS[strategy_key]
        rows.append(f'<tr><td rowspan="4"><strong>{strategy_key}</strong><br><small>{label}</small></td>')
        for i, cat in enumerate(CATEGORY_ORDER):
            d = ablation_results.get(cat, {}).get(strategy_key, {})
            n = d.get('n', 0)
            mar = d.get('median_annual_return', 0)
            mdd = d.get('median_max_drawdown', 0)
            wr = d.get('win_rate_vs_bh', 0) if strategy_key != 'F' else '-'
            me = d.get('median_excess', 0) if strategy_key != 'F' else '-'
            pv = d.get('p_value', 1) if strategy_key != 'F' else '-'
            sig = _sig_label(pv) if isinstance(pv, float) else ''

            wr_cell = f'<td style="background:{_wr_color(wr)}">{wr:.1f}%</td>' if isinstance(wr, (int, float)) else f'<td>{wr}</td>'
            me_cell = f'<td>{me:.2f}%</td>' if isinstance(me, (int, float)) else f'<td>{me}</td>'
            pv_cell = f'<td>{pv:.4f}</td>' if isinstance(pv, float) else f'<td>{pv}</td>'

            prefix = '</tr><tr>' if i > 0 else ''
            rows.append(f'{prefix}<td>{CATEGORY_LABELS[cat]}</td><td>{n}</td><td>{_fmt(mar)}%</td><td>{_fmt(mdd)}%</td>{wr_cell}{me_cell}{pv_cell}<td>{sig}</td>')
        rows.append('</tr>')
    return f'''<table class="detail-table">
        <thead><tr><th>策略</th><th>期限</th><th>样本</th><th>中位年化</th><th>中位回撤</th><th>vs持有胜率</th><th>中位超额</th><th>p值</th><th>显著性</th></tr></thead>
        <tbody>{"".join(rows)}</tbody></table>'''


def _build_boxplot_data(ablation_results):
    datasets = {}
    for cat in CATEGORY_ORDER:
        box_data = []
        outlier_data = []
        labels = []
        for strategy_key in ['A', 'B', 'C', 'D', 'E', 'F']:
            d = ablation_results.get(cat, {}).get(strategy_key, {})
            ar = d.get('annual_returns', [])
            if ar:
                sorted_ar = sorted(ar)
                q1 = sorted_ar[len(sorted_ar) // 4]
                q2 = sorted_ar[len(sorted_ar) // 2]
                q3 = sorted_ar[3 * len(sorted_ar) // 4]
                iqr = q3 - q1
                lower = max(sorted_ar[0], q1 - 1.5 * iqr)
                upper = min(sorted_ar[-1], q3 + 1.5 * iqr)
                box_data.append([lower, q1, q2, q3, upper])
                outliers = [v for v in sorted_ar if v < lower or v > upper]
                for o in outliers:
                    outlier_data.append([labels.__len__(), o])
            else:
                box_data.append([0, 0, 0, 0, 0])
            labels.append(f'{strategy_key}: {STRATEGY_LABELS[strategy_key][:12]}')
        datasets[cat] = {
            'box_data': box_data,
            'outlier_data': outlier_data,
            'labels': labels,
            'title': CATEGORY_LABELS[cat],
        }
    return datasets


def _build_heatmap_data(grid_results):
    combos = grid_results.get('_combos', [])
    sell_steps = sorted(set(s for s, b in combos))
    buy_steps = sorted(set(b for s, b in combos))

    heatmap_data = {}
    for cat in CATEGORY_ORDER:
        cat_data = grid_results.get(cat, {})
        mar_matrix = []
        wr_matrix = []
        for sell in sell_steps:
            mar_row = []
            wr_row = []
            for buy in buy_steps:
                key = f"卖{sell*100:.2f}%买{buy*100:.2f}%"
                d = cat_data.get(key, {})
                mar_row.append(d.get('median_annual_return', 0))
                wr_row.append(d.get('win_rate_vs_bh', 0))
            mar_matrix.append(mar_row)
            wr_matrix.append(wr_row)

        extra_points = []
        for sell, buy in combos:
            if sell not in sell_steps or buy not in buy_steps:
                key = f"卖{sell*100:.2f}%买{buy*100:.2f}%"
                d = cat_data.get(key, {})
                extra_points.append({
                    'name': key,
                    'median_annual_return': d.get('median_annual_return', 0),
                    'win_rate_vs_bh': d.get('win_rate_vs_bh', 0),
                    'sell_pct': sell,
                    'buy_pct': buy,
                })

        heatmap_data[cat] = {
            'mar_matrix': mar_matrix,
            'wr_matrix': wr_matrix,
            'sell_labels': [f'{s*100:.2f}%' for s in sell_steps],
            'buy_labels': [f'{b*100:.2f}%' for b in buy_steps],
            'extra_points': extra_points,
            'title': CATEGORY_LABELS[cat],
        }
    return heatmap_data


def _build_top3_table(grid_results):
    top3 = grid_results.get('_top3', [])
    sig = grid_results.get('_top3_significance', [])

    rows = []
    for i, (name, data) in enumerate(top3):
        rows.append(f'<tr><td>#{i+1}</td><td>{name}</td>')
        for cat in CATEGORY_ORDER:
            cat_d = grid_results.get(cat, {}).get(name, {})
            mar = cat_d.get('median_annual_return', 0)
            wr = cat_d.get('win_rate_vs_bh', 0)
            rows.append(f'<td>{_fmt(mar)}%</td><td style="background:{_wr_color(wr)}">{wr:.1f}%</td>')
        rows.append('</tr>')

    sig_rows = []
    for pair, d in sig.items():
        p = d.get('p_value', 1)
        s = '显著' if d.get('significant', False) else '不显著'
        sig_rows.append(f'<tr><td>{pair}</td><td>{p:.4f}</td><td>{s}</td></tr>')

    return f'''
    <h3>长期中位年化收益排名前3</h3>
    <table class="detail-table">
        <thead><tr><th>排名</th><th>参数</th>
        {''.join(f'<th>{CATEGORY_LABELS[cat]} 年化</th><th>{CATEGORY_LABELS[cat]} 胜率</th>' for cat in CATEGORY_ORDER)}
        </tr></thead>
        <tbody>{"".join(rows)}</tbody></table>
    <h3>前3组合配对显著性检验</h3>
    <table class="detail-table">
        <thead><tr><th>对比</th><th>p值</th><th>结论</th></tr></thead>
        <tbody>{"".join(sig_rows)}</tbody></table>'''


def _build_ai_summary(ablation_results, grid_results):
    lines = []

    d_long = ablation_results.get('long', {})
    d_wr = {}
    for k in ['A', 'B', 'C', 'D', 'E']:
        d_wr[k] = d_long.get(k, {}).get('win_rate_vs_bh', 0)

    if d_wr.get('D', 0) < 50 and d_wr.get('E', 0) < 50:
        lines.append('<li><strong>耦合效应</strong>：纯网格(D)和纯再平准(E)长期胜率均低于50%，说明<strong>网格与再平准必须协同使用</strong>，单独使用任一模块都会大幅失效。</li>')
    elif d_wr.get('D', 0) < 50:
        lines.append(f'<li><strong>纯网格失效</strong>：策略D(纯网格)长期胜率仅{d_wr["D"]:.1f}%，远低于完整策略A的{d_wr.get("A",0):.1f}%，<strong>再平准是策略不可或缺的组件</strong>。</li>')
    elif d_wr.get('E', 0) < 50:
        lines.append(f'<li><strong>纯再平准失效</strong>：策略E(纯再平准)长期胜率仅{d_wr["E"]:.1f}%，远低于完整策略A的{d_wr.get("A",0):.1f}%，<strong>网格交易是策略核心收益来源</strong>。</li>')
    else:
        lines.append(f'<li><strong>协同增强</strong>：完整策略A长期胜率{d_wr.get("A",0):.1f}%，高于纯网格D({d_wr["D"]:.1f}%)和纯再平准E({d_wr["E"]:.1f}%)，网格与再平准存在正向协同效应。</li>')

    a_wr = d_wr.get('A', 0)
    b_wr = d_wr.get('B', 0)
    a_me = d_long.get('A', {}).get('median_excess', 0)
    b_me = d_long.get('B', {}).get('median_excess', 0)
    if a_wr > b_wr:
        lines.append(f'<li><strong>1.2/0.5 vs 1.0/0.5</strong>：策略A(卖1.2%买0.5%)长期胜率{a_wr:.1f}%，高于策略B(卖1.0%买0.5%)的{b_wr:.1f}%，中位超额分别为{a_me:.2f}%和{b_me:.2f}%。<strong>1.2/0.5更优</strong>。</li>')
    elif b_wr > a_wr:
        lines.append(f'<li><strong>1.2/0.5 vs 1.0/0.5</strong>：策略B(卖1.0%买0.5%)长期胜率{b_wr:.1f}%，高于策略A(卖1.2%买0.5%)的{a_wr:.1f}%。<strong>1.0/0.5更优</strong>。</li>')
    else:
        lines.append(f'<li><strong>1.2/0.5 vs 1.0/0.5</strong>：两者长期胜率接近({a_wr:.1f}% vs {b_wr:.1f}%)，差异不大。</li>')

    top3 = grid_results.get('_top3', [])
    if top3:
        best_name, best_data = top3[0]
        best_mar = best_data.get('median_annual_return', 0)
        lines.append(f'<li><strong>全局最佳参数</strong>：测试二寻优得出，长期中位年化收益最高的参数组合为<strong>{best_name}</strong>，中位年化{best_mar:.2f}%。</li>')

    return ''.join(lines)


def generate_ablation_report(ablation_results, grid_results, n_samples1=1000, n_samples2=500, output_path=None):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "research", "ablation_grid_search_report.html")

    ablation_table_html = _build_ablation_table(ablation_results)
    boxplot_datasets = _build_boxplot_data(ablation_results)
    heatmap_datasets = _build_heatmap_data(grid_results)
    top3_html = _build_top3_table(grid_results)
    ai_summary = _build_ai_summary(ablation_results, grid_results)

    boxplot_json = json.dumps(boxplot_datasets, ensure_ascii=False)
    heatmap_json = json.dumps(heatmap_datasets, ensure_ascii=False)

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>策略剥离对比与极密参数寻优报告</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
.header {{ text-align: center; padding: 30px 0; border-bottom: 2px solid #3498db; margin-bottom: 30px; }}
.header h1 {{ color: #f1c40f; font-size: 2em; margin-bottom: 10px; }}
.header .meta {{ display: flex; justify-content: center; gap: 20px; color: #8899aa; font-size: 0.9em; flex-wrap: wrap; }}
.card {{ background: #16213e; border-radius: 8px; padding: 24px; margin-bottom: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
.card h2 {{ color: #3498db; margin-bottom: 16px; border-bottom: 1px solid #2c3e50; padding-bottom: 8px; }}
.card h3 {{ color: #f1c40f; margin: 16px 0 8px; }}
.detail-table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.85em; }}
.detail-table th {{ background: #0f3460; color: #f1c40f; padding: 8px 10px; text-align: center; white-space: nowrap; }}
.detail-table td {{ padding: 6px 10px; text-align: center; border-bottom: 1px solid #2c3e50; }}
.detail-table tr:hover {{ background: rgba(52,152,219,0.1); }}
.chart-container {{ width: 100%; height: 450px; margin: 16px 0; }}
.charts-row {{ display: flex; flex-wrap: wrap; gap: 16px; }}
.charts-row .chart-item {{ flex: 1; min-width: 400px; }}
.summary {{ border: 2px solid #f1c40f; }}
.summary ul {{ padding-left: 24px; line-height: 2; }}
.summary li {{ margin-bottom: 8px; }}
.note {{ color: #8899aa; font-style: italic; margin-top: 8px; }}
.extra-badge {{ display: inline-block; background: #e74c3c; color: #fff; font-size: 0.75em; padding: 2px 6px; border-radius: 3px; margin-left: 4px; }}
@media (max-width: 768px) {{
    .header h1 {{ font-size: 1.5em; }}
    th, td {{ padding: 4px 6px; font-size: 0.75em; }}
    .charts-row .chart-item {{ min-width: 100%; }}
}}
</style>
</head>
<body>
<div class="header">
    <h1>策略剥离对比与极密参数寻优报告</h1>
    <div class="meta">
        <span>蒙特卡洛随机回测</span>
        <span>测试一: {n_samples1}样本×6策略 | 测试二: {n_samples2}样本×27组合</span>
        <span>手续费: 万0.854(0.5起)+印花税万5+免5</span>
        <span>底仓50% | 调仓30% | 月度再平准</span>
    </div>
</div>

<div class="card">
    <h2>测试一：策略模块剥离与极密步长对比</h2>
    <p class="note">6组策略在完全相同的随机时间段上配对比较，验证网格与再平准的耦合协同效应</p>
    {ablation_table_html}
</div>

<div class="card">
    <h2>测试一：年化收益箱线图</h2>
    <div class="charts-row">
        <div class="chart-item"><div id="boxplot-short" class="chart-container"></div></div>
        <div class="chart-item"><div id="boxplot-medium" class="chart-container"></div></div>
    </div>
    <div class="charts-row">
        <div class="chart-item"><div id="boxplot-medium_long" class="chart-container"></div></div>
        <div class="chart-item"><div id="boxplot-long" class="chart-container"></div></div>
    </div>
</div>

<div class="card">
    <h2>测试二：极密参数二维网格寻优 - 中位年化收益热力图</h2>
    <p class="note">5×5标准网格 + 额外2点(卖1.20%买0.50%、卖1.20%买0.40%)以<span class="extra-badge">★</span>标注</p>
    <div class="charts-row">
        <div class="chart-item"><div id="heatmap-mar-short" class="chart-container"></div></div>
        <div class="chart-item"><div id="heatmap-mar-medium" class="chart-container"></div></div>
    </div>
    <div class="charts-row">
        <div class="chart-item"><div id="heatmap-mar-medium_long" class="chart-container"></div></div>
        <div class="chart-item"><div id="heatmap-mar-long" class="chart-container"></div></div>
    </div>
</div>

<div class="card">
    <h2>测试二：极密参数二维网格寻优 - vs持有胜率热力图</h2>
    <div class="charts-row">
        <div class="chart-item"><div id="heatmap-wr-short" class="chart-container"></div></div>
        <div class="chart-item"><div id="heatmap-wr-medium" class="chart-container"></div></div>
    </div>
    <div class="charts-row">
        <div class="chart-item"><div id="heatmap-wr-medium_long" class="chart-container"></div></div>
        <div class="chart-item"><div id="heatmap-wr-long" class="chart-container"></div></div>
    </div>
</div>

<div class="card">
    <h2>测试二：前3组合详细对比</h2>
    {top3_html}
</div>

<div class="card summary">
    <h2>AI 智能总结</h2>
    <ul>
        {ai_summary}
    </ul>
</div>

<script>
var boxplotData = {boxplot_json};
var heatmapData = {heatmap_json};

Object.keys(boxplotData).forEach(function(cat) {{
    var d = boxplotData[cat];
    var chart = echarts.init(document.getElementById('boxplot-' + cat));
    chart.setOption({{
        title: {{ text: d.title + ' - 年化收益分布', left: 'center', textStyle: {{ color: '#f1c40f', fontSize: 14 }} }},
        tooltip: {{ trigger: 'item' }},
        xAxis: {{ type: 'category', data: d.labels, axisLabel: {{ color: '#aaa', fontSize: 10, rotate: 30 }} }},
        yAxis: {{ type: 'value', name: '年化收益%', axisLabel: {{ color: '#aaa' }}, splitLine: {{ lineStyle: {{ color: '#2c3e50' }} }} }},
        series: [
            {{ name: '箱线', type: 'boxplot', data: d.box_data, itemStyle: {{ color: 'rgba(52,152,219,0.6)', borderColor: '#3498db' }} }},
            {{ name: '异常值', type: 'scatter', data: d.outlier_data, itemStyle: {{ color: '#e74c3c' }} }}
        ]
    }});
}});

function renderHeatmap(containerId, cat, dataType) {{
    var d = heatmapData[cat];
    var matrix = dataType === 'mar' ? d.mar_matrix : d.wr_matrix;
    var data = [];
    for (var i = 0; i < d.sell_labels.length; i++) {{
        for (var j = 0; j < d.buy_labels.length; j++) {{
            data.push([j, i, matrix[i][j]]);
        }}
    }}

    var extraScatterData = [];
    if (dataType === 'mar' && d.extra_points && d.extra_points.length > 0) {{
        d.extra_points.forEach(function(ep) {{
            var buyIdx = d.buy_labels.indexOf((ep.buy_pct * 100).toFixed(2) + '%');
            var sellIdx = d.sell_labels.indexOf((ep.sell_pct * 100).toFixed(2) + '%');
            if (buyIdx === -1) {{
                var closestBuy = 0;
                var minDist = 999;
                for (var bi = 0; bi < d.buy_labels.length; bi++) {{
                    var bv = parseFloat(d.buy_labels[bi]);
                    var dist = Math.abs(bv - ep.buy_pct * 100);
                    if (dist < minDist) {{ minDist = dist; closestBuy = bi; }}
                }}
                buyIdx = closestBuy;
            }}
            if (sellIdx === -1) {{
                var closestSell = 0;
                var minDist2 = 999;
                for (var si = 0; si < d.sell_labels.length; si++) {{
                    var sv = parseFloat(d.sell_labels[si]);
                    var dist2 = Math.abs(sv - ep.sell_pct * 100);
                    if (dist2 < minDist2) {{ minDist2 = dist2; closestSell = si; }}
                }}
                sellIdx = closestSell;
            }}
            extraScatterData.push({{
                value: [buyIdx, sellIdx, ep.median_annual_return],
                name: ep.name,
                itemStyle: {{ color: '#e74c3c', borderColor: '#fff', borderWidth: 2 }}
            }});
        }});
    }} else if (dataType === 'wr' && d.extra_points && d.extra_points.length > 0) {{
        d.extra_points.forEach(function(ep) {{
            var buyIdx = d.buy_labels.indexOf((ep.buy_pct * 100).toFixed(2) + '%');
            var sellIdx = d.sell_labels.indexOf((ep.sell_pct * 100).toFixed(2) + '%');
            if (buyIdx === -1) {{
                var closestBuy = 0;
                var minDist = 999;
                for (var bi = 0; bi < d.buy_labels.length; bi++) {{
                    var bv = parseFloat(d.buy_labels[bi]);
                    var dist = Math.abs(bv - ep.buy_pct * 100);
                    if (dist < minDist) {{ minDist = dist; closestBuy = bi; }}
                }}
                buyIdx = closestBuy;
            }}
            if (sellIdx === -1) {{
                var closestSell = 0;
                var minDist2 = 999;
                for (var si = 0; si < d.sell_labels.length; si++) {{
                    var sv = parseFloat(d.sell_labels[si]);
                    var dist2 = Math.abs(sv - ep.sell_pct * 100);
                    if (dist2 < minDist2) {{ minDist2 = dist2; closestSell = si; }}
                }}
                sellIdx = closestSell;
            }}
            extraScatterData.push({{
                value: [buyIdx, sellIdx, ep.win_rate_vs_bh],
                name: ep.name,
                itemStyle: {{ color: '#e74c3c', borderColor: '#fff', borderWidth: 2 }}
            }});
        }});
    }}

    var chart = echarts.init(document.getElementById(containerId));
    var valLabel = dataType === 'mar' ? '中位年化收益%' : 'vs持有胜率%';
    var seriesList = [
        {{ name: valLabel, type: 'heatmap', data: data, label: {{ show: true, color: '#fff', fontSize: 10, formatter: function(p) {{ return p.value[2].toFixed(1); }} }} }}
    ];
    if (extraScatterData.length > 0) {{
        seriesList.push({{
            name: '额外参数点',
            type: 'scatter',
            data: extraScatterData,
            symbolSize: 30,
            label: {{ show: true, formatter: '★', color: '#fff', fontSize: 14, fontWeight: 'bold' }},
            tooltip: {{ formatter: function(p) {{ return p.name + '<br>' + valLabel + ': ' + p.value[2].toFixed(2) + '%'; }} }},
            z: 10
        }});
    }}

    chart.setOption({{
        title: {{ text: d.title + ' - ' + valLabel, left: 'center', textStyle: {{ color: '#f1c40f', fontSize: 14 }} }},
        tooltip: {{ position: 'top', formatter: function(p) {{ return '卖出: ' + d.sell_labels[p.value[1]] + '<br>买入: ' + d.buy_labels[p.value[0]] + '<br>' + valLabel + ': ' + p.value[2].toFixed(2) + '%'; }} }},
        grid: {{ left: '15%', right: '15%', bottom: '15%' }},
        xAxis: {{ type: 'category', data: d.buy_labels, name: '买入步长', axisLabel: {{ color: '#aaa' }}, nameTextStyle: {{ color: '#aaa' }} }},
        yAxis: {{ type: 'category', data: d.sell_labels, name: '卖出步长', axisLabel: {{ color: '#aaa' }}, nameTextStyle: {{ color: '#aaa' }} }},
        visualMap: {{ min: dataType === 'mar' ? 15 : 50, max: dataType === 'mar' ? 50 : 100, calculable: true, orient: 'horizontal', left: 'center', bottom: '0%', inRange: {{ color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#fee090', '#fdae61', '#f46d43', '#d73027'] }}, textStyle: {{ color: '#aaa' }} }},
        series: seriesList
    }});
}}

renderHeatmap('heatmap-mar-short', 'short', 'mar');
renderHeatmap('heatmap-mar-medium', 'medium', 'mar');
renderHeatmap('heatmap-mar-medium_long', 'medium_long', 'mar');
renderHeatmap('heatmap-mar-long', 'long', 'mar');
renderHeatmap('heatmap-wr-short', 'short', 'wr');
renderHeatmap('heatmap-wr-medium', 'medium', 'wr');
renderHeatmap('heatmap-wr-medium_long', 'medium_long', 'wr');
renderHeatmap('heatmap-wr-long', 'long', 'wr');
</script>
</body>
</html>'''

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return output_path
