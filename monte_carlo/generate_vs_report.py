import json
import os

CATEGORY_ORDER = ['short', 'medium', 'medium_long', 'long']
CATEGORY_LABELS = {'short': '短期(0~6个月)', 'medium': '中期(7~12个月)',
                   'medium_long': '中长期(12~24个月)', 'long': '长期(>24个月)'}


def _fmt(v, d=2):
    if v is None:
        return '-'
    return f'{v:.{d}f}'


def _sig(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def _wr_bg(wr):
    if wr >= 60:
        return 'rgba(46,204,113,0.4)'
    if wr >= 55:
        return 'rgba(46,204,113,0.2)'
    if wr >= 45:
        return 'rgba(241,196,15,0.3)'
    return 'rgba(231,76,60,0.3)'


def _build_base_table(base):
    rows = []
    for cat in CATEGORY_ORDER:
        d = base.get(cat, {})
        n = d.get('n', 0)
        a_ar = d.get('a_median_ar', 0)
        b_ar = d.get('b_median_ar', 0)
        a_dd = d.get('a_median_dd', 0)
        b_dd = d.get('b_median_dd', 0)
        wr = d.get('win_rate', 0)
        me = d.get('median_excess', 0)
        pv = d.get('p_value', 1)
        rows.append(f'''<tr>
            <td>{CATEGORY_LABELS[cat]}</td><td>{n}</td>
            <td>{_fmt(a_ar)}%</td><td>{_fmt(b_ar)}%</td>
            <td style="color:{'#2ecc71' if a_ar > b_ar else '#e74c3c'}">{_fmt(a_ar-b_ar,2)}%</td>
            <td>{_fmt(a_dd)}%</td><td>{_fmt(b_dd)}%</td>
            <td style="background:{_wr_bg(wr)}">{wr:.1f}%</td>
            <td>{_fmt(me)}%</td><td>{pv:.4f}</td><td>{_sig(pv)}</td>
        </tr>''')
    return f'''<table class="dt"><thead><tr>
        <th>期限</th><th>样本</th>
        <th>A中位年化</th><th>B中位年化</th><th>年化差(A-B)</th>
        <th>A中位回撤</th><th>B中位回撤</th>
        <th>A胜率</th><th>中位超额</th><th>p值</th><th>显著性</th>
    </tr></thead><tbody>{"".join(rows)}</tbody></table>'''


def _build_boxplot_data(base):
    datasets = {}
    for cat in CATEGORY_ORDER:
        d = base.get(cat, {})
        a_ar = d.get('a_returns', [])
        b_ar = d.get('b_returns', [])
        box_data = []
        outlier_data = []
        labels = ['A: 卖1.2%买0.5%', 'B: 卖1.75%买0.75%']
        for ar_list in [a_ar, b_ar]:
            if ar_list:
                s = sorted(ar_list)
                q1 = s[len(s)//4]
                q2 = s[len(s)//2]
                q3 = s[3*len(s)//4]
                iqr = q3 - q1
                lo = max(s[0], q1 - 1.5*iqr)
                hi = min(s[-1], q3 + 1.5*iqr)
                box_data.append([lo, q1, q2, q3, hi])
                outliers = [v for v in s if v < lo or v > hi]
                for o in outliers:
                    outlier_data.append([len(box_data)-1, o])
            else:
                box_data.append([0, 0, 0, 0, 0])
        datasets[cat] = {'box_data': box_data, 'outlier_data': outlier_data,
                         'labels': labels, 'title': CATEGORY_LABELS[cat]}
    return datasets


def _build_condition_table(cond_name, cond_results):
    rows = []
    for cond_val, cond_data in cond_results.items():
        for cat in CATEGORY_ORDER:
            d = cond_data.get(cat, {})
            n = d.get('n', 0)
            wr = d.get('win_rate', 0)
            me = d.get('median_excess', 0)
            pv = d.get('p_value', 1)
            rows.append(f'''<tr>
                <td>{cond_val}</td><td>{CATEGORY_LABELS[cat]}</td><td>{n}</td>
                <td style="background:{_wr_bg(wr)}">{wr:.1f}%</td>
                <td>{_fmt(me)}%</td><td>{pv:.4f}</td><td>{_sig(pv)}</td>
            </tr>''')
    return f'''<h3>{cond_name}</h3>
        <table class="dt"><thead><tr>
            <th>条件</th><th>期限</th><th>样本</th><th>A胜率</th><th>中位超额</th><th>p值</th><th>显著性</th>
        </tr></thead><tbody>{"".join(rows)}</tbody></table>'''


def _build_condition_bar_data(cond_name, cond_results):
    datasets = {}
    for cat in CATEGORY_ORDER:
        labels = []
        wr_a = []
        for cond_val, cond_data in cond_results.items():
            d = cond_data.get(cat, {})
            if d.get('n', 0) > 0:
                labels.append(cond_val)
                wr_a.append(d.get('win_rate', 0))
        datasets[cat] = {'labels': labels, 'wr_a': wr_a, 'title': CATEGORY_LABELS[cat]}
    return datasets


def _build_convergence_data(conv):
    sample_sizes = []
    win_rates = []
    p_values = []
    median_excess = []
    for n_str in ['200', '500', '1000', '2000']:
        d = conv.get(n_str, {})
        if d.get('n', 0) > 0:
            sample_sizes.append(int(n_str))
            win_rates.append(d.get('win_rate', 0))
            p_values.append(d.get('p_value', 1))
            median_excess.append(d.get('median_excess', 0))
    return {'sample_sizes': sample_sizes, 'win_rates': win_rates,
            'p_values': p_values, 'median_excess': median_excess}


def _build_ai_summary(all_results):
    lines = []
    base = all_results.get('base', {})
    long_d = base.get('long', {})
    wr = long_d.get('win_rate', 0)
    me = long_d.get('median_excess', 0)
    pv = long_d.get('p_value', 1)
    sig = _sig(pv)

    if wr > 55 and pv < 0.05:
        lines.append(f'<li><strong>基础结论</strong>：在默认条件下，卖1.2%买0.5%(A)长期胜率{wr:.1f}%，中位超额{_fmt(me)}%，p={pv:.4f}{sig}，<strong>A显著优于B</strong>。</li>')
    elif wr < 45 and pv < 0.05:
        lines.append(f'<li><strong>基础结论</strong>：在默认条件下，卖1.75%买0.75%(B)长期胜率{100-wr:.1f}%，<strong>B显著优于A</strong>。</li>')
    else:
        lines.append(f'<li><strong>基础结论</strong>：在默认条件下，A长期胜率{wr:.1f}%，p={pv:.4f}，两者差异<strong>不显著</strong>。</li>')

    fee = all_results.get('fee', {})
    for fee_name, fee_data in fee.items():
        fd = fee_data.get('long', {})
        fwr = fd.get('win_rate', 0)
        if fee_name == '高佣金(万3)' and fwr < 50:
            lines.append(f'<li><strong>手续费敏感</strong>：高佣金环境下A胜率仅{fwr:.1f}%，极密网格(1.2/0.5)因交易频繁受手续费影响更大，<strong>B更抗手续费</strong>。</li>')
            break

    cap = all_results.get('capital', {})
    small_cap_d = cap.get('10000', {}).get('long', {})
    large_cap_d = cap.get('500000', {}).get('long', {})
    if small_cap_d.get('n', 0) > 0 and large_cap_d.get('n', 0) > 0:
        swr = small_cap_d.get('win_rate', 0)
        lwr = large_cap_d.get('win_rate', 0)
        if swr < lwr:
            lines.append(f'<li><strong>资金效应</strong>：小资金(1万)A胜率{swr:.1f}%，大资金(50万)A胜率{lwr:.1f}%，<strong>资金越大A越占优</strong>（手续费占比降低）。</li>')

    reb = all_results.get('rebalance', {})
    no_reb = reb.get('无', {}).get('long', {})
    monthly = reb.get('月度', {}).get('long', {})
    if no_reb.get('n', 0) > 0 and monthly.get('n', 0) > 0:
        nwr = no_reb.get('win_rate', 0)
        mwr = monthly.get('win_rate', 0)
        if nwr < mwr:
            lines.append(f'<li><strong>再平准加成</strong>：无再平准时A胜率{nwr:.1f}%，月度再平准时A胜率{mwr:.1f}%，<strong>再平准对A的增益更大</strong>。</li>')

    conv = all_results.get('convergence', {})
    conv_1000 = conv.get('1000', {})
    conv_2000 = conv.get('2000', {})
    if conv_1000.get('n', 0) > 0 and conv_2000.get('n', 0) > 0:
        wr1 = conv_1000.get('win_rate', 0)
        wr2 = conv_2000.get('win_rate', 0)
        if abs(wr1 - wr2) < 3:
            lines.append(f'<li><strong>收敛性</strong>：1000样本胜率{wr1:.1f}%，2000样本胜率{wr2:.1f}%，差异<3%，<strong>结果已收敛</strong>。</li>')

    return ''.join(lines)


def generate_vs_report(all_results, n_samples=1000, output_path=None):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "research", "vs_comparison_report.html")

    base = all_results.get('base', {})
    base_table = _build_base_table(base)
    boxplot_data = _build_boxplot_data(base)
    conv_data = _build_convergence_data(all_results.get('convergence', {}))
    ai_summary = _build_ai_summary(all_results)

    cond_tables = ''
    cond_bar_jsons = {}
    cond_dims = [
        ('资金维度', 'capital'), ('调仓比例', 'trade_ratio'),
        ('底仓比例', 'min_position'), ('再平准频率', 'rebalance'),
        ('市场环境', 'market_env'), ('标的数量', 'bank_count'),
        ('手续费档位', 'fee'),
    ]
    for dim_name, dim_key in cond_dims:
        dim_data = all_results.get(dim_key, {})
        if dim_data:
            cond_tables += _build_condition_table(dim_name, dim_data)
            cond_bar_jsons[dim_key] = _build_condition_bar_data(dim_name, dim_data)

    boxplot_json = json.dumps(boxplot_data, ensure_ascii=False)
    conv_json = json.dumps(conv_data, ensure_ascii=False)
    cond_bar_json = json.dumps(cond_bar_jsons, ensure_ascii=False)

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>卖1.2%买0.5% vs 卖1.75%买0.75% 多条件对比报告</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#1a1a2e; color:#e0e0e0; padding:20px; }}
.hd {{ text-align:center; padding:30px 0; border-bottom:2px solid #3498db; margin-bottom:30px; }}
.hd h1 {{ color:#f1c40f; font-size:2em; margin-bottom:10px; }}
.hd .meta {{ display:flex; justify-content:center; gap:20px; color:#8899aa; font-size:0.9em; flex-wrap:wrap; }}
.card {{ background:#16213e; border-radius:8px; padding:24px; margin-bottom:24px; box-shadow:0 4px 6px rgba(0,0,0,0.3); }}
.card h2 {{ color:#3498db; margin-bottom:16px; border-bottom:1px solid #2c3e50; padding-bottom:8px; }}
.card h3 {{ color:#f1c40f; margin:16px 0 8px; }}
.dt {{ width:100%; border-collapse:collapse; margin:12px 0; font-size:0.85em; }}
.dt th {{ background:#0f3460; color:#f1c40f; padding:8px 10px; text-align:center; white-space:nowrap; }}
.dt td {{ padding:6px 10px; text-align:center; border-bottom:1px solid #2c3e50; }}
.dt tr:hover {{ background:rgba(52,152,219,0.1); }}
.cc {{ width:100%; height:400px; margin:16px 0; }}
.cr {{ display:flex; flex-wrap:wrap; gap:16px; }}
.cr .ci {{ flex:1; min-width:400px; }}
.sm {{ border:2px solid #f1c40f; }}
.sm ul {{ padding-left:24px; line-height:2; }}
.sm li {{ margin-bottom:8px; }}
.nt {{ color:#8899aa; font-style:italic; margin-top:8px; }}
.tag-a {{ color:#2ecc71; font-weight:bold; }}
.tag-b {{ color:#e74c3c; font-weight:bold; }}
@media (max-width:768px) {{
    .hd h1 {{ font-size:1.5em; }}
    th,td {{ padding:4px 6px; font-size:0.75em; }}
    .cr .ci {{ min-width:100%; }}
}}
</style>
</head>
<body>
<div class="hd">
    <h1>卖1.2%买0.5% vs 卖1.75%买0.75% 多条件对比报告</h1>
    <div class="meta">
        <span>蒙特卡洛随机回测</span>
        <span>每条件{n_samples}样本×2策略配对</span>
        <span>手续费:万0.854(0.5起)+印花税万5</span>
        <span>底仓50%|调仓30%|月度再平准</span>
    </div>
</div>

<div class="card">
    <h2>基础对比：默认条件</h2>
    <p class="nt"><span class="tag-a">A: 卖1.2%买0.5%</span> vs <span class="tag-b">B: 卖1.75%买0.75%</span>，胜率=A年化>B年化的比例</p>
    {base_table}
</div>

<div class="card">
    <h2>基础对比：年化收益箱线图</h2>
    <div class="cr">
        <div class="ci"><div id="bp-short" class="cc"></div></div>
        <div class="ci"><div id="bp-medium" class="cc"></div></div>
    </div>
    <div class="cr">
        <div class="ci"><div id="bp-medium_long" class="cc"></div></div>
        <div class="ci"><div id="bp-long" class="cc"></div></div>
    </div>
</div>

<div class="card">
    <h2>多条件对比：各维度详细数据</h2>
    {cond_tables}
</div>

<div class="card">
    <h2>多条件对比：A胜率柱状图（长期）</h2>
    <div class="cr">
        <div class="ci"><div id="bar-capital" class="cc"></div></div>
        <div class="ci"><div id="bar-trade_ratio" class="cc"></div></div>
    </div>
    <div class="cr">
        <div class="ci"><div id="bar-min_position" class="cc"></div></div>
        <div class="ci"><div id="bar-rebalance" class="cc"></div></div>
    </div>
    <div class="cr">
        <div class="ci"><div id="bar-market_env" class="cc"></div></div>
        <div class="ci"><div id="bar-bank_count" class="cc"></div></div>
    </div>
    <div class="cr">
        <div class="ci"><div id="bar-fee" class="cc"></div></div>
    </div>
</div>

<div class="card">
    <h2>收敛性验证（中长期）</h2>
    <div class="cr">
        <div class="ci"><div id="conv-wr" class="cc"></div></div>
        <div class="ci"><div id="conv-pv" class="cc"></div></div>
    </div>
</div>

<div class="card sm">
    <h2>AI 智能总结</h2>
    <ul>{ai_summary}</ul>
</div>

<script>
var bpData = {boxplot_json};
var convData = {conv_json};
var condBar = {cond_bar_json};

Object.keys(bpData).forEach(function(cat) {{
    var d = bpData[cat];
    var ch = echarts.init(document.getElementById('bp-' + cat));
    ch.setOption({{
        title: {{ text: d.title + ' - A vs B 年化收益', left:'center', textStyle:{{ color:'#f1c40f', fontSize:14 }} }},
        tooltip: {{ trigger:'item' }},
        xAxis: {{ type:'category', data:d.labels, axisLabel:{{ color:'#aaa', fontSize:10 }} }},
        yAxis: {{ type:'value', name:'年化%', axisLabel:{{ color:'#aaa' }}, splitLine:{{ lineStyle:{{ color:'#2c3e50' }} }} }},
        series: [
            {{ name:'箱线', type:'boxplot', data:d.box_data, itemStyle:{{ color:'rgba(52,152,219,0.6)', borderColor:'#3498db' }} }},
            {{ name:'异常值', type:'scatter', data:d.outlier_data, itemStyle:{{ color:'#e74c3c' }} }}
        ]
    }});
}});

var convCh1 = echarts.init(document.getElementById('conv-wr'));
convCh1.setOption({{
    title: {{ text:'A胜率随样本量收敛', left:'center', textStyle:{{ color:'#f1c40f', fontSize:14 }} }},
    tooltip: {{ trigger:'axis' }},
    xAxis: {{ type:'category', data:convData.sample_sizes.map(String), name:'样本量', axisLabel:{{ color:'#aaa' }}, nameTextStyle:{{ color:'#aaa' }} }},
    yAxis: {{ type:'value', name:'胜率%', min:40, max:70, axisLabel:{{ color:'#aaa' }}, splitLine:{{ lineStyle:{{ color:'#2c3e50' }} }} }},
    series: [
        {{ name:'A胜率', type:'line', data:convData.win_rates, smooth:true,
           lineStyle:{{ color:'#2ecc71', width:3 }}, itemStyle:{{ color:'#2ecc71' }},
           markLine:{{ data:[{{yAxis:50, lineStyle:{{color:'#e74c3c',type:'dashed'}}, label:{{formatter:'50%基准'}}}}] }}
        }}
    ]
}});

var convCh2 = echarts.init(document.getElementById('conv-pv'));
convCh2.setOption({{
    title: {{ text:'p值随样本量收敛', left:'center', textStyle:{{ color:'#f1c40f', fontSize:14 }} }},
    tooltip: {{ trigger:'axis' }},
    xAxis: {{ type:'category', data:convData.sample_sizes.map(String), name:'样本量', axisLabel:{{ color:'#aaa' }}, nameTextStyle:{{ color:'#aaa' }} }},
    yAxis: {{ type:'value', name:'p值', min:0, max:1, axisLabel:{{ color:'#aaa' }}, splitLine:{{ lineStyle:{{ color:'#2c3e50' }} }} }},
    series: [
        {{ name:'p值', type:'line', data:convData.p_values, smooth:true,
           lineStyle:{{ color:'#3498db', width:3 }}, itemStyle:{{ color:'#3498db' }},
           markLine:{{ data:[{{yAxis:0.05, lineStyle:{{color:'#e74c3c',type:'dashed'}}, label:{{formatter:'p=0.05'}}}}] }}
        }}
    ]
}});

Object.keys(condBar).forEach(function(dimKey) {{
    var el = document.getElementById('bar-' + dimKey);
    if (!el) return;
    var d = condBar[dimKey].long || condBar[dimKey][Object.keys(condBar[dimKey])[0]];
    if (!d || !d.labels) return;
    var ch = echarts.init(el);
    ch.setOption({{
        title: {{ text: dimKey + ' - 长期A胜率', left:'center', textStyle:{{ color:'#f1c40f', fontSize:14 }} }},
        tooltip: {{ trigger:'axis', formatter:function(p){{ return p[0].name+': '+p[0].value.toFixed(1)+'%'; }} }},
        xAxis: {{ type:'category', data:d.labels, axisLabel:{{ color:'#aaa', rotate:20 }} }},
        yAxis: {{ type:'value', name:'A胜率%', min:30, max:80, axisLabel:{{ color:'#aaa' }}, splitLine:{{ lineStyle:{{ color:'#2c3e50' }} }} }},
        series: [
            {{ name:'A胜率', type:'bar', data:d.wr_a.map(function(v){{ return {{ value:v, itemStyle:{{ color: v>=55?'#2ecc71':v>=50?'#f1c40f':'#e74c3c' }} }}; }}),
               markLine:{{ data:[{{yAxis:50, lineStyle:{{color:'#fff',type:'dashed'}}, label:{{formatter:'50%'}}}}] }}
            }}
        ]
    }});
}});
</script>
</body>
</html>'''

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return output_path
