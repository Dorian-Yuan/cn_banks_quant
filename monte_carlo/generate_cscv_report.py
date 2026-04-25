import json
import os
import numpy as np


def _pbo_color(pbo):
    if pbo < 0.3:
        return '#2ecc71'
    if pbo < 0.5:
        return '#f1c40f'
    return '#e74c3c'


def _pbo_label(pbo):
    if pbo < 0.3:
        return '低过拟合风险 - 策略鲁棒'
    if pbo < 0.5:
        return '中等风险 - 需谨慎'
    return '高过拟合风险 - 策略可能不稳健'


def _build_ai_summary(result):
    pbo = result.get('pbo', 0)
    lines = []

    if pbo < 0.3:
        lines.append('<li><strong>过拟合风险：低</strong>。PBO=%.1f%%，说明样本内最优策略在样本外大概率仍保持领先，策略参数选择不存在严重过拟合。</li>' % (pbo * 100))
    elif pbo < 0.5:
        lines.append('<li><strong>过拟合风险：中等</strong>。PBO=%.1f%%，样本内最优策略在样本外有一定概率排名下滑，建议关注参数鲁棒性。</li>' % (pbo * 100))
    else:
        lines.append('<li><strong>过拟合风险：高</strong>。PBO=%.1f%%，样本内最优策略在样本外大概率排名下滑，策略参数可能过拟合了历史数据。</li>' % (pbo * 100))

    bt = result.get('base_tracking', {})
    if bt:
        is_med = bt.get('is_median_rank', 0)
        oos_med = bt.get('oos_median_rank', 0)
        M = result.get('M', 56)
        if oos_med <= M * 0.25:
            lines.append('<li><strong>基准参数(卖1.2%%买0.5%%)鲁棒性：强</strong>。OOS中位排名%.1f/%d（前%.0f%%），在样本外仍保持较好表现。</li>' % (oos_med, M, oos_med / M * 100))
        elif oos_med <= M * 0.5:
            lines.append('<li><strong>基准参数鲁棒性：中等</strong>。OOS中位排名%.1f/%d，在样本外表现中等。</li>' % (oos_med, M))
        else:
            lines.append('<li><strong>基准参数鲁棒性：弱</strong>。OOS中位排名%.1f/%d，在样本外表现不佳，可能过拟合。</li>' % (oos_med, M))

        if is_med < oos_med:
            lines.append('<li>IS→OOS排名衰减：IS中位%.1f → OOS中位%.1f，衰减%.1f位。衰减越小策略越鲁棒。</li>' % (is_med, oos_med, oos_med - is_med))

    top_params = result.get('top_is_params', [])
    if top_params:
        top3 = top_params[:3]
        top_str = '、'.join(['%s(%d次)' % (n, c) for n, c in top3])
        lines.append('<li><strong>IS最频繁最优参数</strong>：%s。如果这些参数在OOS中也排名靠前，说明策略整体鲁棒。</li>' % top_str)

    return ''.join(lines)


def generate_cscv_report(result, output_path=None):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "research", "cscv_report.html")

    pbo = result.get('pbo', 0)
    M = result.get('M', 56)
    S = result.get('S', 16)
    n_combos = result.get('n_combos', 0)
    base_label = result.get('base_label', '卖1.2%买0.5%')
    dates_range = result.get('dates_range', '')
    trading_days = result.get('trading_days', 0)

    logit_ranks = result.get('logit_ranks', [])
    relative_ranks = result.get('relative_ranks', [])
    is_best_sharpes = result.get('is_best_sharpes', [])
    oos_sharpes_of_best = result.get('oos_sharpes_of_best', [])

    bt = result.get('base_tracking', {})
    top_params = result.get('top_is_params', [])

    n_bins = 30
    logit_arr = np.array(logit_ranks)
    logit_arr = logit_arr[np.isfinite(logit_arr)]
    hist_counts, bin_edges = np.histogram(logit_arr, bins=n_bins)
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)]
    hist_list = hist_counts.tolist()

    rank_arr = np.array(relative_ranks)
    rank_arr = rank_arr[np.isfinite(rank_arr)]
    rank_hist, rank_bins = np.histogram(rank_arr, bins=20, range=(0, 1))
    rank_centers = [(rank_bins[i] + rank_bins[i + 1]) / 2 for i in range(20)]
    rank_hist_list = rank_hist.tolist()

    scatter_is = is_best_sharpes[:2000] if len(is_best_sharpes) > 2000 else is_best_sharpes
    scatter_oos = oos_sharpes_of_best[:2000] if len(oos_sharpes_of_best) > 2000 else oos_sharpes_of_best

    base_is_dist = bt.get('is_rank_dist', [])[:2000] if bt else []
    base_oos_dist = bt.get('oos_rank_dist', [])[:2000] if bt else []

    ai_summary = _build_ai_summary(result)

    top_params_rows = ''
    for i, (name, count) in enumerate(top_params[:10]):
        pct = count / n_combos * 100 if n_combos > 0 else 0
        is_base = ' ★' if name == base_label else ''
        top_params_rows += '<tr><td>#%d</td><td>%s%s</td><td>%d</td><td>%.1f%%</td></tr>' % (i + 1, name, is_base, count, pct)

    base_tracking_html = ''
    if bt:
        base_tracking_html = '''
        <h3>基准参数鲁棒性追踪</h3>
        <table class="dt">
            <thead><tr><th>指标</th><th>样本内(IS)</th><th>样本外(OOS)</th></tr></thead>
            <tbody>
                <tr><td>平均排名</td><td>%.1f/%d</td><td>%.1f/%d</td></tr>
                <tr><td>中位排名</td><td>%.1f</td><td>%.1f</td></tr>
                <tr><td>Top1概率</td><td>%.1f%%</td><td>-</td></tr>
                <tr><td>Top5概率</td><td>-</td><td>%.1f%%</td></tr>
            </tbody>
        </table>''' % (
            bt.get('is_avg_rank', 0), M, bt.get('oos_avg_rank', 0), M,
            bt.get('is_median_rank', 0), bt.get('oos_median_rank', 0),
            bt.get('is_top1_pct', 0), bt.get('oos_top5_pct', 0))

    chart_data = json.dumps({
        'logit': {'centers': bin_centers, 'counts': hist_list},
        'rank': {'centers': rank_centers, 'counts': rank_hist_list},
        'scatter': {'is': scatter_is, 'oos': scatter_oos},
        'base_is_dist': base_is_dist,
        'base_oos_dist': base_oos_dist,
        'pbo': pbo,
        'M': M,
    }, ensure_ascii=False)

    pbo_pct = '%.1f' % (pbo * 100)
    pbo_col = _pbo_color(pbo)
    pbo_lbl = _pbo_label(pbo)

    html_parts = []
    html_parts.append('''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CSCV 过拟合测试报告</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#1a1a2e;color:#e0e0e0;padding:20px}
.hd{text-align:center;padding:30px 0;border-bottom:2px solid #3498db;margin-bottom:30px}
.hd h1{color:#f1c40f;font-size:2em;margin-bottom:10px}
.hd .meta{display:flex;justify-content:center;gap:20px;color:#8899aa;font-size:0.9em;flex-wrap:wrap}
.card{background:#16213e;border-radius:8px;padding:24px;margin-bottom:24px;box-shadow:0 4px 6px rgba(0,0,0,0.3)}
.card h2{color:#3498db;margin-bottom:16px;border-bottom:1px solid #2c3e50;padding-bottom:8px}
.card h3{color:#f1c40f;margin:16px 0 8px}
.dt{width:100%;border-collapse:collapse;margin:12px 0;font-size:0.85em}
.dt th{background:#0f3460;color:#f1c40f;padding:8px 10px;text-align:center;white-space:nowrap}
.dt td{padding:6px 10px;text-align:center;border-bottom:1px solid #2c3e50}
.dt tr:hover{background:rgba(52,152,219,0.1)}
.cc{width:100%;height:400px;margin:16px 0}
.pbo-big{text-align:center;padding:30px;margin:20px 0;border-radius:12px}
.pbo-big .num{font-size:5em;font-weight:bold}
.pbo-big .label{font-size:1.2em;margin-top:8px}
.sm{border:2px solid #f1c40f}
.sm ul{padding-left:24px;line-height:2}
.sm li{margin-bottom:8px}
.nt{color:#8899aa;font-style:italic;margin-top:8px}
.cr{display:flex;flex-wrap:wrap;gap:16px}
.cr .ci{flex:1;min-width:400px}
@media(max-width:768px){.hd h1{font-size:1.5em}th,td{padding:4px 6px;font-size:0.75em}.cr .ci{min-width:100%}}
</style>
</head>
<body>
<div class="hd">
    <h1>CSCV 过拟合测试报告</h1>
    <div class="meta">
        <span>Combinatorially Symmetric Cross-Validation</span>
        <span>''' + str(M) + '组参数 × ' + str(S) + '区块 × ' + str(n_combos) + '''种组合</span>
        <span>''' + dates_range + ' (' + str(trading_days) + '''交易日)</span>
        <span>基准: ''' + base_label + '''</span>
    </div>
</div>

<div class="card">
    <h2>PBO (Probability of Backtest Overfitting)</h2>
    <div class="pbo-big" style="background:''' + pbo_col + '''22">
        <div class="num" style="color:''' + pbo_col + '''">''' + pbo_pct + '''%</div>
        <div class="label" style="color:''' + pbo_col + '''">''' + pbo_lbl + '''</div>
    </div>
    <p class="nt">PBO = IS最优策略在OOS排名跌至后50%的概率。越低越好，&lt;30%为低风险。</p>
</div>

<div class="card">
    <h2>Logit排名分布图</h2>
    <div id="logit-chart" class="cc"></div>
    <p class="nt">横轴为logit(相对排名)，0对应排名中位数。左侧(负值)表示OOS表现好，右侧(正值)表示OOS表现差。</p>
</div>

<div class="card">
    <h2>OOS相对排名分布</h2>
    <div id="rank-chart" class="cc"></div>
    <p class="nt">横轴为OOS相对排名(0=最好,1=最差)，红色区域(>0.5)为过拟合区。</p>
</div>

<div class="card">
    <h2>IS最优策略的IS vs OOS夏普比率</h2>
    <div id="scatter-chart" class="cc"></div>
    <p class="nt">每个点代表一种IS/OOS组合。若点集中在左下方，说明IS最优在OOS表现也佳。</p>
</div>

<div class="card">
    <h2>基准参数鲁棒性</h2>
    ''' + base_tracking_html + '''
    <div class="cr">
        <div class="ci"><div id="base-is-chart" class="cc"></div></div>
        <div class="ci"><div id="base-oos-chart" class="cc"></div></div>
    </div>
</div>

<div class="card">
    <h2>IS最频繁最优参数 Top10</h2>
    <table class="dt">
        <thead><tr><th>排名</th><th>参数</th><th>IS最优次数</th><th>占比</th></tr></thead>
        <tbody>''' + top_params_rows + '''</tbody>
    </table>
    <p class="nt">★ = 基准参数</p>
</div>

<div class="card sm">
    <h2>AI 智能总结</h2>
    <ul>''' + ai_summary + '''</ul>
</div>

<script>
var data = ''' + chart_data + ''';

var logitCh = echarts.init(document.getElementById('logit-chart'));
logitCh.setOption({
    title:{text:'Logit(Rank)分布',left:'center',textStyle:{color:'#f1c40f',fontSize:14}},
    tooltip:{trigger:'axis'},
    xAxis:{type:'category',data:data.logit.centers.map(function(v){return v.toFixed(2)}),axisLabel:{color:'#aaa',rotate:45}},
    yAxis:{type:'value',name:'频率',axisLabel:{color:'#aaa'},splitLine:{lineStyle:{color:'#2c3e50'}}},
    series:[{type:'bar',data:data.logit.counts,
        itemStyle:{color:function(p){return p.dataIndex<data.logit.centers.length/2?'#2ecc71':'#e74c3c'}},
        markLine:{data:[{xAxis:Math.floor(data.logit.centers.length/2),lineStyle:{color:'#f1c40f',type:'dashed'},label:{formatter:'PBO分界'}}]}
    }]
});

var rankCh = echarts.init(document.getElementById('rank-chart'));
rankCh.setOption({
    title:{text:'OOS相对排名分布',left:'center',textStyle:{color:'#f1c40f',fontSize:14}},
    tooltip:{trigger:'axis'},
    xAxis:{type:'category',data:data.rank.centers.map(function(v){return v.toFixed(2)}),axisLabel:{color:'#aaa'}},
    yAxis:{type:'value',name:'频率',axisLabel:{color:'#aaa'},splitLine:{lineStyle:{color:'#2c3e50'}}},
    series:[{type:'bar',data:data.rank.counts,
        itemStyle:{color:function(p){return p.dataIndex>=10?'rgba(231,76,60,0.7)':'rgba(46,204,113,0.7)'}},
        markLine:{data:[{xAxis:10,lineStyle:{color:'#f1c40f',type:'dashed'},label:{formatter:'50%分位'}}]}
    }]
});

var scCh = echarts.init(document.getElementById('scatter-chart'));
scCh.setOption({
    title:{text:'IS最优策略: IS vs OOS夏普',left:'center',textStyle:{color:'#f1c40f',fontSize:14}},
    tooltip:{trigger:'item',formatter:function(p){return 'IS夏普:'+p.value[0].toFixed(3)+'<br>OOS夏普:'+p.value[1].toFixed(3)}},
    xAxis:{type:'value',name:'IS夏普',axisLabel:{color:'#aaa'},splitLine:{lineStyle:{color:'#2c3e50'}},nameTextStyle:{color:'#aaa'}},
    yAxis:{type:'value',name:'OOS夏普',axisLabel:{color:'#aaa'},splitLine:{lineStyle:{color:'#2c3e50'}},nameTextStyle:{color:'#aaa'}},
    series:[{type:'scatter',data:data.scatter.is.map(function(v,i){return [v,data.scatter.oos[i]]}),
        symbolSize:4,itemStyle:{color:'rgba(52,152,219,0.6)'},
        markLine:{data:[{type:'average',name:'OOS均值'}]}
    }]
});

if(data.base_is_dist.length>0){
    var bisCh=echarts.init(document.getElementById('base-is-chart'));
    var isHist={};
    data.base_is_dist.forEach(function(v){var k=Math.ceil(v);isHist[k]=(isHist[k]||0)+1});
    bisCh.setOption({
        title:{text:'基准参数IS排名分布',left:'center',textStyle:{color:'#f1c40f',fontSize:14}},
        tooltip:{trigger:'axis'},
        xAxis:{type:'category',data:Object.keys(isHist),name:'排名',axisLabel:{color:'#aaa'},nameTextStyle:{color:'#aaa'}},
        yAxis:{type:'value',name:'次数',axisLabel:{color:'#aaa'},splitLine:{lineStyle:{color:'#2c3e50'}}},
        series:[{type:'bar',data:Object.values(isHist),itemStyle:{color:'#2ecc71'}}]
    });
    var boosCh=echarts.init(document.getElementById('base-oos-chart'));
    var oosHist={};
    data.base_oos_dist.forEach(function(v){var k=Math.ceil(v);oosHist[k]=(oosHist[k]||0)+1});
    boosCh.setOption({
        title:{text:'基准参数OOS排名分布',left:'center',textStyle:{color:'#f1c40f',fontSize:14}},
        tooltip:{trigger:'axis'},
        xAxis:{type:'category',data:Object.keys(oosHist),name:'排名',axisLabel:{color:'#aaa'},nameTextStyle:{color:'#aaa'}},
        yAxis:{type:'value',name:'次数',axisLabel:{color:'#aaa'},splitLine:{lineStyle:{color:'#2c3e50'}}},
        series:[{type:'bar',data:Object.values(oosHist),itemStyle:{color:'#3498db'}}]
    });
}
</script>
</body>
</html>''')

    html = ''.join(html_parts)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return output_path
