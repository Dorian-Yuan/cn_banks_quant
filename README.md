# 波段再平准策略量化回测

## 项目说明

这是一个用于回测银行股波段再平准策略的可视化系统。

## 访问地址

**GitHub Pages**：[https://dorian-yuan.github.io/cn_banks_quant/reports/grid_rebalance_backtest.html](https://dorian-yuan.github.io/cn_banks_quant/reports/grid_rebalance_backtest.html)

## 功能特性

- 支持6只银行股（工行、农行、中行、建行、交行、邮储）
- 网格交易策略 + 定期再平准
- 红利税计算（先进先出，差别化税率）
- 可自定义参数：时间范围、初始资金、网格百分比、调仓比例、佣金设置
- 实时可视化：净值曲线、持仓分布、交易日志
- 与买入持有策略对比

## 数据说明

- 数据来源：AkShare
- 数据格式：前复权日K线
- 数据文件：`data/ashare/*.csv`

## 技术栈

- 前端：纯HTML + JavaScript
- 图表：ECharts
- CSV解析：PapaParse

## 策略规则

1. **开仓**：等分资金买入，每只买入够钱的100股最大倍数
2. **网格**：日内触及基准价±网格%，买卖持仓×调仓比例
3. **除息**：分红日手动将基准价下调等额分红
4. **再平准**：周期末等权归齐各股权重，重置基准价为当日收盘价

## 运行本地服务器

```bash
python -m http.server 8080
# 访问 http://localhost:8080/reports/grid_rebalance_backtest.html
```
