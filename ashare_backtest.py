import os
import sys
import io
import pandas as pd
import backtrader as bt
from datetime import datetime
from collections import deque

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ashare")

STOCKS = {
    "601328": "交通银行",
    "601939": "建设银行",
    "601398": "工商银行",
    "601288": "农业银行",
    "601988": "中国银行",
    "600036": "招商银行",
    "000001": "平安银行",
}

INITIAL_CASH = 30000
START_DATE = "2021-04-13"
END_DATE = "2026-04-13"


class DividendYieldData(bt.feeds.PandasData):
    lines = ('dividend_yield_ttm',)
    params = (
        ('dividend_yield_ttm', 'DividendYieldTTM'),
        ('openinterest', -1),
    )


class DividendYieldStrategy(bt.Strategy):
    params = (
        ('printlog', False),
    )

    def __init__(self):
        self.order = None
        self.total_buy_shares = 0
        self.total_sell_shares = 0
        self.total_buy_amount = 0.0
        self.total_sell_amount = 0.0
        self.max_position = 0
        self.buy_queue = deque()
        self.sell_profit_count = 0
        self.sell_loss_count = 0
        self.sell_total_count = 0

    def log(self, txt, dt=None):
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'[{dt}] {txt}')

    def next(self):
        if self.order:
            return

        current_dy = self.datas[0].dividend_yield_ttm[0]
        current_close = self.datas[0].close[0]
        current_pos = self.getposition(self.datas[0]).size
        current_date = self.datas[0].datetime.date(0)

        if current_date < pd.Timestamp(START_DATE).date():
            return

        cash = self.broker.get_cash()

        buy_pct = 0
        if current_dy >= 5.0:
            buy_pct = 0.10
        elif current_dy >= 4.9:
            buy_pct = 0.08
        elif current_dy >= 4.8:
            buy_pct = 0.06
        elif current_dy >= 4.7:
            buy_pct = 0.05

        sell_pct = 0
        if current_dy < 4.0:
            sell_pct = 0.10
        elif current_dy < 4.1:
            sell_pct = 0.08
        elif current_dy < 4.2:
            sell_pct = 0.06
        elif current_dy < 4.3:
            sell_pct = 0.05

        if buy_pct > 0:
            buy_amount = cash * buy_pct
            buy_size = int(buy_amount / current_close / 100) * 100
            if buy_size < 100 and cash >= current_close * 100:
                buy_size = 100
            if buy_size >= 100:
                self.order = self.buy(size=buy_size)
                self.total_buy_shares += buy_size
                self.total_buy_amount += buy_size * current_close
                self.buy_queue.append({'price': current_close, 'size': buy_size})

        elif sell_pct > 0 and current_pos > 0:
            sell_size = int(current_pos * sell_pct / 100) * 100
            sell_size = (sell_size // 100) * 100
            if sell_size < 100 and current_pos >= 100:
                sell_size = 100
            if sell_size > current_pos:
                sell_size = (current_pos // 100) * 100
            if current_pos > 0 and sell_size < 100:
                sell_size = current_pos
            if sell_size > 0:
                self.order = self.sell(size=sell_size)
                self.total_sell_shares += sell_size
                self.total_sell_amount += sell_size * current_close

                remaining = sell_size
                while remaining > 0 and self.buy_queue:
                    batch = self.buy_queue[0]
                    if batch['size'] <= remaining:
                        self.buy_queue.popleft()
                        pnl = (current_close - batch['price']) * batch['size']
                        self.sell_total_count += 1
                        if pnl > 0:
                            self.sell_profit_count += 1
                        else:
                            self.sell_loss_count += 1
                        remaining -= batch['size']
                    else:
                        pnl = (current_close - batch['price']) * remaining
                        self.sell_total_count += 1
                        if pnl > 0:
                            self.sell_profit_count += 1
                        else:
                            self.sell_loss_count += 1
                        batch['size'] -= remaining
                        remaining = 0

        pos = self.getposition(self.datas[0]).size
        if pos > self.max_position:
            self.max_position = pos

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入: 价格={order.executed.price:.2f}, '
                         f'数量={order.executed.size}, 成本={order.executed.value:.2f}, '
                         f'佣金={order.executed.comm:.4f}')
            elif order.issell():
                self.log(f'卖出: 价格={order.executed.price:.2f}, '
                         f'数量={order.executed.size}, 成本={order.executed.value:.2f}, '
                         f'佣金={order.executed.comm:.4f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'订单失败: {order.status}')

        self.order = None


class NoMinCommission(bt.CommInfoBase):
    params = (
        ('commission', 0.000086),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * price * self.p.commission


def load_data(symbol, start_date=START_DATE, end_date=END_DATE):
    csv_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件不存在: {csv_path}，请先运行 ashare_data_fetcher.py")

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.loc[start_date:end_date]

    if df.empty:
        raise ValueError(f"股票 {symbol} 在 {start_date} ~ {end_date} 期间无数据")

    return DividendYieldData(dataname=df)


def run_backtest(symbol, name):
    print(f"\n{'='*60}")
    print(f"回测: {symbol} {name}")
    print(f"{'='*60}")

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.addcommissioninfo(NoMinCommission())

    data = load_data(symbol)
    cerebro.adddata(data, name=f"{symbol}_{name}")
    cerebro.addstrategy(DividendYieldStrategy, printlog=False)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    cash = cerebro.broker.get_cash()
    position = strat.getposition(strat.datas[0]).size
    close_price = strat.datas[0].close[0]
    position_value = position * close_price

    drawdown = strat.analyzers.getbyname('drawdown')

    max_dd = 0
    if hasattr(drawdown, 'get_analysis'):
        dd = drawdown.get_analysis()
        max_dd = dd.get('max', {}).get('drawdown', 0)

    sell_total = strat.sell_total_count
    sell_profit = strat.sell_profit_count
    sell_loss = strat.sell_loss_count
    win_rate = (sell_profit / sell_total * 100) if sell_total > 0 else 0

    actual_profit = final_value - INITIAL_CASH
    actual_return = actual_profit / INITIAL_CASH * 100

    result = {
        'symbol': symbol,
        'name': name,
        'initial_cash': INITIAL_CASH,
        'final_value': final_value,
        'cash': cash,
        'position': position,
        'close_price': close_price,
        'position_value': position_value,
        'actual_profit': actual_profit,
        'actual_return': actual_return,
        'total_buy_shares': strat.total_buy_shares,
        'total_sell_shares': strat.total_sell_shares,
        'total_buy_amount': strat.total_buy_amount,
        'total_sell_amount': strat.total_sell_amount,
        'max_position': strat.max_position,
        'sell_total': sell_total,
        'sell_profit': sell_profit,
        'sell_loss': sell_loss,
        'win_rate': win_rate,
        'max_drawdown': max_dd,
    }

    print(f"\n--- 回测结果 ---")
    print(f"初始资金: {INITIAL_CASH:,.0f}元")
    print(f"最终总资产: {final_value:,.2f}元")
    print(f"  现金: {cash:,.2f}元")
    print(f"  持仓: {position:,}股 x {close_price:.2f}元 = {position_value:,.2f}元")
    print(f"实际收益: {actual_profit:,.2f}元")
    print(f"收益率: {actual_return:.2f}%")
    print(f"累计买入: {strat.total_buy_shares:,}股, 金额: {strat.total_buy_amount:,.2f}元")
    print(f"累计卖出: {strat.total_sell_shares:,}股, 金额: {strat.total_sell_amount:,.2f}元")
    print(f"最大持仓: {strat.max_position:,}股")
    print(f"卖出交易: {sell_total}笔 (盈利{sell_profit}笔, 亏损{sell_loss}笔)")
    print(f"胜率: {win_rate:.1f}%")
    print(f"最大回撤: {max_dd:.2f}%")

    return result


def main():
    print("=" * 80)
    print("A股银行股自适应股息率策略回测 (3万本金)")
    print("=" * 80)
    print(f"买入规则: 股息率TTM 4.7~4.8%用5%现金买, 4.8~4.9%用6%现金买, 4.9~5.0%用8%现金买, >=5.0%用10%现金买")
    print(f"卖出规则: 股息率TTM 4.2~4.3%卖5%持仓, 4.1~4.2%卖6%持仓, 4.0~4.1%卖8%持仓, <4.0%卖10%持仓")
    print(f"观望区间: 4.3%~4.7%")
    print(f"手续费: 万0.86, 免5")
    print(f"初始资金: {INITIAL_CASH:,.0f}元/只")
    print(f"回测区间: {START_DATE} ~ {END_DATE}")

    all_results = []
    for symbol, name in STOCKS.items():
        try:
            result = run_backtest(symbol, name)
            all_results.append(result)
        except Exception as e:
            print(f"\n[{symbol} {name}] 回测失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 120)
    print("汇总对比")
    print("=" * 120)

    header = (f"{'股票':<8} {'代码':<8} {'初始资金':>10} {'最终资产':>12} "
              f"{'实际收益':>10} {'收益率':>8} {'买入股数':>8} {'卖出股数':>8} "
              f"{'最终持仓':>8} {'最大持仓':>8} {'胜率':>8} {'最大回撤':>8}")
    print(header)
    print("-" * 120)

    for r in all_results:
        row = (f"{r['name']:<8} {r['symbol']:<8} {r['initial_cash']:>10,.0f} "
               f"{r['final_value']:>12,.2f} {r['actual_profit']:>10,.2f} "
               f"{r['actual_return']:>7.2f}% {r['total_buy_shares']:>8,} "
               f"{r['total_sell_shares']:>8,} {r['position']:>8,} "
               f"{r['max_position']:>8,} {r['win_rate']:>7.1f}% "
               f"{r['max_drawdown']:>7.2f}%")
        print(row)

    print("\n" + "=" * 120)
    print("详细分析")
    print("=" * 120)
    for r in all_results:
        print(f"\n{r['name']}({r['symbol']}):")
        print(f"  初始资金: {r['initial_cash']:,.0f}元")
        print(f"  最终资产: {r['final_value']:,.2f}元 (现金: {r['cash']:,.2f} + 持仓市值: {r['position_value']:,.2f})")
        print(f"  实际收益: {r['actual_profit']:,.2f}元, 收益率: {r['actual_return']:.2f}%")
        print(f"  累计买入: {r['total_buy_shares']:,}股, 金额: {r['total_buy_amount']:,.2f}元")
        print(f"  累计卖出: {r['total_sell_shares']:,}股, 金额: {r['total_sell_amount']:,.2f}元")
        print(f"  最终持仓: {r['position']:,}股, 最大持仓: {r['max_position']:,}股")
        print(f"  卖出交易: {r['sell_total']}笔 (盈利{r['sell_profit']}笔, 亏损{r['sell_loss']}笔), 胜率: {r['win_rate']:.1f}%")
        print(f"  最大回撤: {r['max_drawdown']:.2f}%")

    total_profit = sum(r['actual_profit'] for r in all_results)
    total_initial = sum(r['initial_cash'] for r in all_results)
    total_sell = sum(r['sell_total'] for r in all_results)
    total_sell_profit = sum(r['sell_profit'] for r in all_results)
    overall_win_rate = (total_sell_profit / total_sell * 100) if total_sell > 0 else 0
    print(f"\n{'='*80}")
    print(f"7只银行股合计:")
    print(f"  总初始资金: {total_initial:,.0f}元")
    print(f"  总实际收益: {total_profit:,.2f}元")
    print(f"  总收益率: {total_profit/total_initial*100:.2f}%")
    print(f"  总卖出交易: {total_sell}笔, 总体胜率: {overall_win_rate:.1f}%")


if __name__ == "__main__":
    main()
