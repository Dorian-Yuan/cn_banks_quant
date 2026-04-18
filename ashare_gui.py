import os
import json
import pandas as pd
import backtrader as bt
from collections import deque
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ashare")

STOCKS = [
    {"symbol": "601328", "name": "交通银行"},
    {"symbol": "601939", "name": "建设银行"},
    {"symbol": "601398", "name": "工商银行"},
    {"symbol": "601288", "name": "农业银行"},
    {"symbol": "601988", "name": "中国银行"},
    {"symbol": "600036", "name": "招商银行"},
    {"symbol": "000001", "name": "平安银行"},
]

STOCKS_MAP = {s["symbol"]: s["name"] for s in STOCKS}


def round_to_100(n):
    result = round(n / 100) * 100
    return result if result >= 100 else 0


class DividendYieldData(bt.feeds.PandasData):
    lines = ('dividend_yield_ttm',)
    params = (
        ('dividend_yield_ttm', 'DividendYieldTTM'),
        ('openinterest', -1),
    )


class DividendYieldStrategy(bt.Strategy):
    params = (
        ('mode', 'percentage'),
        ('buy_values', None),
        ('sell_values', None),
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
        self.daily_records = []

    def next(self):
        if self.order:
            return

        current_dy = self.datas[0].dividend_yield_ttm[0]
        current_close = self.datas[0].close[0]
        current_pos = self.getposition(self.datas[0]).size
        current_date = self.datas[0].datetime.date(0)

        cash = self.broker.get_cash()

        buy_val = 0
        if current_dy >= 5.0:
            buy_val = self.p.buy_values[3]
        elif current_dy >= 4.9:
            buy_val = self.p.buy_values[2]
        elif current_dy >= 4.8:
            buy_val = self.p.buy_values[1]
        elif current_dy >= 4.7:
            buy_val = self.p.buy_values[0]

        sell_val = 0
        if current_dy < 4.0:
            sell_val = self.p.sell_values[3]
        elif current_dy < 4.1:
            sell_val = self.p.sell_values[2]
        elif current_dy < 4.2:
            sell_val = self.p.sell_values[1]
        elif current_dy < 4.3:
            sell_val = self.p.sell_values[0]

        action = None
        action_shares = 0

        if buy_val > 0:
            if self.p.mode == 'percentage':
                buy_size = round_to_100(cash * buy_val / 100 / current_close)
            else:
                buy_size = round_to_100(buy_val)

            if buy_size >= 100 and cash >= buy_size * current_close:
                self.order = self.buy(size=buy_size)
                self.total_buy_shares += buy_size
                self.total_buy_amount += buy_size * current_close
                self.buy_queue.append({'price': current_close, 'size': buy_size})
                action = 'buy'
                action_shares = buy_size

        elif sell_val > 0 and current_pos > 0:
            if self.p.mode == 'percentage':
                sell_size = round_to_100(current_pos * sell_val / 100)
            else:
                sell_size = round_to_100(min(sell_val, current_pos))

            if sell_size > current_pos:
                sell_size = round_to_100(current_pos)
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

                action = 'sell'
                action_shares = sell_size

        pos = self.getposition(self.datas[0]).size
        if pos > self.max_position:
            self.max_position = pos

        self.daily_records.append({
            'date': current_date.isoformat(),
            'close': round(current_close, 2),
            'dy': round(current_dy, 2),
            'action': action,
            'shares': action_shares,
            'position': pos,
            'portfolio_value': round(self.broker.getvalue(), 2),
        })

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        self.order = None


class NoMinCommission(bt.CommInfoBase):
    params = (
        ('commission', 0.000086),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * price * self.p.commission


def run_backtest(symbol, start_date, end_date, initial_cash, mode, buy_values, sell_values):
    csv_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(csv_path):
        return {"error": f"数据文件不存在: {symbol}"}

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.loc[start_date:end_date]
    if df.empty:
        return {"error": f"股票 {symbol} 在 {start_date} ~ {end_date} 期间无数据"}

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.addcommissioninfo(NoMinCommission())

    data = DividendYieldData(dataname=df)
    cerebro.adddata(data, name=symbol)
    cerebro.addstrategy(
        DividendYieldStrategy,
        mode=mode,
        buy_values=buy_values,
        sell_values=sell_values,
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    cash = cerebro.broker.get_cash()
    position = strat.getposition(strat.datas[0]).size
    close_price = strat.datas[0].close[0]
    position_value = position * close_price

    dd = strat.analyzers.getbyname('drawdown').get_analysis()
    max_dd = dd.get('max', {}).get('drawdown', 0)

    sell_total = strat.sell_total_count
    sell_profit = strat.sell_profit_count
    win_rate = (sell_profit / sell_total * 100) if sell_total > 0 else 0

    actual_profit = final_value - initial_cash
    actual_return = actual_profit / initial_cash * 100

    dates = [r['date'] for r in strat.daily_records]
    closes = [r['close'] for r in strat.daily_records]
    dy_list = [r['dy'] for r in strat.daily_records]
    portfolio_values = [r['portfolio_value'] for r in strat.daily_records]
    buy_signals = [
        {"date": r['date'], "price": r['close'], "shares": r['shares']}
        for r in strat.daily_records if r['action'] == 'buy'
    ]
    sell_signals = [
        {"date": r['date'], "price": r['close'], "shares": r['shares']}
        for r in strat.daily_records if r['action'] == 'sell'
    ]

    return {
        "summary": {
            "initial_cash": initial_cash,
            "final_value": round(final_value, 2),
            "cash": round(cash, 2),
            "position": position,
            "close_price": round(close_price, 2),
            "position_value": round(position_value, 2),
            "actual_profit": round(actual_profit, 2),
            "actual_return": round(actual_return, 2),
            "total_buy_shares": strat.total_buy_shares,
            "total_sell_shares": strat.total_sell_shares,
            "total_buy_amount": round(strat.total_buy_amount, 2),
            "total_sell_amount": round(strat.total_sell_amount, 2),
            "max_position": strat.max_position,
            "sell_total": sell_total,
            "sell_profit": sell_profit,
            "sell_loss": sell_total - sell_profit,
            "win_rate": round(win_rate, 1),
            "max_drawdown": round(max_dd, 2),
        },
        "chart_data": {
            "dates": dates,
            "close": closes,
            "dividend_yield": dy_list,
            "portfolio_value": portfolio_values,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
        }
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/stocks')
def get_stocks():
    return jsonify({"stocks": STOCKS})


@app.route('/api/backtest', methods=['POST'])
def backtest():
    data = request.json

    symbol = data.get('symbol', '601398')
    start_date = data.get('start_date', '2021-04-14')
    end_date = data.get('end_date', '2026-04-14')
    initial_cash = data.get('initial_cash', 30000)
    mode = data.get('mode', 'percentage')

    buy_thresholds = data.get('buy_thresholds', {})
    sell_thresholds = data.get('sell_thresholds', {})

    buy_values = [
        buy_thresholds.get('4.7_4.8', 5),
        buy_thresholds.get('4.8_4.9', 6),
        buy_thresholds.get('4.9_5.0', 8),
        buy_thresholds.get('5.0_plus', 10),
    ]
    sell_values = [
        sell_thresholds.get('4.2_4.3', 5),
        sell_thresholds.get('4.1_4.2', 6),
        sell_thresholds.get('4.0_4.1', 8),
        sell_thresholds.get('below_4.0', 10),
    ]

    if mode == 'fixed':
        buy_values = [v * 100 for v in buy_values]
        sell_values = [v * 100 for v in sell_values]

    try:
        result = run_backtest(symbol, start_date, end_date, initial_cash, mode, buy_values, sell_values)
    except Exception as e:
        result = {"error": str(e)}

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
