"""
fetch_banks_10yr.py
===================
Fetches 10 years of daily K-line data (前复权) + Daily PB Ratios (市净率)
+ Precise dividend cash flows for the six major state-owned Chinese banks.

Data is saved to: cn_banks_quant/data/ashare/<symbol>.csv

Columns in output CSV:
  Date, Open, High, Low, Close, Volume, DivCash, PB
"""

import os

# Disable proxies to avoid connection issues in local environment
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['no_proxy'] = '*'

import sys
import io
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Relative path: this file is in cn_banks_quant/fetchers/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "ashare")
os.makedirs(DATA_DIR, exist_ok=True)

# Six major state-owned Chinese banks
BIG6_BANKS = {
    "601398": "工商银行",
    "601288": "农业银行",
    "601988": "中国银行",
    "601939": "建设银行",
    "601328": "交通银行",
    "601658": "邮储银行",
}

# Date range: 10 years back from today
END_DATE   = datetime.today().strftime("%Y%m%d")
START_DATE = (datetime.today() - timedelta(days=365 * 10 + 5)).strftime("%Y%m%d")

session = requests.Session()
session.trust_env = False
session.proxies = {'http': None, 'https': None}
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
})


# ── K-Line via AkShare (前复权 = qfq) ─────────────────────────────────────────
def fetch_kline(symbol: str) -> pd.DataFrame:
    import akshare as ak
    print(f"  [{symbol}] 获取K线 (qfq) {START_DATE}→{END_DATE}...")
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=START_DATE,
                end_date=END_DATE,
                adjust="qfq",
            )
            df = df.rename(columns={
                "日期": "Date",
                "开盘": "Open",
                "最高": "High",
                "最低": "Low",
                "收盘": "Close",
                "成交量": "Volume",
            })[["Date", "Open", "High", "Low", "Close", "Volume"]]
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            df.set_index("Date", inplace=True)
            print(f"    K线行数: {len(df)}  ({df.index[0].date()} ~ {df.index[-1].date()})")
            return df
        except Exception as e:
            print(f"    第{attempt+1}次失败: {e}")
            time.sleep(3)
    return pd.DataFrame()


# ── PB Ratio via AkShare (stock_value_em) ─────────────────────────────────────
def fetch_pb(symbol: str) -> pd.Series:
    import akshare as ak
    print(f"  [{symbol}] 获取日度PB (市净率)...")
    for attempt in range(3):
        try:
            df = ak.stock_value_em(symbol=symbol)
            # Columns (encoded but known): 日期, 收盘价, 涨跌幅, 总市值, 流通市值, 总股本, 流通股本, PE(TTM), PE(静), 市销率, PEG值, PB, 市净率
            # The known last column is PB; pick the column labeled 'PB' (market vs book)
            # Rename to 'Date' the first column regardless of encoding
            df.columns = [
                "Date", "Close_val", "PctChg", "TotalMktCap", "FreeMktCap",
                "TotalShares", "FreeShares", "PE_TTM", "PE_static", "PS", "PEG", "PB", "extra"
            ][:len(df.columns)]
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").set_index("Date")
            pb_series = df["PB"].astype(float)
            print(f"    PB行数: {len(pb_series)}  ({pb_series.index[0].date()} ~ {pb_series.index[-1].date()})")
            return pb_series
        except Exception as e:
            print(f"    第{attempt+1}次失败: {e}")
            time.sleep(3)
    return pd.Series(dtype=float, name="PB")


# ── Dividend (EastMoney API) ───────────────────────────────────────────────────
def fetch_dividend(symbol: str) -> pd.DataFrame:
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "sortColumns": "PLAN_NOTICE_DATE",
        "sortTypes": "-1",
        "pageSize": "50",
        "pageNumber": "1",
        "reportName": "RPT_SHAREBONUS_DET",
        "columns": "ALL",
        "source": "WEB",
        "client": "WEB",
        "filter": f'(SECURITY_CODE="{symbol}")',
    }
    print(f"  [{symbol}] 获取分红数据...")
    for attempt in range(3):
        try:
            r = session.get(url, params=params, timeout=15)
            data = r.json()
            if not data.get("result") or not data["result"].get("data"):
                return pd.DataFrame()
            rows = []
            for item in data["result"]["data"]:
                ex_date     = item.get("EX_DIVIDEND_DATE")
                pretax      = item.get("PRETAX_BONUS_RMB")
                progress    = item.get("ASSIGN_PROGRESS", "")
                if not pretax or pretax <= 0 or progress == "预案":
                    continue
                ex_date_ts = pd.Timestamp(ex_date.split(" ")[0]) if ex_date else pd.NaT
                rows.append({"ex_dividend_date": ex_date_ts, "cash_per_share": pretax / 10.0})
            df = pd.DataFrame(rows).dropna(subset=["ex_dividend_date"])
            print(f"    分红条数: {len(df)}")
            return df
        except Exception as e:
            print(f"    第{attempt+1}次失败: {e}")
            time.sleep(2)
    return pd.DataFrame()


# ── Merge & Save ──────────────────────────────────────────────────────────────
def build_and_save(symbol: str, name: str):
    print(f"\n{'='*55}")
    print(f"[{symbol} {name}] START")

    kline = fetch_kline(symbol)
    if kline.empty:
        print(f"  [{symbol}] K线获取失败，跳过"); return

    pb_series = fetch_pb(symbol)
    if not pb_series.empty:
        # On non-trading days PB data may use last-observation; interpolate daily into kline index
        pb_full = pb_series.reindex(kline.index, method="ffill")
        kline["PB"] = pb_full
    else:
        kline["PB"] = np.nan

    # Dividends → DivCash column
    kline["DivCash"] = 0.0
    div_df = fetch_dividend(symbol)
    if not div_df.empty:
        for _, row in div_df.iterrows():
            ex_date = row["ex_dividend_date"]
            if ex_date in kline.index:
                kline.loc[ex_date, "DivCash"] += row["cash_per_share"]

    csv_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    kline.to_csv(csv_path, encoding="utf-8-sig")
    
    # Summary
    pb_now = kline["PB"].dropna().iloc[-1] if not kline["PB"].dropna().empty else float("nan")
    total_div = kline["DivCash"].sum()
    print(f"  [{symbol}] 保存完成: {csv_path}")
    print(f"    最新PB: {pb_now:.3f} | 合计分红现金流: {total_div:.4f} 元/股")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("六大行 10年 日K + PB + 分红 数据抓取程序")
    print(f"时间范围: {START_DATE} → {END_DATE}")
    print("=" * 55)

    for symbol, name in BIG6_BANKS.items():
        build_and_save(symbol, name)
        time.sleep(2)  # Polite delay between requests

    print("\n" + "=" * 55)
    print("全部完成！数据已保存至 cn_banks_quant/data/ashare/")
    print("=" * 55)
