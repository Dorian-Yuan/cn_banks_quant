import os
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['no_proxy'] = '*'

import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "ashare")
os.makedirs(DATA_DIR, exist_ok=True)

BIG6_BANKS = {
    "601398": ("1", "工商银行"),
    "601288": ("1", "农业银行"),
    "601988": ("1", "中国银行"),
    "601939": ("1", "建设银行"),
    "601328": ("1", "交通银行"),
    "601658": ("1", "邮储银行"),
}

END_DATE = datetime.today().strftime("%Y%m%d")
START_DATE = (datetime.today() - timedelta(days=365 * 10 + 5)).strftime("%Y%m%d")

session = requests.Session()
session.trust_env = False
session.proxies = {'http': None, 'https': None}
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
})


def fetch_kline_em(symbol, market):
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": f"{market}.{symbol}",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "0",
        "beg": START_DATE,
        "end": END_DATE,
    }
    print(f"  [{symbol}] 获取K线 (不复权) {START_DATE}→{END_DATE}...")
    for attempt in range(3):
        try:
            r = session.get(url, params=params, timeout=30)
            data = r.json()
            klines = data.get("data", {}).get("klines", [])
            if not klines:
                print(f"    无数据返回")
                return pd.DataFrame()
            rows = []
            for k in klines:
                parts = k.split(",")
                rows.append({
                    "Date": parts[0],
                    "Open": float(parts[1]),
                    "High": float(parts[2]),
                    "Low": float(parts[3]),
                    "Close": float(parts[4]),
                    "Volume": int(parts[5]),
                })
            df = pd.DataFrame(rows)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            df.set_index("Date", inplace=True)
            print(f"    K线行数: {len(df)}  ({df.index[0].date()} ~ {df.index[-1].date()})")
            return df
        except Exception as e:
            print(f"    第{attempt+1}次失败: {e}")
            time.sleep(5)
    return pd.DataFrame()


def fetch_pb(symbol):
    import akshare as ak
    print(f"  [{symbol}] 获取日度PB (市净率)...")
    for attempt in range(3):
        try:
            df = ak.stock_value_em(symbol=symbol)
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


def fetch_dividend(symbol):
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
                ex_date = item.get("EX_DIVIDEND_DATE")
                pretax = item.get("PRETAX_BONUS_RMB")
                progress = item.get("ASSIGN_PROGRESS", "")
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


def build_and_save(symbol, market, name):
    print(f"\n{'='*55}")
    print(f"[{symbol} {name}] START")

    kline = fetch_kline_em(symbol, market)
    if kline.empty:
        print(f"  [{symbol}] K线获取失败，跳过")
        return

    pb_series = fetch_pb(symbol)
    if not pb_series.empty:
        pb_full = pb_series.reindex(kline.index, method="ffill")
        kline["PB"] = pb_full
    else:
        kline["PB"] = np.nan

    kline["DivCash"] = 0.0
    div_df = fetch_dividend(symbol)
    if not div_df.empty:
        for _, row in div_df.iterrows():
            ex_date = row["ex_dividend_date"]
            if ex_date in kline.index:
                kline.loc[ex_date, "DivCash"] += row["cash_per_share"]

    csv_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    kline.to_csv(csv_path, encoding="utf-8-sig")

    pb_now = kline["PB"].dropna().iloc[-1] if not kline["PB"].dropna().empty else float("nan")
    total_div = kline["DivCash"].sum()
    print(f"  [{symbol}] 保存完成: {csv_path}")
    print(f"    最新PB: {pb_now:.3f} | 合计分红现金流: {total_div:.4f} 元/股")


if __name__ == "__main__":
    print("=" * 55)
    print("六大行 10年 日K(不复权) + PB + 分红 数据抓取程序")
    print(f"时间范围: {START_DATE} → {END_DATE}")
    print("=" * 55)

    for symbol, (market, name) in BIG6_BANKS.items():
        build_and_save(symbol, market, name)
        time.sleep(2)

    print("\n" + "=" * 55)
    print("全部完成！数据已保存至 cn_banks_quant/data/ashare/")
    print("=" * 55)
