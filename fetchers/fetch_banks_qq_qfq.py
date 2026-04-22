import os
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['no_proxy'] = '*'

import urllib.request
import urllib.parse
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "ashare")
os.makedirs(DATA_DIR, exist_ok=True)

BIG6_BANKS = {
    "601398": "工商银行",
    "601288": "农业银行",
    "601988": "中国银行",
    "601939": "建设银行",
    "601328": "交通银行",
    "601658": "邮储银行",
}

BANK_INDEX = {"399986": "中证银行"}

END_DATE = datetime.today().strftime("%Y-%m-%d")
START_DATE = "2015-01-01"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

QQ_MAX_COUNT = 800


def fetch_kline_qq(symbol, market_prefix="sh", fq="qfq"):
    qq_symbol = f"{market_prefix}{symbol}"
    all_rows = []
    end_dt = END_DATE
    print(f"  [{symbol}] 获取K线 (腾讯财经, 分页, {fq}) {START_DATE}→{END_DATE}...")

    for page in range(10):
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={qq_symbol},day,,{end_dt},{QQ_MAX_COUNT},{fq}"
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                resp = urllib.request.urlopen(req, timeout=30)
                raw = resp.read().decode('utf-8', errors='replace')
                data = json.loads(raw)
                if data.get('code') != 0 or not isinstance(data.get('data'), dict):
                    print(f"    第{page+1}页: API返回错误")
                    break
                sh_data = data['data'].get(qq_symbol, {})
                if fq == "qfq":
                    klines = sh_data.get('qfqday', [])
                else:
                    klines = sh_data.get('day', [])
                if not klines:
                    print(f"    第{page+1}页: 无数据")
                    break
                for k in klines:
                    all_rows.append({
                        "Date": k[0],
                        "Open": float(k[1]),
                        "Close": float(k[2]),
                        "High": float(k[3]),
                        "Low": float(k[4]),
                        "Volume": int(float(k[5])),
                    })
                earliest = klines[0][0]
                print(f"    第{page+1}页: {len(klines)}条, 最早={earliest}")
                if earliest <= START_DATE:
                    print(f"    已覆盖起始日期")
                    break
                end_dt = (pd.Timestamp(earliest) - timedelta(days=1)).strftime("%Y-%m-%d")
                break
            except Exception as e:
                print(f"    第{page+1}页第{attempt+1}次失败: {e}")
                time.sleep(3)
        else:
            break
        if earliest <= START_DATE:
            break
        time.sleep(0.5)

    if not all_rows:
        print(f"    无数据返回")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["Date"] = pd.to_datetime(df["Date"])
    start_dt = pd.to_datetime(START_DATE)
    df = df[df["Date"] >= start_dt].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    df.set_index("Date", inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    print(f"    K线总行数: {len(df)}  ({df.index[0].date()} ~ {df.index[-1].date()})")

    today_str = END_DATE
    if len(df) > 0 and df.index[-1].strftime("%Y-%m-%d") < today_str and fq == "qfq":
        print(f"    qfq数据未更新到今天({today_str})，尝试从不复权数据补充...")
        nfq_df = fetch_kline_qq(symbol, market_prefix=market_prefix, fq="")
        if not nfq_df.empty and len(nfq_df) > 0:
            missing_dates = nfq_df.index.difference(df.index)
            if len(missing_dates) > 0:
                last_qfq_close = df["Close"].iloc[-1]
                last_nfq_close = nfq_df["Close"].iloc[nfq_df.index <= df.index[-1]].iloc[-1] if len(nfq_df[nfq_df.index <= df.index[-1]]) > 0 else last_qfq_close
                adj_ratio = last_qfq_close / last_nfq_close if last_nfq_close > 0 else 1.0
                missing_rows = nfq_df.loc[missing_dates].copy()
                for col in ["Open", "Close", "High", "Low"]:
                    missing_rows[col] = missing_rows[col] * adj_ratio
                df = pd.concat([df, missing_rows]).sort_index()
                df = df[~df.index.duplicated(keep='last')]
                print(f"    补充了{len(missing_dates)}天数据，最新={df.index[-1].date()}，调整比例={adj_ratio:.6f}")

    return df


def fetch_pb(symbol):
    import akshare as ak
    print(f"  [{symbol}] 获取日度PB (市净率, AkShare)...")
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
    base_url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    url = base_url + "?" + urllib.parse.urlencode(params)
    print(f"  [{symbol}] 获取分红数据 (东方财富)...")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={
                **HEADERS,
                'Referer': 'https://data.eastmoney.com/',
            })
            resp = urllib.request.urlopen(req, timeout=15)
            raw = resp.read().decode('utf-8', errors='replace')
            data = json.loads(raw)
            if not data.get("result") or not data["result"].get("data"):
                return pd.DataFrame()
            rows = []
            for item in data["result"]["data"]:
                ex_date = item.get("EX_DIVIDEND_DATE")
                pretax = item.get("PRETAX_BONUS_RMB")
                progress = item.get("ASSIGN_PROGRESS", "")
                if not pretax or pretax <= 0 or "预案" in str(progress):
                    continue
                if not ex_date:
                    continue
                ex_date_ts = pd.Timestamp(ex_date.split(" ")[0])
                rows.append({"ex_dividend_date": ex_date_ts, "cash_per_share": pretax / 10.0})
            df = pd.DataFrame(rows).dropna(subset=["ex_dividend_date"])
            print(f"    分红条数: {len(df)}")
            return df
        except Exception as e:
            print(f"    第{attempt+1}次失败: {e}")
            time.sleep(2)
    return pd.DataFrame()


def build_and_save(symbol, name):
    print(f"\n{'='*55}")
    print(f"[{symbol} {name}] START")

    kline = fetch_kline_qq(symbol, market_prefix="sh", fq="qfq")
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

    sample_date = pd.Timestamp("2024-01-15")
    if sample_date in kline.index:
        sample_close = kline.loc[sample_date, "Close"]
        print(f"    验证: 2024-01-15 Close = {sample_close}")

    print(f"  [{symbol}] 保存完成: {csv_path}")
    print(f"    最新PB: {pb_now:.3f} | 合计分红现金流: {total_div:.4f} 元/股")


def build_index(symbol, name):
    print(f"\n{'='*55}")
    print(f"[{symbol} {name}] START (指数)")

    kline = fetch_kline_qq(symbol, market_prefix="sz", fq="")
    if kline.empty:
        print(f"  [{symbol}] K线获取失败，跳过")
        return

    csv_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    kline.to_csv(csv_path, encoding="utf-8-sig")

    print(f"  [{symbol}] 保存完成: {csv_path}")
    print(f"    行数: {len(kline)}  ({kline.index[0].date()} ~ {kline.index[-1].date()})")


if __name__ == "__main__":
    print("=" * 55)
    print("六大行 + 银行指数 日K(前复权) + PB + 分红 数据抓取程序")
    print("数据源: 腾讯财经(K线) + AkShare(PB) + 东方财富(分红)")
    print(f"时间范围: {START_DATE} → {END_DATE}")
    print("=" * 55)

    for symbol, name in BIG6_BANKS.items():
        build_and_save(symbol, name)
        time.sleep(2)

    for symbol, name in BANK_INDEX.items():
        build_index(symbol, name)
        time.sleep(2)

    print("\n" + "=" * 55)
    print("全部完成！数据已保存至 cn_banks_quant/data/ashare/")
    print("=" * 55)
