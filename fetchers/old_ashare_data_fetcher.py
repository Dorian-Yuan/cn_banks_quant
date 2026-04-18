import os
import sys
import io
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ashare")
os.makedirs(DATA_DIR, exist_ok=True)

session = requests.Session()
session.trust_env = False
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
})

STOCKS = {
    "601328": "交通银行",
    "601939": "建设银行",
    "601398": "工商银行",
    "601288": "农业银行",
    "601988": "中国银行",
    "600036": "招商银行",
    "000001": "平安银行",
    "600015": "华夏银行",
    "601166": "兴业银行",
    "600919": "江苏银行",
    "600000": "浦发银行",
    "601818": "光大银行",
}


def fetch_stock_hist_sina(symbol, datalen=1500):
    market = "sh" if symbol.startswith("6") else "sz"
    url = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
    params = {
        "symbol": f"{market}{symbol}",
        "scale": "240",
        "ma": "no",
        "datalen": str(datalen),
    }

    for attempt in range(3):
        try:
            r = session.get(url, params=params, timeout=15)
            data = r.json()
            if not data:
                print(f"  [{symbol}] 新浪行情API返回空数据")
                return pd.DataFrame()

            rows = []
            for item in data:
                rows.append({
                    "Date": item["day"],
                    "Open": float(item["open"]),
                    "High": float(item["high"]),
                    "Low": float(item["low"]),
                    "Close": float(item["close"]),
                    "Volume": float(item["volume"]),
                })

            df = pd.DataFrame(rows)
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            return df
        except Exception as e:
            print(f"  [{symbol}] 新浪行情API第{attempt+1}次尝试失败: {e}")
            time.sleep(2)

    return pd.DataFrame()


def fetch_dividend_detail(symbol):
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "sortColumns": "PLAN_NOTICE_DATE",
        "sortTypes": "-1",
        "pageSize": "50",
        "pageNumber": "1",
        "reportName": "RPT_SHAREBONUS_DET",
        "columns": "ALL",
        "quoteColumns": "",
        "source": "WEB",
        "client": "WEB",
        "filter": f'(SECURITY_CODE="{symbol}")',
    }

    for attempt in range(3):
        try:
            r = session.get(url, params=params, timeout=15)
            data = r.json()
            if not data.get("result") or not data["result"].get("data"):
                print(f"  [{symbol}] 东方财富分红API返回空数据")
                return pd.DataFrame()

            rows = []
            for item in data["result"]["data"]:
                ex_date = item.get("EX_DIVIDEND_DATE")
                pretax_bonus = item.get("PRETAX_BONUS_RMB")
                progress = item.get("ASSIGN_PROGRESS", "")
                report_date = item.get("REPORT_DATE", "")
                plan_notice_date = item.get("PLAN_NOTICE_DATE", "")

                if not pretax_bonus or pretax_bonus <= 0:
                    continue

                ex_date_str = ex_date.split(" ")[0] if ex_date else None
                report_date_str = report_date.split(" ")[0] if report_date else None
                plan_notice_str = plan_notice_date.split(" ")[0] if plan_notice_date else None

                if progress in ("预案",):
                    continue

                rows.append({
                    "ex_dividend_date": pd.Timestamp(ex_date_str) if ex_date_str else pd.NaT,
                    "cash_per_share": pretax_bonus / 10.0,
                    "report_date": pd.Timestamp(report_date_str) if report_date_str else pd.NaT,
                    "plan_notice_date": pd.Timestamp(plan_notice_str) if plan_notice_str else pd.NaT,
                    "progress": progress,
                })

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows)
            df.sort_values("report_date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
        except Exception as e:
            print(f"  [{symbol}] 东方财富分红API第{attempt+1}次尝试失败: {e}")
            time.sleep(2)

    return pd.DataFrame()


def calculate_dividend_yield_ttm(price_df, dividend_df):
    if price_df.empty or dividend_df.empty:
        price_df["DividendYieldTTM"] = np.nan
        return price_df

    dividend_df = dividend_df[dividend_df["cash_per_share"] > 0].copy()
    dividend_df = dividend_df.dropna(subset=["report_date"]).reset_index(drop=True)

    if dividend_df.empty:
        price_df["DividendYieldTTM"] = 0.0
        return price_df

    dividend_df["fiscal_year"] = dividend_df["report_date"].dt.year

    fy_data = {}
    for fy, group in dividend_df.groupby("fiscal_year"):
        total_dps = group["cash_per_share"].sum()

        plan_dates = group["plan_notice_date"].dropna()
        ex_dates = group["ex_dividend_date"].dropna()

        if len(plan_dates) > 0:
            known_from = plan_dates.max()
        elif len(ex_dates) > 0:
            known_from = ex_dates.min()
        else:
            continue

        fy_data[fy] = {
            "total_dps": total_dps,
            "known_from": known_from,
        }

    if not fy_data:
        price_df["DividendYieldTTM"] = 0.0
        return price_df

    sorted_fys = sorted(fy_data.keys())
    fy_boundaries = []
    for fy in sorted_fys:
        fy_boundaries.append((fy_data[fy]["known_from"], fy_data[fy]["total_dps"]))
    fy_boundaries.sort(key=lambda x: x[0])

    dividend_yield_list = []
    for date in price_df.index:
        applicable_dps = 0.0
        for known_from, total_dps in fy_boundaries:
            if date >= known_from:
                applicable_dps = total_dps
            else:
                break

        close_price = price_df.loc[date, "Close"]
        if close_price > 0 and applicable_dps > 0:
            dy = (applicable_dps / close_price) * 100
        else:
            dy = 0.0
        dividend_yield_list.append(dy)

    price_df["DividendYieldTTM"] = dividend_yield_list
    return price_df


def fetch_and_save_stock(symbol, name, force=False):
    csv_path = os.path.join(DATA_DIR, f"{symbol}.csv")

    if os.path.exists(csv_path) and not force:
        print(f"[{symbol} {name}] 数据已存在，跳过: {csv_path}")
        return csv_path

    print(f"[{symbol} {name}] 开始获取数据...")

    print(f"  [{symbol}] 正在获取行情数据...")
    price_df = fetch_stock_hist_sina(symbol, datalen=1500)
    if price_df.empty:
        print(f"  [{symbol}] 行情数据获取失败，跳过")
        return None
    print(f"  [{symbol}] 行情数据: {len(price_df)}条, {price_df.index[0].date()} ~ {price_df.index[-1].date()}")

    print(f"  [{symbol}] 正在获取分红数据...")
    dividend_df = fetch_dividend_detail(symbol)
    if dividend_df.empty:
        print(f"  [{symbol}] 分红数据获取失败，使用0填充股息率")
    else:
        print(f"  [{symbol}] 分红数据: {len(dividend_df)}条")
        for _, row in dividend_df.iterrows():
            ex_str = str(row['ex_dividend_date'].date()) if pd.notna(row['ex_dividend_date']) else "未实施"
            plan_str = str(row['plan_notice_date'].date()) if pd.notna(row['plan_notice_date']) else "N/A"
            print(f"    报告期={row['report_date'].date()}, 预案公告日={plan_str}, "
                  f"除权日={ex_str}, 每股派息={row['cash_per_share']:.4f}元, 进度={row['progress']}")

    print(f"  [{symbol}] 正在计算预期年度股息率...")
    price_df = calculate_dividend_yield_ttm(price_df, dividend_df)

    # 注入精准的现金分红
    print(f"  [{symbol}] 正在合并精准的日度现金分红(DivCash)...")
    price_df["DivCash"] = 0.0
    if not dividend_df.empty:
        for _, row in dividend_df.iterrows():
            if pd.notna(row['ex_dividend_date']):
                ex_date = row['ex_dividend_date']
                if ex_date in price_df.index:
                    price_df.loc[ex_date, "DivCash"] += row['cash_per_share']

    price_df.to_csv(csv_path, encoding='utf-8-sig')
    print(f"  [{symbol}] 数据已保存: {csv_path}")

    dy_stats = price_df["DividendYieldTTM"].describe()
    print(f"  [{symbol}] 股息率统计: 均值={dy_stats['mean']:.2f}%, 最大={dy_stats['max']:.2f}%, 最小={dy_stats['min']:.2f}%")
    last = price_df.iloc[-1]
    print(f"  [{symbol}] 最新数据: 日期={price_df.index[-1].date()}, 收盘={last['Close']:.2f}, 股息率={last['DividendYieldTTM']:.2f}%")

    return csv_path


def main():
    print("=" * 60)
    print("A股银行股数据抓取程序 (预期年度股息率)")
    print("=" * 60)

    results = {}
    for symbol, name in STOCKS.items():
        csv_path = fetch_and_save_stock(symbol, name, force=True)
        results[symbol] = csv_path
        time.sleep(1)

    print("\n" + "=" * 60)
    print("数据抓取完成汇总")
    print("=" * 60)
    for symbol, csv_path in results.items():
        status = "成功" if csv_path else "失败"
        print(f"  {symbol} {STOCKS[symbol]}: {status}")


if __name__ == "__main__":
    main()
