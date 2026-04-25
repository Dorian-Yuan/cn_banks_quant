import os
import pandas as pd
import numpy as np
from .config import DATA_DIR, BANK_INDEX_CODE, FIVE_BANK_CODES, SIX_BANK_CODES, PSBC_START, DATA_START


_cache = {}


def load_bank_data(symbol):
    if symbol in _cache:
        return _cache[symbol]
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    if "DivCash" in df.columns:
        df["DivCash"] = pd.to_numeric(df["DivCash"], errors="coerce").fillna(0.0)
    else:
        df["DivCash"] = 0.0
    for col in ["Open", "Close", "High", "Low"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])
    _cache[symbol] = df
    return df


def load_all_data(symbols):
    result = {}
    for sym in symbols:
        result[sym] = load_bank_data(sym)
    return result


def load_index_data():
    return load_bank_data(BANK_INDEX_CODE)


def build_merged_data(data_dict, start_date, end_date):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    filtered = {}
    all_dates = set()
    for sym, df in data_dict.items():
        mask = (df["Date"] >= start) & (df["Date"] <= end)
        sub = df.loc[mask].copy()
        sub = sub.set_index("Date")
        filtered[sym] = sub
        all_dates.update(sub.index.tolist())

    if not all_dates:
        return {}, []

    dates = sorted(all_dates)
    date_index = pd.DatetimeIndex(dates)

    merged = {}
    for sym, sub in filtered.items():
        reindexed = sub.reindex(date_index)
        reindexed = reindexed.ffill()
        merged[sym] = reindexed

    valid_end = end
    for sym, sub in merged.items():
        last_valid = sub.index[-1] if len(sub) > 0 else end
        if last_valid < valid_end:
            valid_end = last_valid

    final_dates = [d for d in dates if d <= valid_end]
    for sym in merged:
        merged[sym] = merged[sym].loc[final_dates]

    return merged, final_dates


def get_trading_dates(symbols, start_date, end_date):
    data_dict = load_all_data(symbols)
    _, dates = build_merged_data(data_dict, start_date, end_date)
    return [d.strftime("%Y-%m-%d") for d in dates]


def get_all_trading_dates(start_date=None, end_date=None):
    if start_date is None:
        start_date = DATA_START
    if end_date is None:
        end_date = _get_latest_date()
    all_dates = set()
    for sym in FIVE_BANK_CODES:
        df = load_bank_data(sym)
        mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)
        all_dates.update(df.loc[mask, "Date"].tolist())
    return sorted([d.strftime("%Y-%m-%d") for d in all_dates])


def _get_latest_date():
    df = load_bank_data("601398")
    return df["Date"].max().strftime("%Y-%m-%d")
