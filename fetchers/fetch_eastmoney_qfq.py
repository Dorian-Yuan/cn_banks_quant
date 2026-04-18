import requests
import csv
import time
import os

BANKS = {
    '601398': '工商银行',
    '601288': '农业银行',
    '601988': '中国银行',
    '601939': '建设银行',
    '601328': '交通银行',
    '601658': '邮储银行'
}

KLINE_URL = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
DIV_URL = 'https://datacenter-web.eastmoney.com/api/data/v1/get'

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'ashare')
os.makedirs(OUT_DIR, exist_ok=True)

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
})


def fetch_all_klines(sym, fqt='0'):
    all_klines = []
    lmt = 500
    offset = 0
    retries = 3
    for attempt in range(retries):
        while True:
            params = {
                'fields1': 'f1,f2,f3,f4,f5,f6',
                'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
                'ut': '7eea3edcaed734bea9cbfc24409ed989',
                'klt': '101',
                'fqt': fqt,
                'secid': f'1.{sym}',
                'beg': '0',
                'end': '20500101',
                'lmt': str(lmt),
                'off': str(offset)
            }
            try:
                r = session.get(KLINE_URL, params=params, timeout=30)
                data = r.json()
                klines = data.get('data', {}).get('klines', [])
                if not klines:
                    break
                all_klines.extend(klines)
                if len(klines) < lmt:
                    break
                offset += lmt
            except Exception as e:
                print(f'  Kline error (attempt {attempt+1}/{retries}): {e}')
                if attempt < retries - 1:
                    print('  Retrying...')
                    time.sleep(2)
                    break
                else:
                    print('  Max retries reached')
                    return all_klines
    return all_klines


def fetch_dividends(sym):
    div_map = {}
    params = {
        'sortColumns': 'PLAN_NOTICE_DATE',
        'sortTypes': '-1',
        'pageSize': '50',
        'pageNumber': '1',
        'reportName': 'RPT_SHAREBONUS_DET',
        'columns': 'ALL',
        'source': 'WEB',
        'client': 'WEB',
        'filter': f'(SECURITY_CODE="{sym}")',
    }
    try:
        r = session.get(DIV_URL, params=params, timeout=15)
        data = r.json()
        if not data.get('result') or not data['result'].get('data'):
            return div_map
        for item in data['result']['data']:
            ex_date = item.get('EX_DIVIDEND_DATE')
            pretax = item.get('PRETAX_BONUS_RMB')
            progress = item.get('ASSIGN_PROGRESS', '')
            if not pretax or pretax <= 0 or progress == '预案':
                continue
            if ex_date:
                date_str = ex_date.split(' ')[0]
                cash_per_share = round(pretax / 10.0, 6)
                div_map[date_str] = div_map.get(date_str, 0) + cash_per_share
    except Exception as e:
        print(f'  Dividend API error: {e}')
    return div_map


def detect_div_from_ratio(bfq_klines, qfq_klines):
    bfq_map = {}
    for k in bfq_klines:
        parts = k.split(',')
        bfq_map[parts[0]] = float(parts[2])

    prev_ratio = None
    div_dict = {}

    for k in qfq_klines:
        parts = k.split(',')
        date = parts[0]
        qfq_close = float(parts[2])
        bfq_close = bfq_map.get(date)

        if bfq_close and qfq_close > 0:
            ratio = bfq_close / qfq_close
        else:
            ratio = prev_ratio if prev_ratio else 1.0

        div_cash = 0.0
        if prev_ratio is not None and prev_ratio > ratio + 0.001:
            div_cash = round((prev_ratio - ratio) * qfq_close, 4)

        div_dict[date] = div_cash
        prev_ratio = ratio

    return div_dict


for sym, name in BANKS.items():
    print(f'\nFetching {sym} {name}...')

    bfq_klines = fetch_all_klines(sym, fqt='0')
    if not bfq_klines:
        print(f'  Failed to fetch bfq data, skipping')
        continue
    print(f'  bfq (不复权): {len(bfq_klines)} records')

    qfq_klines = fetch_all_klines(sym, fqt='1')
    print(f'  qfq (前复权): {len(qfq_klines)} records')

    div_api = fetch_dividends(sym)
    print(f'  Dividend API: {len(div_api)} events')

    div_ratio = detect_div_from_ratio(bfq_klines, qfq_klines)
    ratio_events = sum(1 for v in div_ratio.values() if v > 0)
    print(f'  Ratio detection: {ratio_events} events')

    div_final = {}
    all_dates = set(div_api.keys()) | set(div_ratio.keys())
    for d in all_dates:
        api_val = div_api.get(d, 0)
        ratio_val = div_ratio.get(d, 0)
        if api_val > 0:
            div_final[d] = api_val
        elif ratio_val > 0:
            div_final[d] = ratio_val

    out_path = os.path.join(OUT_DIR, f'{sym}.csv')
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'DivCash'])
        for k in bfq_klines:
            parts = k.split(',')
            date = parts[0]
            writer.writerow([
                date,
                parts[1],  # Open
                parts[3],  # High
                parts[4],  # Low
                parts[2],  # Close
                parts[5],  # Volume
                div_final.get(date, 0.0)
            ])

    div_count = sum(1 for v in div_final.values() if v > 0)
    print(f'  Saved {out_path} ({div_count} dividend events)')

    if '2024-01-15' in [k.split(',')[0] for k in bfq_klines]:
        for k in bfq_klines:
            if k.startswith('2024-01-15'):
                p = k.split(',')
                print(f'  Verify 2024-01-15: Open={p[1]} Close={p[2]} High={p[3]} Low={p[4]}')
                break

    time.sleep(1)

print('\nDone!')
