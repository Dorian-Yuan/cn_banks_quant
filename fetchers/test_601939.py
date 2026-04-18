import requests
import time

KLINE_URL = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
})

def fetch_klines(sym, fqt='0'):
    params = {
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'ut': '7eea3edcaed734bea9cbfc24409ed989',
        'klt': '101',
        'fqt': fqt,
        'secid': f'1.{sym}',
        'beg': '20240101',
        'end': '20240131',
        'lmt': '50',
        'off': '0'
    }
    try:
        r = session.get(KLINE_URL, params=params, timeout=30)
        data = r.json()
        klines = data.get('data', {}).get('klines', [])
        return klines
    except Exception as e:
        print(f'Error: {e}')
        return []

print('Fetching 601939 建设银行 data...')

# 测试不复权数据
bfq_klines = fetch_klines('601939', fqt='0')
print(f'bfq (不复权) records: {len(bfq_klines)}')

for k in bfq_klines:
    parts = k.split(',')
    date = parts[0]
    if date == '2024-01-15':
        print(f'2024-01-15 (bfq): Open={parts[1]}, Close={parts[2]}, High={parts[3]}, Low={parts[4]}')

# 测试前复权数据
qfq_klines = fetch_klines('601939', fqt='1')
print(f'qfq (前复权) records: {len(qfq_klines)}')

for k in qfq_klines:
    parts = k.split(',')
    date = parts[0]
    if date == '2024-01-15':
        print(f'2024-01-15 (qfq): Open={parts[1]}, Close={parts[2]}, High={parts[3]}, Low={parts[4]}')

print('Done!')