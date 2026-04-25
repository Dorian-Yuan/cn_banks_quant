import pandas as pd
import json
import os

# 配置路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "ashare")
OUTPUT_DIR = os.path.join(BASE_DIR, "research", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "bank_valuation.json")

# 银行映射
BIG6_BANKS = {
    "601398": "工商银行",
    "601939": "建设银行",
    "601988": "中国银行",
    "601288": "农业银行",
    "601328": "交通银行",
    "601658": "邮储银行",
}

def process_data():
    all_data = {}
    
    for symbol, name in BIG6_BANKS.items():
        csv_path = os.path.join(DATA_DIR, f"{symbol}.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found.")
            continue
            
        print(f"Processing {name} ({symbol})...")
        df = pd.read_csv(csv_path)
        
        # 确保日期格式正确
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 过滤掉 PB 缺失的数据（主要集中在早期，如 2016-2017）
        df = df.dropna(subset=['PB', 'Close'])
        
        # 按日期排序
        df = df.sort_values('Date')
        
        # 提取核心数据
        # 我们只保留日期、收盘价和 PB
        bank_info = {
            "name": name,
            "symbol": symbol,
            "dates": df['Date'].dt.strftime('%Y-%m-%d').tolist(),
            "prices": [round(float(x), 3) for x in df['Close'].tolist()],
            "pbs": [round(float(x), 4) for x in df['PB'].tolist()],
            "stats": {
                "max_pb": round(float(df['PB'].max()), 4),
                "min_pb": round(float(df['PB'].min()), 4),
                "avg_pb": round(float(df['PB'].mean()), 4),
                "latest_pb": round(float(df['PB'].iloc[-1]), 4),
                "latest_price": round(float(df['Close'].iloc[-1]), 3)
            }
        }
        
        all_data[symbol] = bank_info
        
    # 写入 JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully generated {OUTPUT_FILE}")

if __name__ == "__main__":
    process_data()
