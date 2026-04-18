import pandas as pd

# Load the equity curve
df = pd.read_csv('backtests/equity_curve.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# Initial Value
INITIAL_CASH = 30000.0

# Get Year End Values
years = sorted(df['Year'].unique())
year_ends = {}

# Special case for "Start of 2019"
year_ends[2018] = INITIAL_CASH # Effectively "Value before start of 2019"

for y in years:
    # Last trading day of the year
    y_val = df[df['Year'] == y]['Value'].iloc[-1]
    year_ends[y] = y_val

def generate_table(duration):
    rows = []
    # Start year goes from 2019 up to whatever allows the duration
    for start_y in range(2019, 2026 - duration + 2):
        end_y = start_y + duration - 1
        if end_y > 2026: break
        
        # Period: Start of start_y to End of end_y
        start_val = year_ends[start_y - 1]
        end_val = year_ends[end_y]
        
        ret = (end_val / start_val - 1) * 100
        rows.append({
            '时间段': f"{start_y % 100:02d}-{end_y % 100:02d}",
            '总收益率': f"{ret:.2f}%"
        })
    return pd.DataFrame(rows)

print("\n--- 2年期收益率表 ---")
print(generate_table(2).to_markdown(index=False))

print("\n--- 3年期收益率表 ---")
print(generate_table(3).to_markdown(index=False))

print("\n--- 4年期收益率表 ---")
print(generate_table(4).to_markdown(index=False))

print("\n--- 5年期收益率表 ---")
print(generate_table(5).to_markdown(index=False))
