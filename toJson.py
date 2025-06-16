import pandas as pd
import json
DATA_PATH = 'datasets/vd_livetraffic_data_0501_0521.csv'
df = pd.read_csv(DATA_PATH, parse_dates=['Timestamp'])
df = df.sort_values(['SectionID', 'Timestamp'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.set_index('Timestamp')
    # 中位數
agg_cfg = {
    'Occupancy':'median',
    'VehicleType_S_Volume':'median','VehicleType_S_Speed':'median',
    'VehicleType_L_Volume':'median','VehicleType_L_Speed':'median',
    'VehicleType_T_Volume':'median','VehicleType_T_Speed':'median',
    # 'TravelTime':'median','TravelSpeed':'median',
    # 'StnPres':'median','Temperature':'median','RH':'median','WS':'median','WD':'median',
    # 'WSGust':'median','WDGust':'median','Precip':'median','CongestionLevel':'median'
}

df_resampled = (
    df.groupby('SectionID')
    .resample('1min')
    .agg(agg_cfg)
    .dropna()
)
# 假設 df_resampled 已依前述方式生成，且 index 為 Timestamp，並且有多層索引 SectionID
# 篩選 SectionID=23
df23 = df_resampled.xs(23, level='SectionID')

# 篩選時間區間
start, end = '2025-05-10 00:00:00', '2025-05-10 23:59:00'
df23 = df23.loc[start:end]

# 僅保留指定欄位
cols = [
    'Occupancy',
    'VehicleType_S_Volume','VehicleType_S_Speed',
    'VehicleType_L_Volume','VehicleType_L_Speed',
    # 'StnPres','Temperature','RH','WS','WD','WSGust','WDGust','Precip'
]
df23 = df23[cols]

# 重設索引以取得 Timestamp 欄位
df23 = df23.reset_index()

# 格式化 Timestamp，轉成字串
df23['Timestamp'] = df23['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# 轉成 list of dict
records = df23.to_dict(orient='records')

# 組成最終結構
output = {
    "SectionID": "23",
    "data": records
}

# 寫入 JSON（UTF-8、縮排 2）
with open('traffic_section23_2025_05_10.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
