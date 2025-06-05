import pandas as pd
import os

# 設定檔案路徑
ROAD_GEOM_PATH = '/mnt/data/road_geometry_translated.csv'
TRAFFIC_PATH = '/mnt/data/traffic_data.csv'
OUTPUT_PATH = '/mnt/data/traffic_with_geometry.csv'

# 讀取道路幾何資料
df_road = pd.read_csv(ROAD_GEOM_PATH, encoding='utf-8')

# 讀取路況資料
df_traffic = pd.read_csv(TRAFFIC_PATH, encoding='utf-8')

# 解析 VDID，擷取公里數 (例如 'VD-N1-N-26.780-M-RS' → 26.780)
def parse_vdid_km(vdid_str):
    parts = vdid_str.split('-')
    for part in parts:
        try:
            return float(part)
        except ValueError:
            continue
    return None

df_traffic['VD_KM'] = df_traffic['VDID'].apply(parse_vdid_km)

# 解析 Kilometer Marker (例如 '026K+800' → 26.800)
def parse_marker_km(marker_str):
    try:
        km_part, m_part = marker_str.split('K+')
        km = int(km_part)
        m = int(m_part)
        return km + m / 1000.0
    except (ValueError, AttributeError):
        return None

df_road['Road_KM'] = df_road['Kilometer Marker'].apply(parse_marker_km)

# 移除解析失敗的列
df_traffic = df_traffic.dropna(subset=['VD_KM']).reset_index(drop=True)
df_road    = df_road.dropna(subset=['Road_KM']).reset_index(drop=True)

# 依 Road_KM 排序
df_road_sorted = df_road.sort_values('Road_KM').reset_index(drop=True)
# 依 VD_KM 排序
df_traffic_sorted = df_traffic.sort_values('VD_KM').reset_index(drop=True)

# 使用 merge_asof 合併最接近的道路幾何資料
df_merged = pd.merge_asof(
    df_traffic_sorted,
    df_road_sorted,
    left_on='VD_KM',
    right_on='Road_KM',
    direction='nearest',
    suffixes=('_traffic', '_road')
)

# 回復原始順序（若需要）
df_merged_final = df_merged.sort_index()

# 儲存合併後的結果
df_merged_final.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
