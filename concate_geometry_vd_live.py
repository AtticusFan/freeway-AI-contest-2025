import pandas as pd
import numpy as np

# --- 1. 載入最終確認的資料集 ---
try:
    # 載入您的 VD 偵測器資料
    vd_df = pd.read_csv('datasets/0501_0610/merged_traffic_weather_0501_0610.csv')
    # 載入最新的道路幾何資料檔案
    geom_df = pd.read_csv('datasets/N0010_road_geometry.csv')
    print("成功載入 'vd_data.csv' 與 'N0010_road_geometry.csv'。")
except FileNotFoundError as e:
    print(f"錯誤：找不到檔案 {e.filename}。請確認檔案存在於相同目錄下。")
    exit()

# --- 2. 預處理 N0010_road_geometry 資料 ---
def convert_mileage_to_km(mileage_mark):
    """將 'MileageMark' (例如 031K+400) 轉換為公里數 (例如 31.4)"""
    try:
        if isinstance(mileage_mark, str):
            parts = mileage_mark.split('K+')
            km = int(parts[0])
            meters = int(parts[1])
            return km + meters / 1000.0
    except (ValueError, IndexError, TypeError):
        return np.nan
    return np.nan

# [請在此處確認] 假定里程欄位為 'MileageMark'
if 'MileageMark' in geom_df.columns:
    geom_df['MileageKM'] = geom_df['MileageMark'].apply(convert_mileage_to_km)
else:
    print("錯誤: 在 'N0010_road_geometry.csv' 中找不到 'MileageMark' 欄位。")
    exit()

# [請在此處確認] 假定方向欄位為 'Direction'
if 'Direction' in geom_df.columns:
    direction_map = {'N': 'Northbound', 'S': 'Southbound'}
    direction_abbr_map = {v: k for k, v in direction_map.items()}
    geom_df['DirectionAbbr'] = geom_df['Direction'].map(direction_abbr_map)
else:
    print("錯誤: 在 'N0010_road_geometry.csv' 中找不到 'Direction' 欄位。")
    exit()

geom_df.dropna(subset=['MileageKM', 'DirectionAbbr'], inplace=True)

# --- 3. 預處理 VDID 資料 ---
if 'VDID' in vd_df.columns:
    vd_df['VD_Direction'] = vd_df['VDID'].astype(str).str.split('-').str[2]
    vd_df['VD_MileageKM'] = pd.to_numeric(vd_df['VDID'].astype(str).str.split('-').str[3], errors='coerce')
else:
    print("錯誤：'vd_data.csv' 中找不到 'VDID' 欄位。")
    exit()
    
vd_df.dropna(subset=['VD_MileageKM'], inplace=True)


# --- 4. 匹配與合併 ---
# 將幾何資料依方向分群，以提高匹配效率
grouped_geom = dict(tuple(geom_df.groupby('DirectionAbbr')))
matched_indices = []

# 遍歷每一筆 VDID 資料
for index, row in vd_df.iterrows():
    vd_dir = row['VD_Direction']
    vd_km = row['VD_MileageKM']

    # 從預先分群的字典中取得對應方向的子集
    subset_geom_df = grouped_geom.get(vd_dir)

    if subset_geom_df is not None and not subset_geom_df.empty:
        # 計算里程差異的絕對值
        mileage_diff = np.abs(subset_geom_df['MileageKM'] - vd_km)
        # 找到差異最小的那一筆資料的索引
        closest_match_original_idx = mileage_diff.idxmin()
        matched_indices.append(closest_match_original_idx)
    else:
        # 如果找不到對應方向的資料，則無法匹配
        matched_indices.append(np.nan)

# 根據找到的索引，從原始 geom_df 中提取匹配的行
matched_geometries_df = geom_df.loc[matched_indices].reset_index(drop=True)

# 將原始VDID資料與匹配到的幾何資料合併
final_df = pd.concat([vd_df.reset_index(drop=True), matched_geometries_df.add_prefix('geom_')], axis=1)

# 移除使用者指定不要的欄位
columns_to_drop = ['geom_Direction', 'geom_MileageMark']
final_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')


# --- 5. 輸出結果 ---
print("\n資料合併完成。結果預覽：")
print(final_df.head())

# 將結果儲存為新的 CSV 檔案
try:
    final_df.to_csv('datasets/0501_0610/merged_traffic_geometry_weather_0501_0610.csv', index=False)
    print("\n已將合併結果儲存至 'merged_vd_geometry.csv'")
except Exception as e:
    print(f"\n儲存檔案時發生錯誤: {e}")