import pandas as pd

def main():
    files = [
        'datasets/integrated_Apr.csv',
        'datasets/integrated_May_later.csv',
        'datasets/integrated_vd_livetraffic_data_sorted.csv'
    ]

    dfs = []
    for f in files:
        # 1. 讀取為字串以確保 Timestamp 格式一致
        df = pd.read_csv(f, dtype={'Timestamp': str})
        # 2. 統一格式並解析為帶時區的 datetime
        df['Timestamp'] = pd.to_datetime(
            df['Timestamp'].str.replace('T', ' '),
            format='%Y-%m-%d %H:%M:%S%z',
            errors='coerce' # 若有無法解析的格式則設為 NaT
        )
        dfs.append(df)

    # 3. 合併、排序並移除時間解析失敗的資料
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=['Timestamp']) # 移除 NaT
    df = df.sort_values('Timestamp')

    # 4. 資料篩選
    # 4.1. 時間範圍：2025-04-01 00:00:00+08:00 ～ 2025-06-10 23:59:00+08:00
    start = pd.Timestamp('2025-04-01 00:00:00+08:00')
    end   = pd.Timestamp('2025-06-10 23:59:00+08:00')
    time_condition = (df['Timestamp'] >= start) & (df['Timestamp'] <= end)
    
    # 4.2. Status 條件
    #status_condition = df['Status'] == 0
    
    # 4.3. CongestionLevel 條件：篩選數值為 1~5 的資料
    # 確保 CongestionLevel 欄位存在且為數值型態
    if 'CongestionLevel' in df.columns:
        df['CongestionLevel'] = pd.to_numeric(df['CongestionLevel'], errors='coerce')
        congestion_condition = df['CongestionLevel'].between(1, 5)
        
        # 4.4. 套用所有篩選條件
        df = df[time_condition &  congestion_condition]
    else:
        # 若部分檔案無 CongestionLevel 欄位，則僅依其他條件篩選
        df = df[time_condition ]


    # 5. 刪除欄位 (保留 CongestionLevel, 刪除 CongestionLevelID)
    cols_to_drop = [
        'FileName', 'Status', 'LinkID', 'LaneID', 'LaneType',
        'LaneSpeed', 'CongestionLevelID', 'HasHistorical', 'HasVD',
        'HasAVI', 'HasETAG', 'HasGVP', 'HasCVP', 'HasOthers'
    ]
    # 僅刪除資料集中實際存在的欄位
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 6. 輸出
    df.to_csv('datasets/0401_0610/vd_livetraffic_data_0401_0610.csv', index=False)

if __name__ == '__main__':
    main()