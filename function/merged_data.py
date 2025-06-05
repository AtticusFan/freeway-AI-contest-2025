import pandas as pd
def load_and_merge_traffic_weather(traffic_path: str, weather_path: str) -> pd.DataFrame:
    """
    合併路況與氣象原始資料後回傳
    
    參數:
        traffic_path: 路況資料 CSV 檔案路徑 (須含 Timestamp 欄位)
        weather_path: 氣象資料 CSV 檔案路徑 (須含 ObsTime 欄位)
    
    回傳:
        pandas.DataFrame: 合併後的資料 (已移除 ObsTime 欄位)
    """
    # 1. 讀取路況資料
    traffic = pd.read_csv(
        traffic_path,
        parse_dates=['Timestamp']
    )

    # 2. 讀取氣象資料
    weather = pd.read_csv(
        weather_path,
        parse_dates=['ObsTime']
    )
    weather['ObsTime'] = weather['ObsTime'].dt.tz_localize('Asia/Taipei')

    # 3. 將 traffic 的 Timestamp 轉換到台北時區，取整點向上（+1 小時）作為 ObsTime
    traffic['ObsTime'] = (
        traffic['Timestamp']
        .dt.tz_convert('Asia/Taipei')
        .dt.floor('h')
        + pd.Timedelta(hours=1)
    )

    # 4. 以 ObsTime 為 key 合併，並移除 ObsTime 欄位
    merged_df = traffic.merge(
        weather,
        on='ObsTime',
        how='left'
    ).drop(columns=['ObsTime'])

    return merged_df
