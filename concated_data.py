#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def main():
    # 1. 讀取已篩選過的路況資料（Timestamp 已含 +08:00 時區）
    traffic = pd.read_csv(
        'datasets/0501_0610/vd_livetraffic_data_0501_0610.csv',
        parse_dates=['Timestamp']
    )

    # 2. 讀取已填補過的氣象資料，解析 ObsTime 為 datetime 並指定台北時區
    weather = pd.read_csv(
        'datasets/0501_0610/combined_weather_data_0501_0610.csv',
        parse_dates=['ObsTime']
    )
    weather['ObsTime'] = weather['ObsTime'].dt.tz_localize('Asia/Taipei')

    # 3. 在 traffic 裡建立對應的觀測時段鍵（整點往前取一小時段 +1 小時）
    #    讓 00:00–00:59 的資料對應到 01:00 的 ObsTime
    traffic['ObsTime'] = (
        traffic['Timestamp']
        .dt.tz_convert('Asia/Taipei')
        .dt.floor('h')
        + pd.Timedelta(hours=1)
    )

    # 4. 以 ObsTime 為 key 將 weather 的所有欄位併入 traffic
    merged = traffic.merge(
        weather,
        on='ObsTime',
        how='left'
    )

    # 5. 若不需要 ObsTime 欄位，可將它移除
    merged = merged.drop(columns=['ObsTime'])

    # 6. 輸出合併後的結果
    merged.to_csv(
        'datasets/0501_0610/merged_traffic_weather_0501_0610.csv',
        index=False
    )

if __name__ == '__main__':
    main()
