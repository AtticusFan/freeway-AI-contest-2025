#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def main():
    # 1. 建立 VDID → SectionID 的對應字典
    mapping = {
        # SectionID 23
        'VD-N1-S-26.750-N-LOOP': 23,
        'VD-N1-S-27-I-ES-1-三重': 23,
        'VD-N1-S-27-O-NE-22-三重': 23,
        'VD-N1-S-27-O-NW-1-三重': 23,
        'VD-N1-S-26.350-M-RS': 23,
        # SectionID 24
        'VD-N1-N-27-I-EN-1-三重': 24,
        'VD-N1-N-27.100-N-LOOP': 24,
        'VD-N1-N-27-O-SE-1-三重': 24,
        'VD-N1-N-26.780-M-RS': 24,
        # SectionID 25
        'VD-N1-S-28.840-M-RS': 25,
        'VD-N1-S-32.080-N-LOOP': 25,
        'VD-N1-S-31.400-M-LOOP': 25,
        'VD-N1-S-30.540-M-LOOP': 25,
        # SectionID 26
        'VD-N1-N-30.550-M-RS': 26,
        'VD-N1-N-31.830-N-LOOP': 26,
        'VD-N1-N-27.970-M-LOOP': 26,
        'VD-N1-N-28.670-M-LOOP': 26,
        # SectionID 28
        'VD-N1-N-32.200-M-LOOP': 28,
    }

    # 2. 讀取 vdlive 資料並解析 Timestamp
    vdlive = pd.read_csv(
        'vdlive_data_may_sorted.csv',
        parse_dates=['Timestamp']
    )

    # 3. 篩選出需要的 VDID，並對應 SectionID
    vdlive = vdlive[vdlive['VDID'].isin(mapping)].copy()
    vdlive['SectionID'] = vdlive['VDID'].map(mapping)

    # 4. 刪除不需要的欄位
    drop_cols = ['FileName', 'VDID', 'LinkID', 'Status', 'LaneID', 'LaneType']
    vdlive = vdlive.drop(columns=[c for c in drop_cols if c in vdlive.columns])

    # 5. 計算同一 SectionID & Timestamp 下各數值欄位的中位數
    agg_cols = [c for c in vdlive.columns if c not in ['SectionID', 'Timestamp']]
    vdlive_median = (
        vdlive
        .groupby(['SectionID', 'Timestamp'], as_index=False)[agg_cols]
        .median()
    )

    # 6. 讀取已合併氣象的路況資料
    merged = pd.read_csv(
        'merged_traffic_weather_0501_0506.csv',
        parse_dates=['Timestamp']
    )

    # 7. 以 SectionID & Timestamp 為 key，將 vdlive_median 併入 merged
    final = merged.merge(
        vdlive_median,
        on=['SectionID', 'Timestamp'],
        how='left'
    )

    # 8. 輸出最終合併結果
    final.to_csv(
        'final_merged_traffic_weather_vdlive_0501_0506.csv',
        index=False
    )

if __name__ == '__main__':
    main()
