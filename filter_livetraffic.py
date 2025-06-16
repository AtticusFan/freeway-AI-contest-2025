import pandas as pd

def main():
    # 1. 三個要合併的檔案路徑（可按時間順序或任意順序，之後會再統一排序）
    files = [
        'datasets/integrated_Apr.csv',
        'datasets/integrated_May_later.csv',
        'datasets/integrated_vd_livetraffic_data_sorted.csv'
    ]

    # 2. 讀取並解析 Timestamp 欄位
    dfs = []
    for f in files:
        df = pd.read_csv(f, parse_dates=['Timestamp'])
        dfs.append(df)

    # 3. 合併並依 Timestamp 排序
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('Timestamp')

    # 4. 刪除不需要的欄位
    cols_to_drop = [
        'FileName',
        'Status'
        'LinkID',
        'LaneID',
        'LaneType',
        'LaneSpeed',
        'CongestionLevelID',
        'HasHistorical',
        'HasVD',
        'HasAVI',
        'HasETAG',
        'HasGVP',
        'HasCVP',
        'HasOthers'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 5. 輸出最終結果
    df.to_csv('datasets/vd_livetraffic_data_0401_0610.csv', index=False)

if __name__ == '__main__':
    main()
