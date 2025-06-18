import pandas as pd

def main():
    files = [
        'datasets/integrated_May_later.csv',
        'datasets/integrated_vd_livetraffic_data_sorted.csv'
    ]

    dfs = []
    for f in files:
        # 1. 讀取為字串
        df = pd.read_csv(f, dtype={'Timestamp': str})
        # 2. 統一格式並解析為帶時區的 datetime
        df['Timestamp'] = pd.to_datetime(
            df['Timestamp'].str.replace('T', ' '),
            format='%Y-%m-%d %H:%M:%S%z'
        )
        dfs.append(df)

    # 3. 合併並排序
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('Timestamp')

    # 4. 時間篩選：2025-05-01 00:00:00+08:00 ～ 2025-06-10 23:59:00+08:00
    start = pd.Timestamp('2025-05-01 00:00:00+08:00')
    end   = pd.Timestamp('2025-06-10 23:59:00+08:00')
    df = df[(df['Timestamp'] >= start) & (df['Timestamp'] <= end)]

    # 5. 刪除欄位（已修正逗號）
    cols_to_drop = [
        'FileName',
        'Status',
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

    # 6. 輸出
    df.to_csv('datasets/train_data/vd_livetraffic_data_0501_0610.csv', index=False)

if __name__ == '__main__':
    main()
