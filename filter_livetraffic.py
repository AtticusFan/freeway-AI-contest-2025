import pandas as pd

def main():
    # 1. 讀取 CSV，將 'Timestamp' 欄位解析為帶時區的 datetime
    df = pd.read_csv(
        'livetraffic_data_sorted.csv',
        parse_dates=['Timestamp']
    )

    # 2. 定義篩選區間：2025-05-01 00:00:00+08:00 ～ 2025-05-06 23:59:59+08:00
    start = pd.Timestamp('2025-05-01 00:00:00+08:00')
    end   = pd.Timestamp('2025-05-06 23:59:00+08:00')

    # 3. 篩選 Timestamp 在此區間內的資料
    mask = (df['Timestamp'] >= start) & (df['Timestamp'] <= end)
    df = df.loc[mask]

    # 4. 去除指定欄位
    cols_to_drop = [
        'FileName',
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

    # 5. 輸出篩選後並刪除欄位的結果
    df.to_csv(
        'livetraffic_data_0501_0506.csv',
        index=False
    )

if __name__ == '__main__':
    main()
