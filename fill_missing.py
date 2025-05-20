import pandas as pd

def try_numeric(col):
    """
    嘗試將欄位轉為數值型態，若失敗就原樣回傳
    """
    try:
        return pd.to_numeric(col)
    except (ValueError, TypeError):
        return col

def main():
    # 1. 讀取 CSV，將 "--" 視為缺失值 (NaN)
    df = pd.read_csv(
        'combined_weather_data_0501_0506.csv',
        na_values='--'
    )

    # 2. 嘗試轉換每個欄位為 numeric（失敗則保留原樣）
    df = df.apply(try_numeric)

    # 3. 對所有數值欄位做線性插值 (前後值平均) 填補缺失值
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].interpolate(
        method='linear',
        limit_direction='both'
    )

    # 4. 輸出填補後的 CSV
    df.to_csv(
        'combined_weather_data_0501_0506_filled.csv',
        index=False
    )

if __name__ == '__main__':
    main()