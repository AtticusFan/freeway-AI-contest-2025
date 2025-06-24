import pandas as pd

# 設定檔案路徑
file_path = 'datasets/0401_0610/merged_traffic_weather_geometry_0401_0610.csv'

try:
    # 讀取 CSV 檔案至 DataFrame
    df = pd.read_csv(file_path)

    # 檢查整個資料框是否含有任何空值
    if df.isnull().values.any():
        print("偵測到空值。")
        
        # 計算每個欄位的空值數量並僅顯示包含空值的欄位
        null_counts = df.isnull().sum()
        print("\n各欄位空值數量：")
        print(null_counts[null_counts > 0])
    else:
        print("資料集中無任何空值。")

except FileNotFoundError:
    print(f"錯誤：檔案 '{file_path}' 不存在。請確認檔案路徑與名稱的正確性。")
except Exception as e:
    print(f"處理檔案時發生錯誤：{e}")