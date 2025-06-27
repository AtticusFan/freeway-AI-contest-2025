import requests
import pandas as pd
import json
from datetime import datetime

# API設定
BASE_URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore"
RESOURCE_ID = "O-A0001-001"  # 自動氣象站-氣象觀測資料
AUTHORIZATION = "CWA-6A873EC6-E09A-45E8-9962-D10BD5C543DA"

def get_weather_data(station_id="466920"):
    """
    從中央氣象署API獲取三重測站天氣資料

    Parameters:
    station_id (str): 測站ID, 設為三重測站

    Returns:
    pandas.DataFrame: 包含天氣資料的DataFrame
    """

    # 建構API請求URL
    url = f"{BASE_URL}/{RESOURCE_ID}"

    # 設定請求參數 - 使用locationName篩選三重測站
    params = {
        "Authorization": AUTHORIZATION,
        "format": "JSON",
#        "locationName": "三重"
    }

    try:
        print("正在從中央氣象署API獲取資料...")

        # 發送API請求
        response = requests.get(url, params=params)
        response.raise_for_status()  # 檢查HTTP錯誤

        # 解析JSON回應
        data = response.json()

        # 檢查API回應狀態
        if not data.get("success"):
            print("API回應失敗")
            return None

        # 提取觀測資料
        stations = data["records"]["Station"]

        # 建立資料列表
        weather_records = []

        for station in stations:
            if station.get("StationName") != "三重":
              continue

            # 取得天氣觀測資料
            weather_element = station.get("WeatherElement", {})

            # 處理陣風資訊
            gust_info = weather_element.get("GustInfo", {})

            # 處理降水量
            now_info = weather_element.get("Now", {})

            # 初始化天氣數據字典
            weather_data = {
                "ObsTime": station.get("ObsTime", {}).get("DateTime", ""),
                "StnPres": weather_element.get("AirPressure", ""),
                "Temperature": weather_element.get("AirTemperature", ""),
                "RH": weather_element.get("RelativeHumidity", ""),
                "WS": weather_element.get("WindSpeed", ""),
                "WD": weather_element.get("WindDirection", ""),
                "WSGust": gust_info.get("PeakGustSpeed", ""),
                "WDGust": gust_info.get("PeakGustDirection", ""),
                "Precip": now_info.get("Precipitation", "")
            }


            # 天氣資料
            record = weather_data
            weather_records.append(record)

        # 轉換為DataFrame
        df = pd.DataFrame(weather_records)

        print(f"成功獲取三重測站的天氣即時資料")
        print(json.dumps(weather_records, indent=2, ensure_ascii=False))
        return df

    except requests.exceptions.RequestException as e:
        print(f"API請求失敗: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON解析失敗: {e}")
        return None
    except Exception as e:
        print(f"發生錯誤: {e}")
        return None

# 主程式執行
if __name__ == "__main__":
    print("=" * 50)

    # 獲取三重測站天氣資料
    weather_df = get_weather_data()

    if weather_df is not None:

        # 顯示主要天氣資訊
        print("\n=== 三重測站即時天氣資料欄位 ===")
        if not weather_df.empty:
            row = weather_df.iloc[0]  # 取第一筆（應該只有一筆）
            print(f"觀測時間: {row.get('ObsTime', 'N/A')}")
            print(f"氣壓: {row.get('StnPres', 'N/A')} hPa")
            print(f"氣溫: {row.get('Temperature', 'N/A')} °C")
            print(f"相對濕度: {row.get('RH', 'N/A')} %")
            print(f"風速: {row.get('WS', 'N/A')} m/s")
            print(f"風向: {row.get('WD', 'N/A')} °")
            print(f"最大瞬間風速: {row.get('WSGust', 'N/A')} m/s")
            print(f"最大瞬間風向: {row.get('WDGust', 'N/A')} °")
            print(f"降水量: {row.get('Precip', 'N/A')} mm")