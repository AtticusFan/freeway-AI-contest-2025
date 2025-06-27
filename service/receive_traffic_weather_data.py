# server.py
import requests
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# --- Flask 應用程式初始化 ---
app = Flask(__name__)
CORS(app)  # 允許跨域請求

# --- 中央氣象署 API 設定 ---
CWA_BASE_URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore"
CWA_RESOURCE_ID = "O-A0001-001"  # 自動氣象站-氣象觀測資料
CWA_AUTHORIZATION = "CWA-6A873EC6-E09A-45E8-9962-D10BD5C543DA" # 您的 API 金鑰
TARGET_STATION_NAME = "三重" # 目標觀測站

def get_weather_data():
    """
    從中央氣象署 API 獲取指定測站的即時天氣資料。

    Returns:
        dict: 包含天氣資料的字典，若失敗則返回 None。
    """
    url = f"{CWA_BASE_URL}/{CWA_RESOURCE_ID}"
    params = {
        "Authorization": CWA_AUTHORIZATION,
        "format": "JSON",
        "stationName": TARGET_STATION_NAME,
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            print("天氣 API 回應失敗")
            return None

        stations = data["records"]["Station"]
        if not stations:
            print(f"在 API 回應中找不到測站 '{TARGET_STATION_NAME}' 的資料")
            return None

        for station in stations:
            if station.get("StationName") == TARGET_STATION_NAME:
                weather_element = station.get("WeatherElement", {})
                gust_info = weather_element.get("GustInfo", {})
                now_info = weather_element.get("Now", {})

                weather_data = {
                    # 此處 ObsTime 的格式是 {'DateTime': 'YYYY-MM-DDTHH:MM:SS+08:00'}
                    "ObsTime": station.get("ObsTime", {}).get("DateTime", ""),
                    "StnPres": weather_element.get("AirPressure", -99),
                    "Temperature": weather_element.get("AirTemperature", -99),
                    "RH": weather_element.get("RelativeHumidity", -99),
                    "WS": weather_element.get("WindSpeed", -99),
                    "WD": weather_element.get("WindDirection", -99),
                    "WSGust": gust_info.get("PeakGustSpeed", -99),
                    "WDGust": gust_info.get("PeakGustDirection", -99),
                    "Precip": now_info.get("Precipitation", -99)
                }
                return weather_data
        
        print(f"指定的測站 '{TARGET_STATION_NAME}' 資料未找到。")
        return None

    except requests.exceptions.RequestException as e:
        print(f"天氣 API 請求失敗: {e}")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"解析天氣 API 回應時發生錯誤: {e}")
        return None

@app.route('/api/process-data', methods=['POST'])
def process_traffic_and_weather_data():
    """
    接收路況資料，結合天氣資料，並回應合併後的結果。
    """
    traffic_data = request.get_json()
    required_fields = ["TimeStamp", "Occupancy", "Vehicle_Median_Speed", "VehicleType_S_Volume", "VehicleType_L_Volume"]
    
    if not traffic_data or not all(field in traffic_data for field in required_fields):
        return jsonify({
            "success": False,
            "message": "請求的 JSON 主體缺少必要的路況資料欄位。"
        }), 400

    print("正在獲取天氣資料...")
    weather_data = get_weather_data()
    
    if weather_data is None:
        return jsonify({
            "success": False,
            "message": "無法從外部 API 獲取天氣資料，請檢查伺服器日誌。"
        }), 503

    # --- 主要修改處 ---
    # 根據需求，從天氣資料中移除觀測時間欄位，僅保留路況的 TimeStamp
    weather_data.pop("ObsTime", None)

    print("成功獲取天氣資料。")

    # 合併路況與天氣資料
    combined_data = {**traffic_data, **weather_data}

    print("資料合併完成。")

    return jsonify({
        "success": True,
        "message": "路況與天氣資料成功合併",
        "data": combined_data
    }), 200

# --- 錯誤處理 ---
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "message": "API 端點不存在"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "message": "伺服器內部錯誤"
    }), 500

# --- 主程式執行 ---
if __name__ == '__main__':
    print("Flask API Server 啟動中...")
    print("API 端點:")
    print("  POST   /api/process-data  - 接收路況資料，結合天氣資料後回應")
    print("\n伺服器運行在 http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)