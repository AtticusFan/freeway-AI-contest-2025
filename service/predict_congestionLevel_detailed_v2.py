import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import joblib 
import requests  
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque 
import schedule               
import time                   
import threading
from flask import Flask, request, jsonify

# -------------------
# GRU 模型定義
# -------------------
class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, drop_prob=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# -------------------
# 預測功能
# -------------------
class TrafficPredictionClient:
    def __init__(self, model_path: str, scaler_path: str, config: Dict, base_url: str = "http://localhost:5000/api"):
        self.base_url = base_url
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = joblib.load(scaler_path)
        self.model = GRUPredictor(input_size=len(config['features']), hidden_size=64, num_layers=2, num_classes=config['num_classes']).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        self.data_queue = deque(maxlen=self.config['seq_len'])

    def predict_single(self, sequence_data: List[Dict]) -> Dict[str, Any]:
        df = pd.DataFrame(sequence_data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values(by='Timestamp').reset_index(drop=True)
        print(f"預測資料長度: {len(df)}，欄位: {list(df.columns)}")
        features_data = df[self.config['features']]
        scaled_data = self.scaler.transform(features_data.values)
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_tensor).argmax(dim=1).item()
        return {"Section": df['SectionID'].iloc[0], "當前時間": df['Timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S'), f"{self.config['horizon']}分鐘後壅塞程度": prediction+1}

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == 'POST':
                response = self.session.post(url, json=data, timeout=10)
            else:
                raise ValueError(f"不支援的 HTTP 方法: {method}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"網路請求錯誤: {e}")
            raise

    def send_prediction(self, prediction_result: Dict) -> Dict[str, Any]:
        """將預測結果傳送至伺服器。"""
        return self._make_request('POST', '/predictions', prediction_result)

# ---------------------------------------------
# 即時天氣資料獲取功能
# ---------------------------------------------
def get_weather_data() -> Optional[Dict]:
    """從中央氣象署API獲取三重測站的單筆最新天氣資料。"""
    BASE_URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore"
    RESOURCE_ID = "O-A0001-001"
    AUTHORIZATION = "CWA-6A873EC6-E09A-45E8-9962-D10BD5C543DA"
    params = {"Authorization": AUTHORIZATION, "format": "JSON"}
    try:
        response = requests.get(f"{BASE_URL}/{RESOURCE_ID}", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data.get("success"): return None
        for station in data["records"]["Station"]:
            if station.get("StationName") == "三重":
                weather_element = station.get("WeatherElement", {})
                gust_info = weather_element.get("GustInfo", {})
                now_info = weather_element.get("Now", {})
                def to_float(value, default=0.0):
                    try: return float(value)
                    except (ValueError, TypeError): return default
                return {"StnPres": to_float(weather_element.get("AirPressure")), "Temperature": to_float(weather_element.get("AirTemperature")), "RH": to_float(weather_element.get("RelativeHumidity")), "WS": to_float(weather_element.get("WindSpeed")), "WD": to_float(weather_element.get("WindDirection")), "WSGust": to_float(gust_info.get("PeakGustSpeed")), "WDGust": to_float(gust_info.get("PeakGustDirection")), "Precip": to_float(now_info.get("Precipitation"))}
        return None
    except requests.exceptions.RequestException as e:
        print(f"天氣 API 請求失敗: {e}")
        return None
    except Exception as e:
        print(f"獲取天氣資料時發生未預期錯誤: {e}")
        return None


app = Flask(__name__)

MODEL_VERSION = '4'
HORIZON = 15
CONFIG = {
    'seq_len': 30, 'horizon': HORIZON, 'num_classes': 5,
    'features': ['Occupancy', 'VehicleType_S_Volume', 'VehicleType_L_Volume', 'median_speed', 'StnPres', 'Temperature', 'RH', 'WS', 'WD', 'WSGust', 'WDGust', 'Precip']
}
SCALER_PATH = f'../result/model/GRU_congestionLevel_{HORIZON}min_scaler_v{MODEL_VERSION}.pkl'
MODEL_PATH = f'../result/model/GRU_congestionLevel_{HORIZON}min_model_v{MODEL_VERSION}.pth'

try:
    stateful_client = TrafficPredictionClient(MODEL_PATH, SCALER_PATH, CONFIG)
    print("客戶端已成功初始化。")
except FileNotFoundError as e:
    print(f"嚴重錯誤：找不到必要的模型或 Scaler 檔案，無法啟動服務。請檢查路徑。")
    print(e)
    exit()

# -------------------
# Flask API 端點定義
# -------------------
# 接收交通資料的 API 端點
@app.route('/api/traffic-data', methods=['POST'])
def receive_traffic_data():
    """
    接收外部傳入的交通資料，並觸發天氣資料獲取、合併與儲存。
    """
    # 接收並驗證傳入的交通資料
    traffic_data = request.get_json()
    if not traffic_data:
        return jsonify({"status": "error", "message": "無效的 JSON 資料或 Content-Type 標頭不正確"}), 400

    # 定義必要欄位
    REQUIRED_FIELDS = ['Timestamp', 'SectionID', 'Occupancy', 'VehicleType_S_Volume', 'VehicleType_L_Volume', 'median_speed']
    
    # 檢查所有必要欄位是否存在
    missing_fields = [field for field in REQUIRED_FIELDS if field not in traffic_data]
    if missing_fields:
        return jsonify({"status": "error", "message": f"遺失必要欄位: {', '.join(missing_fields)}"}), 400

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 收到交通資料: {traffic_data}")

    # 獲取天氣資料
    weather_data = get_weather_data()
    if not weather_data:
        print("警告：無法獲取即時天氣資料，本次資料點將不會被儲存。")
        return jsonify({"status": "error", "message": "無法獲取天氣資料"}), 503

    # 合併資料
    new_data_point = {
        **traffic_data,
        **weather_data
    }

    # 將合併後的完整資料點加入佇列
    stateful_client.data_queue.append(new_data_point)
    print(f"資料已合併並存入佇列。目前佇列長度: {len(stateful_client.data_queue)}")
    
    # 回應成功訊息
    return jsonify({"status": "success", "message": "資料已成功接收並處理"}), 200

# -------------------
# 排程器相關函式 (維持不變)
# -------------------
def predict_and_send_to_server(client: TrafficPredictionClient):
    """排程呼叫此函式，執行預測並傳送。"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 排程觸發：準備執行預測並傳送...")
    if len(client.data_queue) < client.config['seq_len']:
        print(f"資料不足，取消本次傳送。目前 {len(client.data_queue)}/{client.config['seq_len']} 筆。")
        return
    sequence_to_predict = list(client.data_queue)
    prediction_result = client.predict_single(sequence_to_predict)
    print(">>> 預測已產生:")
    print(json.dumps(prediction_result, ensure_ascii=False, indent=2))
    print(">>> 正在嘗試傳送至伺服器...")
    try:
        server_response = client.send_prediction(prediction_result)
        print(">>> 成功傳送至伺服器，回應如下:")
        print(json.dumps(server_response, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f">>> 傳送失敗: {e}")

def run_scheduler():
    """在獨立的執行緒中執行排程迴圈。"""
    print("背景排程器已啟動。")
    while True:
        schedule.run_pending()
        time.sleep(1)

# -------------------
# 主程式進入點 (維持不變)
# -------------------
if __name__ == "__main__":
    schedule.every().minute.at(":15").do(predict_and_send_to_server, client=stateful_client)
    print(f"排程已設定：每分鐘的第 15 秒執行預測並傳送。")

    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()

    print("正在啟動 Flask 伺服器，監聽埠號 5001...")
    print("請使用 POST請求 將交通資料傳送至 http://<YOUR_IP>:5001/api/traffic-data")
    app.run(host='0.0.0.0', port=5001)