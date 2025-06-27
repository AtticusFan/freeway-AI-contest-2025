# -------------------
# 必要的套件匯入
# -------------------
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import joblib 
import requests  
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque, defaultdict # 匯入 defaultdict 以簡化佇列的建立
import schedule               
import time                   
import threading
from flask import Flask, request, jsonify

# -------------------
# PyTorch 模型定義 (維持不變)
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
# 預測客戶端類別 (已修改)
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

        # === 修改核心 ===
        # 將單一佇列改為一個字典，用來儲存每個 SectionID 各自的佇列
        # defaultdict 會在嘗試存取一個不存在的鍵時，自動為其建立一個新的 deque
        self.section_data_queues = defaultdict(lambda: deque(maxlen=self.config['seq_len']))

    # predict_single, _make_request, send_prediction 方法維持不變
    def predict_single(self, sequence_data: List[Dict]) -> Dict[str, Any]:
        df = pd.DataFrame(sequence_data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values(by='Timestamp').reset_index(drop=True)
        features_data = df[self.config['features']]
        scaled_data = self.scaler.transform(features_data.values)
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_tensor).argmax(dim=1).item()
        return {"Section": df['SectionID'].iloc[0], "當前時間": df['Timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S'), f"{self.config['horizon']}分鐘後壅塞程度": prediction}
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == 'POST': response = self.session.post(url, json=data, timeout=10)
            else: raise ValueError(f"不支援的 HTTP 方法: {method}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"網路請求錯誤: {e}")
            raise
    def send_prediction(self, prediction_result: Dict) -> Dict[str, Any]:
        return self._make_request('POST', '/predictions', prediction_result)


# 天氣資料獲取功能 (維持不變)
def get_weather_data() -> Optional[Dict]:
    # ... 程式碼省略，與前一版相同 ...
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

# -------------------
# 全域實例與設定
# -------------------
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
    print(f"嚴重錯誤：找不到必要的模型或 Scaler 檔案，無法啟動服務。請檢查路徑。\n{e}")
    exit()

# -------------------
# Flask API 端點定義 (已修改)
# -------------------
@app.route('/api/traffic-data', methods=['POST'])
def receive_traffic_data():
    traffic_data = request.get_json()
    REQUIRED_FIELDS = ['Timestamp', 'SectionID', 'Occupancy', 'VehicleType_S_Volume', 'VehicleType_L_Volume', 'median_speed']
    if not traffic_data or any(field not in traffic_data for field in REQUIRED_FIELDS):
        return jsonify({"status": "error", "message": f"遺失必要欄位或無效JSON"}), 400

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 收到交通資料: {traffic_data}")

    weather_data = get_weather_data()
    if not weather_data:
        return jsonify({"status": "error", "message": "無法獲取天氣資料"}), 503

    new_data_point = {**traffic_data, **weather_data}
    print(f"new_data_point:")
    for key, value in new_data_point.items():
        print(f"  {key}: {value}")
    # === 修改核心 ===
    # 從傳入的資料中取得 SectionID
    section_id = traffic_data['SectionID']
    # 將資料點加入對應 SectionID 的專屬佇列中
    stateful_client.section_data_queues[section_id].append(new_data_point)
    
    print(f"資料已合併並存入 SectionID '{section_id}' 的佇列。")
    print(f"目前 SectionID '{section_id}' 的佇列長度: {len(stateful_client.section_data_queues[section_id])}")
    
    return jsonify({"status": "success", "message": "資料已成功接收並處理"}), 200

# -------------------
# 排程器相關函式 (已修改)
# -------------------
def predict_and_send_to_server(client: TrafficPredictionClient):
    """
    排程呼叫此函式，此函式會遍歷所有已知的 SectionID，
    並對每個資料已滿的 SectionID 佇列執行預測與傳送。
    """
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 排程觸發：檢查所有路段是否可進行預測...")
    
    # === 修改核心 ===
    # 遍歷字典中所有的 SectionID 和它們各自的佇列
    # 使用 list(client.section_data_queues.items()) 是為了避免在迭代中修改字典大小可能產生的問題
    for section_id, data_queue in list(client.section_data_queues.items()):
        
        # 檢查該路段的佇列是否已滿
        if len(data_queue) == client.config['seq_len']:
            print(f"--- 處理路段: {section_id} ---")
            
            # 執行預測
            sequence_to_predict = list(data_queue)
            prediction_result = client.predict_single(sequence_to_predict)
            print(f">>> 路段 '{section_id}' 的預測已產生:")
            print(json.dumps(prediction_result, ensure_ascii=False, indent=2))

            # 嘗試傳送結果
            print(f">>> 正在嘗試為路段 '{section_id}' 傳送至伺服器...")
            try:
                server_response = client.send_prediction(prediction_result)
                print(f">>> 成功為路段 '{section_id}' 傳送，回應如下:")
                print(json.dumps(server_response, ensure_ascii=False, indent=2))
            except Exception as e:
                print(f">>> 為路段 '{section_id}' 傳送時失敗: {e}")
        # else:
            # 可選：如果需要顯示未滿的佇列狀態，可以取消註解以下這行
            # print(f"--- 路段 '{section_id}' 資料不足 ({len(data_queue)}/{client.config['seq_len']})，跳過預測。 ---")


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
    app.run(host='0.0.0.0', port=5001)