import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import joblib 
import requests  
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque 
import schedule               
import time                   

class GRUPredictor(nn.Module):
    """
    定義 Gated Recurrent Unit (GRU) 模型架構。
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, drop_prob=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])
# 宣告全域變數
Timestamp = '2000-01-01 00:00:00'
SectionID = '23'  # 假設 SectionID 為 '23'
Occupancy = 0.0  # 初始佔用率為 0.0
VehicleType_S_Volume = 0.0  # 小型車流量
VehicleType_L_Volume = 0.0  # 大型車流量
median_speed = 0.0  # 中位速率
StnPres = 1013.25  # 氣壓 (hPa)
Temperature = 20.0  # 溫度 (°C)
RH = 50.0  # 相對濕度 (%)
WS = 0.0  # 風速 (m/s)
WD = 0.0  # 風向 (°)
WSGust = 0.0  # 突風風速 (m/s)
WDGust = 0.0  # 突風風向 (°)
Precip = 0.0  # 降水量 (mm)

# -------------------
# 預測客戶端類別 (功能完整版)
# -------------------
class TrafficPredictionClient:
    """
    交通預測客戶端。具備預測與傳送預測結果至伺服器的完整功能。
    """
    def __init__(self, model_path: str, scaler_path: str, config: Dict, base_url: str = "http://localhost:5000/api"):
        self.base_url = base_url
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.scaler = joblib.load(scaler_path)
        self.model = GRUPredictor(
            input_size=len(config['features']),
            hidden_size=64,
            num_layers=2,
            num_classes=config['num_classes']
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

        self.data_queue = deque(maxlen=self.config['seq_len'])

    def predict_single(self, sequence_data: List[Dict]) -> Dict[str, Any]:
        """對一個完整的資料序列執行預測。"""
        df = pd.DataFrame(sequence_data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values(by='Timestamp').reset_index(drop=True)
        features_data = df[self.config['features']]
        scaled_data = self.scaler.transform(features_data.values)
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_tensor).argmax(dim=1).item()
        return {
            "Section": df['SectionID'].iloc[0],
            "當前時間": df['Timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S'),
            f"{self.config['horizon']}分鐘後壅塞程度": prediction
        }

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """通用的 HTTP 請求方法。"""
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == 'POST':
                response = self.session.post(url, json=data, timeout=10) # 設定10秒超時
            else: # 可擴充 GET, PUT 等
                raise ValueError(f"不支援的 HTTP 方法: {method}")
            
            response.raise_for_status() # 若狀態碼不是 2xx，則拋出例外
            return response.json()
        except requests.exceptions.RequestException as e:
            # 捕捉所有 requests 可能的錯誤 (如連線、超時等)
            print(f"網路請求錯誤: {e}")
            raise # 將錯誤向上拋出，由呼叫者處理

    def send_prediction(self, prediction_result: Dict) -> Dict[str, Any]:
        """將預測結果發送到伺服器的 /predictions 端點。"""
        return self._make_request('POST', '/predictions', prediction_result)

def pretty_print(data: Dict[str, Any]) -> None:
    """以美化格式輸出 JSON (字典) 資料。"""
    print(json.dumps(data, ensure_ascii=False, indent=2))

def predict_and_send_to_server(client: TrafficPredictionClient):
    """
    此函式由排程器定時呼叫。
    功能：執行預測，然後將結果傳送到伺服器。
    """
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 排程觸發：準備執行預測並傳送...")
    
    if len(client.data_queue) < client.config['seq_len']:
        print(f"資料不足，取消本次傳送。目前 {len(client.data_queue)}/{client.config['seq_len']} 筆。")
        return

    # 執行預測
    sequence_to_predict = list(client.data_queue)
    prediction_result = client.predict_single(sequence_to_predict)
    print(">>> 預測已產生:")
    pretty_print(prediction_result)

    # 嘗試將結果傳送到伺服器
    print(">>> 正在嘗試傳送至伺服器...")
    try:
        server_response = client.send_prediction(prediction_result)
        print(">>> 成功傳送至伺服器，回應如下:")
        pretty_print(server_response)
    except Exception as e:
        # 如果 send_prediction 中發生任何錯誤 (主要為網路問題)，會在此捕捉
        # 這樣可以確保即使伺服器無回應，主程式也能繼續運作
        print(f">>> 傳送失敗 (此為正常現象，若伺服器未開啟): {e}")

# -------------------
# 主程式與示範
# -------------------
def scheduled_demo():
    """
    示範如何使用 `schedule` 函式庫來定時「預測並傳送」。
    """
    MODEL_VERSION = '4'
    HORIZON = 15
    CONFIG = {
        'seq_len': 30, 'horizon': HORIZON, 'num_classes': 5,
        'features': ['Occupancy', 'VehicleType_S_Volume', 'VehicleType_L_Volume', 'median_speed', 'StnPres', 'Temperature', 'RH', 'WS', 'WD', 'WSGust', 'WDGust', 'Precip']
    }
    # 換成實際路徑下的模型與 Scaler 檔案
    SCALER_PATH = f'../result/model/GRU_congestionLevel_{HORIZON}min_scaler_v{MODEL_VERSION}.pkl'
    MODEL_PATH = f'../result/model/GRU_congestionLevel_{HORIZON}min_model_v{MODEL_VERSION}.pth'

    try:
        stateful_client = TrafficPredictionClient(MODEL_PATH, SCALER_PATH, CONFIG)
        print("客戶端已初始化。\n")

        # 設定排程任務：每分鐘的第 15 秒，執行 predict_and_send_to_server 函式
        schedule.every().minute.at(":15").do(predict_and_send_to_server, client=stateful_client)
        print(f"排程已設定：每分鐘的第 15 秒執行預測並傳送。目前時間: {datetime.now().strftime('%H:%M:%S')}")

        print("\n--- 開始模擬即時資料流入 (每 5 秒一筆) ---")
        start_time = datetime.now() - timedelta(minutes=200)
        data_point_index = 0
        last_data_generation_time = time.time()

        while True:
            # 每隔 5 秒模擬一次新資料的產生
            if time.time() - last_data_generation_time >= 5:
                current_timestamp = start_time + timedelta(seconds=data_point_index * 5)
                new_data_point = {
                    'Timestamp': current_timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'SectionID': '23',
                    'Occupancy': 0.2 + 0.15 * np.random.random(), 'VehicleType_S_Volume': 120 + 60 * np.random.random(),
                    'VehicleType_L_Volume': 15 + 15 * np.random.random(), 'median_speed': 70 + 20 * np.random.random(),
                    'StnPres': 1010 + 10 * np.random.random(), 'Temperature': 28 + 5 * np.random.random(),
                    'RH': 70 + 15 * np.random.random(), 'WS': 4 + 4 * np.random.random(), 'WD': 150 + 100 * np.random.random(),
                    'WSGust': 7 + 6 * np.random.random(), 'WDGust': 180 + 100 * np.random.random(), 'Precip': 0.05 * np.random.random()
                }
                
                stateful_client.data_queue.append(new_data_point)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 收到新資料... (佇列長度: {len(stateful_client.data_queue)})")
                
                data_point_index += 1
                last_data_generation_time = time.time()

            # 執行排程器檢查是否有待辦任務需要執行
            schedule.run_pending()
            
            # 暫停 1 秒，以降低 CPU 使用率，避免空轉
            time.sleep(1)

    except FileNotFoundError as e:
        print(f"\n錯誤：找不到必要的檔案。請確認模型與 Scaler 的路徑是否正確。\n{e}")
    except KeyboardInterrupt:
        print("\n程式被使用者中斷。正在結束...")
    except Exception as e:
        print(f"\n執行過程中發生未預期的錯誤: {e}")

if __name__ == "__main__":
    scheduled_demo()