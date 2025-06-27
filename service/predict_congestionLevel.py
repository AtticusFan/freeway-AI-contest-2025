import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import joblib
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, drop_prob=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class TrafficPredictionClient:
    def __init__(self, model_path: str, scaler_path: str, config: Dict, base_url: str = "http://localhost:5000/api"):
        self.base_url = base_url
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 載入模型和標準化器
        self.scaler = joblib.load(scaler_path)
        self.model = GRUPredictor(
            input_size=len(config['features']),
            hidden_size=64,
            num_layers=2,
            num_classes=config['num_classes']
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # 設定 HTTP 客戶端
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
    
    def predict_single(self, sequence_data: List[Dict]) -> Dict[str, Any]:
        """
        對單一序列資料進行預測
        
        Args:
            sequence_data: 包含時間序列資料的字典列表，每個字典包含特徵欄位
        
        Returns:
            預測結果字典
        """
        seq_len = self.config['seq_len']
        features = self.config['features']
        horizon = self.config['horizon']
        
        if len(sequence_data) != seq_len:
            raise ValueError(f"輸入序列長度必須為 {seq_len}，實際為 {len(sequence_data)}")
        
        df = pd.DataFrame(sequence_data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values(by='Timestamp').reset_index(drop=True)
        
        # 取得當前時間（序列的最後一個時間點）
        current_time = df['Timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
        section_id = df['SectionID'].iloc[0] if 'SectionID' in df.columns else "未知"
        
        # 準備特徵資料
        features_data = df[features]
        scaled_data = self.scaler.transform(features_data.values)
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 進行預測
        with torch.no_grad():
            logits = self.model(input_tensor)
            prediction = logits.argmax(dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()
        
        # 預測結果
        prediction_result = {
            "Section": section_id,
            "當前時間": current_time,
            f"{horizon}分鐘後壅塞程度": prediction,
            # "預測信心度": round(confidence, 4),
            # "預測時間戳": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        print(f"預測結果: {prediction_result}")
        return prediction_result
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """發送 HTTP 請求的通用方法"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data)
            elif method.upper() == 'PUT':
                response = self.session.put(url, json=data)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url)
            else:
                raise ValueError(f"不支援的 HTTP 方法: {method}")
            
            result = response.json()
            
            if not response.ok:
                raise requests.exceptions.HTTPError(
                    f"HTTP {response.status_code}: {result.get('message', '請求失敗')}"
                )
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"請求錯誤: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"JSON 解析錯誤: {e}")
            raise
    
    def send_prediction(self, sequence_data: List[Dict]) -> Dict[str, Any]:
        """
        進行預測並將結果發送至伺服器
        
        Args:
            sequence_data: 時間序列資料
            
        Returns:
            伺服器回應
        """
        # 進行預測
        prediction_result = self.predict_single(sequence_data)
        
        # 發送至伺服器
        response = self._make_request('POST', '/predictions', prediction_result)
        
        return response
    
    def get_prediction_history(self, section_id: Optional[str] = None) -> Dict[str, Any]:
        """取得預測歷史記錄"""
        endpoint = '/predictions'
        if section_id:
            endpoint += f'?section_id={section_id}'
        
        return self._make_request('GET', endpoint)
    
    def get_model_info(self) -> Dict[str, Any]:
        """取得模型資訊"""
        model_info = {
            "模型配置": self.config,
            "特徵欄位": self.config['features'],
            "序列長度": self.config['seq_len'],
            "預測範圍": f"{self.config['horizon']}分鐘",
            "分類數量": self.config['num_classes'],
            "設備": str(self.device)
        }
        return model_info

def pretty_print(data: Dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))

def demo():
    MODEL_VERSION = '4'
    HORIZON = 15
    
    CONFIG = {
        'seq_len': 30,
        'horizon': HORIZON,
        'num_classes': 5, 
        'features': [
            'Occupancy', 'VehicleType_S_Volume',
            'VehicleType_L_Volume', 
            'median_speed',
            'StnPres',
            'Temperature', 'RH', 'WS', 'WD', 'WSGust', 'WDGust', 'Precip'
        ]
    }

    SCALER_PATH = f'../result/model/GRU_congestionLevel_{HORIZON}min_scaler_v{MODEL_VERSION}.pkl'
    MODEL_PATH = f'../result/model/GRU_congestionLevel_{HORIZON}min_model_v{MODEL_VERSION}.pth'
    
    try:
        # 初始化
        client = TrafficPredictionClient(MODEL_PATH, SCALER_PATH, CONFIG)
        
        print("=== 交通壅塞預測 API 客戶端示範 ===\n")
        
        print("1. 模型資訊:")
        model_info = client.get_model_info()
        pretty_print(model_info)
        
        # 2. 載入測試資料（需要有 30 筆連續資料，這裡要準備實際的資料
        print("\n2. 載入測試序列資料...")
        
        # 示範用的虛擬資料結構
        sample_sequence = []
        base_time = datetime.now()
        
        for i in range(30):  # 產生 30 筆測試資料
            timestamp = base_time - timedelta(minutes=(29-i)*5)
            sample_data = {
                'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'SectionID': '23',
                'Occupancy': 0.3 + 0.1 * np.random.random(),
                'VehicleType_S_Volume': 150 + 50 * np.random.random(),
                'VehicleType_L_Volume': 20 + 10 * np.random.random(),
                'median_speed': 60 + 20 * np.random.random(),
                'StnPres': 1013.25 + 5 * np.random.random(),
                'Temperature': 25 + 10 * np.random.random(),
                'RH': 60 + 20 * np.random.random(),
                'WS': 5 + 3 * np.random.random(),
                'WD': 180 + 90 * np.random.random(),
                'WSGust': 8 + 5 * np.random.random(),
                'WDGust': 200 + 80 * np.random.random(),
                'Precip': 0.1 * np.random.random()
            }
            sample_sequence.append(sample_data)
        
        print("測試序列資料已生成")
        
        # 進行單筆預測
        print("\n3. 進行單筆預測:")
        prediction_result = client.predict_single(sample_sequence)
        pretty_print(prediction_result)
        
        # 4. 發送預測結果至伺服器（需要伺服器端支援）
        print("\n4. 發送預測結果至伺服器:")
        try:
            server_response = client.send_prediction(sample_sequence)
            pretty_print(server_response)
        except Exception as e:
            print(f"伺服器連線失敗（這是正常的，因為示範環境沒有實際的伺服器）: {e}")
        
        # 5. 嘗試取得預測歷史（需要伺服器端支援）
        print("\n5. 取得預測歷史:")
        try:
            history = client.get_prediction_history('section23')
            pretty_print(history)
        except Exception as e:
            print(f"無法取得預測歷史（這是正常的，因為示範環境沒有實際的伺服器）: {e}")
        
    except FileNotFoundError as e:
        print(f"錯誤：找不到必要的檔案。請確認模型與 Scaler 路徑是否正確。\n{e}")
    except Exception as e:
        print(f"執行預測時發生錯誤：{e}")

if __name__ == "__main__":
    demo()