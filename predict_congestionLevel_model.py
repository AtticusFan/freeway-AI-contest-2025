import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
import joblib # 用於載入 scaler 物件

# 1. 確保模型類別的定義與訓練時相同
class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, drop_prob=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                        batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def run_prediction(input_path, output_path, model_path, scaler_path, config):
    """
    從 JSON 檔案載入資料，執行壅塞程度預測，並將結果儲存為 JSON 檔案。

    Args:
        input_path (str): 輸入 JSON 檔案的路徑。
        output_path (str): 輸出 JSON 檔案的路徑。
        model_path (str): 已儲存的 .pth 模型檔案路徑。
        scaler_path (str): 已儲存的 .pkl scaler 檔案路徑。
        config (dict): 包含模型與資料設定的字典。
    """
    # 讀取設定
    seq_len = config['seq_len']
    features = config['features']
    num_features = len(features)
    num_classes = config['num_classes']
    horizon = config['horizon']

    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 載入輸入資料
    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    section_id = input_data['SectionID']
    df_data = pd.DataFrame(input_data['data'])
    
    if len(df_data) < seq_len:
        raise ValueError(f"輸入資料筆數為 {len(df_data)}，少於模型所需的序列長度 {seq_len}。")

    # 資料預處理
    # 選取最新的 seq_len 筆資料與必要特徵
    df_predict = df_data.tail(seq_len)[features]
    current_time = df_data['Timestamp'].iloc[-1]

    # 載入 scaler 並進行資料正規化
    scaler = joblib.load(scaler_path)
    scaled_data = scaler.transform(df_predict)
    
    # 初始化模型並載入權重
    model = GRUPredictor(input_size=num_features, hidden_size=64, num_layers=2, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 轉換為 Tensor
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device)

    # 執行預測
    with torch.no_grad():
        logits = model(input_tensor)
        prediction = logits.argmax(dim=1).item()
        
    # 建立輸出結果
    output_data = {
        "Section": section_id,
        "當前時間": current_time,
        f"{horizon}分鐘後壅塞程度": prediction
    }

    # 寫入 JSON 檔案
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print(f"預測完成，結果已儲存至：{output_path}")


# ====== 使用範例 ======
if __name__ == '__main__':
    # --- 參數設定 (需與訓練流程對應) ---
    MODEL_VERSION = 'v1' # 假設模型版本為 v1
    HORIZON = 5          # 預測 5 分鐘後
    
    CONFIG = {
        'seq_len': 30,
        'horizon': HORIZON,
        'num_classes': 5, # 假設壅塞等級分為 0, 1, 2, 3 四類
        'features': [
            'Occupancy', 'VehicleType_S_Volume', 'VehicleType_S_Speed',
            'VehicleType_L_Volume', 'VehicleType_L_Speed', 'StnPres',
            'Temperature', 'RH', 'WS', 'WD', 'WSGust', 'WDGust', 'Precip'
        ]
    }

    # --- 檔案路徑設定 ---
    # 重要：scaler 應於訓練模型後儲存，此處僅為示意
    # 例如在訓練腳本中加入: joblib.dump(scaler, 'scaler.pkl')
    SCALER_PATH = f'result/congestionLevel/{HORIZON}min/scaler_v{MODEL_VERSION}.pkl'
    MODEL_PATH = f'result/congestionLevel/{HORIZON}min/GRU_congestionLevel_{HORIZON}min_model_v{MODEL_VERSION}.pth'
    
    INPUT_JSON_PATH = 'input.json'
    OUTPUT_JSON_PATH = 'output.json'

    # --- 執行預測 ---
    try:
        run_prediction(
            input_path=INPUT_JSON_PATH,
            output_path=OUTPUT_JSON_PATH,
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            config=CONFIG
        )
    except FileNotFoundError as e:
        print(f"錯誤：找不到必要的檔案。請確認模型與 Scaler 路徑是否正確。\n{e}")
    except Exception as e:
        print(f"執行預測時發生錯誤：{e}")