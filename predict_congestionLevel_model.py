import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import joblib
import os

# 1. GRUPredictor 類別定義 (與訓練時相同)
class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, drop_prob=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                        batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def run_batch_prediction(input_path, output_path, model_path, scaler_path, config):
    """
    從 JSON 檔案載入資料，使用滑動窗口對整個資料集進行批次預測，
    並將所有預測結果直接覆寫至輸出 JSON 檔案。
    """
    # --- 1. 載入設定與模型 ---
    seq_len = config['seq_len']
    features = config['features']
    horizon = config['horizon']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 載入 scaler
    scaler = joblib.load(scaler_path)
    
    # 初始化模型並載入權重
    model = GRUPredictor(
        input_size=len(features),
        hidden_size=64,
        num_layers=2,
        num_classes=config['num_classes']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- 2. 讀取並處理輸入資料 ---
    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    section_id = input_data['SectionID']
    df_data = pd.DataFrame(input_data['data'])

    # ==================== 第 1 項修正：強制排序 ====================
    # 將 'Timestamp' 欄位轉換為 pandas 的 datetime 物件格式
    df_data['Timestamp'] = pd.to_datetime(df_data['Timestamp'])
    # 依照 'Timestamp' 欄位進行排序，並重設索引
    df_data = df_data.sort_values(by='Timestamp').reset_index(drop=True)
    # =============================================================
    
    if len(df_data) < seq_len:
        print(f"錯誤：輸入資料筆數為 {len(df_data)}，不足以構成一個長度為 {seq_len} 的序列。")
        return

    # --- 3. 執行滑動窗口預測 ---
    newly_predicted_results = []
    total_predictions = len(df_data) - seq_len + 1
    print(f"資料已排序，將對 {total_predictions} 筆序列進行預測...")

    with torch.no_grad():
        for i in range(total_predictions):
            window_df = df_data.iloc[i : i + seq_len]
            current_time = window_df['Timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
            features_data = window_df[features]
            scaled_data = scaler.transform(features_data.values)
            input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(input_tensor)
            prediction = logits.argmax(dim=1).item()
            
            output_data = {
                "Section": section_id,
                "當前時間": current_time,
                f"{horizon}分鐘後壅塞程度": prediction
            }
            newly_predicted_results.append(output_data)

    print(f"批次預測完成，共產生 {len(newly_predicted_results)} 筆新結果。")

    # ==================== 第 2 項修正：改為覆寫模式 ====================
    # 直接將新產生的結果列表寫入檔案，若檔案已存在則會被覆蓋
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(newly_predicted_results, f, ensure_ascii=False, indent=4)
    # ===============================================================
    
    print(f"所有結果已「覆寫」至：{output_path}")

# ====== 使用範例 ======
if __name__ == '__main__':
    # --- 參數設定 (需與訓練流程對應) ---
    MODEL_VERSION = '3' # 請根據您的模型版本修改
    HORIZON = 5
    
    CONFIG = {
        'seq_len': 30,
        'horizon': HORIZON,
        'num_classes': 6, 
        'features': [
            'Occupancy', 'VehicleType_S_Volume', 'VehicleType_S_Speed',
            'VehicleType_L_Volume', 'VehicleType_L_Speed', 'StnPres',
            'Temperature', 'RH', 'WS', 'WD', 'WSGust', 'WDGust', 'Precip'
        ]
    }

    # --- 檔案路徑設定 ---
    SCALER_PATH = f'result/GRU_congestionLevel_5min_scaler_v{MODEL_VERSION}.pkl'
    MODEL_PATH = f'result/GRU_congestionLevel_5min_model_v{MODEL_VERSION}.pth'
    
    INPUT_JSON_PATH = 'section23_2025_05_10.json'
    OUTPUT_JSON_PATH = 'section23_2025_05_10_output.json'

    # --- 執行預測 ---
    try:
        run_batch_prediction(
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