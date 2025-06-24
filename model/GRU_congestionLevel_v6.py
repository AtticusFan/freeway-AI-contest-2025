import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
### functions區
# from function.merged_data import load_and_merge_traffic_weather

if __name__ == '__main__':
    # 固定隨機種子
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # 設定 CUDA_LAUNCH_BLOCKING=1 有助於獲得更精確的錯誤堆疊追蹤
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    DATA_PATH = 'datasets/0401_0610/merged_traffic_weather_0401_0610.csv'
    df = pd.read_csv(DATA_PATH, parse_dates=['Timestamp'])
    df = df.sort_values(['SectionID', 'Timestamp'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp')
    
    agg_cfg = {
        'Occupancy':'median',
        'VehicleType_S_Volume':'median','VehicleType_S_Speed':'median',
        'VehicleType_L_Volume':'median','VehicleType_L_Speed':'median',
        'TravelTime':'median','TravelSpeed':'median',
        'StnPres':'median','Temperature':'median','RH':'median','WS':'median','WD':'median',
        'WSGust':'median','WDGust':'median','Precip':'median','CongestionLevel':'median'
    }

    df_resampled = (
        df.groupby('SectionID')
        .resample('1min')
        .agg(agg_cfg)
        .dropna()
    )
    df_resampled['median_speed'] = df_resampled[['VehicleType_S_Speed','VehicleType_L_Speed']].median(axis=1)
    
    FEATURES = [
        'Occupancy', 'VehicleType_S_Volume', 'VehicleType_L_Volume', 'median_speed',
        'StnPres','Temperature','RH','WS','WD','WSGust','WDGust','Precip'
    ]
    TARGET = 'CongestionLevel'
    
    SEQ_LEN = 30
    HORIZONS = [5, 15, 30]
    model_version = input("請輸入版本號: ")
    
    for HORIZON in HORIZONS:
        print(f"\n========== 訓練 HORIZON = {HORIZON} 分鐘 ==========")
        
        # 建立儲存目錄
        result_dirs = [f'result/congestionLevel/{HORIZON}min', 'result/model']
        for d in result_dirs:
            os.makedirs(d, exist_ok=True)
            
        sequences, targets, section_ids, times = [], [], [], []
        for sec, grp in df_resampled.groupby(level=0):
            arr = grp[FEATURES + [TARGET]].values
            time_index = grp.index.get_level_values('Timestamp').to_numpy()
            for i in range(len(arr) - SEQ_LEN - HORIZON + 1):
                sequences.append(arr[i : i + SEQ_LEN, :-1])
                targets.append(arr[i + SEQ_LEN + HORIZON - 1, -1])
                section_ids.append(sec)
                times.append(time_index[i + SEQ_LEN - 1])

        X = np.stack(sequences)
        
        # ==================== 主要修正點 1: 標籤轉換 ====================
        # 將標籤從 [1, 5] 轉換到 [0, 4] 以符合 CrossEntropyLoss 的要求
        y = np.array(targets, dtype=int) - 1
        # =============================================================

        section_ids = np.array(section_ids)
        times = np.array(times)

        nsamples, _, nfeatures = X.shape
        X_flat = X.reshape(-1, nfeatures)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_flat)
        X = X_scaled.reshape(nsamples, SEQ_LEN, nfeatures)
        
        scaler_path = f'result/model/GRU_congestionLevel_{HORIZON}min_scaler_v{model_version}.pkl'
        joblib.dump(scaler, scaler_path)
        print(f"Scaler 已儲存至：{scaler_path}")
        
        X_temp, X_test, y_temp, y_test, ids_temp, ids_test, times_temp, times_test = train_test_split(
            X, y, section_ids, times, test_size=0.15, shuffle=True, stratify=section_ids, random_state=42
        )
        val_ratio = 0.15 / 0.85
        X_train, X_val, y_train, y_val, ids_train, ids_val, times_train, times_val = train_test_split(
            X_temp, y_temp, ids_temp, times_temp, test_size=val_ratio, shuffle=True, stratify=ids_temp, random_state=42
        )

        def create_loader(X, y, batch_size=128, shuffle=False):
            ds = TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long)
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        train_loader = create_loader(X_train, y_train, shuffle=True)
        val_loader   = create_loader(X_val,   y_val)
        test_loader  = create_loader(X_test,  y_test)

        class GRUPredictor(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, num_classes, drop_prob=0.2):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=drop_prob)
                self.fc = nn.Linear(hidden_size, num_classes)
            def forward(self, x):
                out, _ = self.gru(x)
                return self.fc(out[:, -1, :])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_classes = len(np.unique(y))
        model = GRUPredictor(len(FEATURES), 64, 2, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        EPOCHS = 20

        records, metrics_records = [], []
        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss = 0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * Xb.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval()
            y_true, y_pred = [], []
            val_loss = 0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    logits = model(Xb)
                    val_loss += criterion(logits, yb).item() * Xb.size(0)
                    preds = logits.argmax(dim=1)
                    y_true.extend(yb.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
            val_loss /= len(val_loader.dataset)
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

            print(f"Epoch {epoch}/{EPOCHS}  Train Loss: {train_loss:.4f}  "
                  f"Val Loss: {val_loss:.4f}  Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")
            records.append({'Epoch': epoch, 'Train Loss': train_loss, 'Val Loss': val_loss})
            metrics_records.append({'Epoch': epoch, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1})

        pd.DataFrame(records).to_csv(f'result/congestionLevel/{HORIZON}min/GRU_congestionLevel_{HORIZON}min_training_log_v{model_version}.csv', index=False)
        pd.DataFrame(metrics_records).to_csv(f'result/congestionLevel/{HORIZON}min/GRU_congestionLevel_{HORIZON}min_metrics_v{model_version}.csv', index=False)

        model.eval()
        y_pred_test = []
        with torch.no_grad():
            for Xb, _ in test_loader:
                Xb = Xb.to(device)
                logits = model(Xb)
                y_pred_test.extend(logits.argmax(dim=1).cpu().numpy())
        
        # ==================== 主要修正點 2: 還原輸出標籤 ====================
        # 將 DataFrame 中的真實標籤和預測標籤都加 1，以還原到 [1, 5] 的範圍
        df_pred = pd.DataFrame({
            'SectionID': ids_test,
            'CurrentTime': times_test,
            'TrueCongestionLevel': y_test + 1,
            'PredictedCongestionLevel': np.array(y_pred_test) + 1
        })
        # ===================================================================
        df_pred.to_csv(f'result/congestionLevel/{HORIZON}min/GRU_congestionLevel_{HORIZON}min_predictions_v{model_version}.csv', index=False)
        
        model_path = f'result/model/GRU_congestionLevel_{HORIZON}min_model_v{model_version}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"已將模型儲存至：{model_path}")
        
        output_filename = f'result/congestionLevel/{HORIZON}min/GRU_congestionLevel_{HORIZON}min_predict_info_v{model_version}.txt'
        with open(output_filename, 'w', encoding='utf-8') as fout:
            for sec, ts, pred in zip(ids_test, times_test, y_pred_test):
                ts_pd = pd.to_datetime(ts)
                ts_str = ts_pd.strftime('%Y-%m-%d %H:%M:%S')
                # ==================== 主要修正點 3: 還原純文字輸出標籤 ====================
                # 將預測標籤加 1，還原到 [1, 5] 的範圍
                line = f"Section:{sec}，當前時間：{ts_str}，{HORIZON}分鐘後壅塞程度：{pred + 1}\n"
                # ========================================================================
                fout.write(line)

        print(f"已將測試結果以純文字格式輸出到：{output_filename}")