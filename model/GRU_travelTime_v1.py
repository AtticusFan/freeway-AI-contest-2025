import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# 引入迴歸評估指標，移除分類指標
from sklearn.metrics import mean_squared_error, mean_absolute_error

### functions區
# 假設 function.merged_data.load_and_merge_traffic_weather 函數已定義
from function.merged_data import load_and_merge_traffic_weather

if __name__ == '__main__':
    # 固定隨機種子
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # 資料讀取與合併
    traffic_file = 'datasets/vd_livetraffic_data_0501_0521.csv'
    weather_file = 'datasets/weather_data_0501_0521.csv'
    df = load_and_merge_traffic_weather(traffic_file, weather_file)
    df = df.sort_values(['SectionID', 'Timestamp'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp')

    # 資料重採樣與聚合
    agg_cfg = {
        'Status':'median', 'LaneID':'median', 'LaneType':'median', 'LaneSpeed':'median',
        'Occupancy':'median',
        'VehicleType_S_Volume':'median','VehicleType_S_Speed':'median',
        'VehicleType_L_Volume':'median','VehicleType_L_Speed':'median',
        'VehicleType_T_Volume':'median','VehicleType_T_Speed':'median',
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

    # ===== 修改點 1: 更新特徵與目標 =====
    # 從特徵中移除目標 TravelTime 和與其直接相關的 TravelSpeed
    FEATURES = [
        'Occupancy',
        'VehicleType_S_Volume','VehicleType_S_Speed',
        'VehicleType_L_Volume','VehicleType_L_Speed',
        'VehicleType_T_Volume','VehicleType_T_Speed',
        # 'TravelTime', # 移除
        # 'TravelSpeed', # 移除
        'StnPres','Temperature','RH','WS','WD','WSGust','WDGust','Precip'
    ]
    # 將目標改為 TravelTime
    TARGET = 'TravelTime'

    # 設定序列長度與預測期
    SEQ_LEN = 30
    HORIZONS = [5, 15, 30]
    model_version = input("請輸入版本號: ")

    for HORIZON in HORIZONS:
        print(f"\n========== 訓練 HORIZON = {HORIZON} 分鐘 ==========")

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
        y = np.array(targets, dtype=np.float32)
        section_ids = np.array(section_ids)
        times = np.array(times)

        nsamples, _, nfeatures = X.shape
        X_flat = X.reshape(-1, nfeatures)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_flat)
        X = X_scaled.reshape(nsamples, SEQ_LEN, nfeatures)

        X_temp, X_test, y_temp, y_test, ids_temp, ids_test, times_temp, times_test = train_test_split(
            X, y, section_ids, times,
            test_size=0.15,
            shuffle=True,
            stratify=section_ids,
            random_state=42
        )
        val_ratio = 0.15 / 0.85
        X_train, X_val, y_train, y_val, ids_train, ids_val, times_train, times_val = train_test_split(
            X_temp, y_temp, ids_temp, times_temp,
            test_size=val_ratio,
            shuffle=True,
            stratify=ids_temp,
            random_state=42
        )

        def create_loader(X, y, batch_size=128, shuffle=False):
            ds = TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32)
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        train_loader = create_loader(X_train, y_train, shuffle=True)
        val_loader   = create_loader(X_val,   y_val)
        test_loader  = create_loader(X_test,  y_test)

        class GRURegressor(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size=1, drop_prob=0.2):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, num_layers,
                                  batch_first=True, dropout=drop_prob)
                self.fc = nn.Linear(hidden_size, output_size)
            def forward(self, x):
                out, _ = self.gru(x)
                return self.fc(out[:, -1, :])
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GRURegressor(len(FEATURES), 64, 2, output_size=1).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        EPOCHS = 20
        
        records, metrics_records = [], []
        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss = 0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                yb = yb.view(-1, 1)
                preds = model(Xb)
                loss = criterion(preds, yb)
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
                    yb = yb.view(-1, 1)
                    preds = model(Xb)
                    val_loss += criterion(preds, yb).item() * Xb.size(0)
                    y_true.extend(yb.cpu().numpy().flatten())
                    y_pred.extend(preds.cpu().numpy().flatten())
            
            val_loss /= len(val_loader.dataset)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            # ===== 新增點 1: 計算自定義的 60 秒內準確率 =====
            y_true_np = np.array(y_true)
            y_pred_np = np.array(y_pred)
            # 計算預測值與真實值的絕對誤差
            abs_error = np.abs(y_pred_np - y_true_np)
            # 計算誤差在 60 秒內的比例
            accuracy_within_60s = np.mean(abs_error <= 60)

            # 計算測試集損失 (可選，但有助於監控)
            test_loss = 0
            with torch.no_grad():
                for Xb, yb in test_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    yb = yb.view(-1, 1)
                    preds = model(Xb)
                    test_loss += criterion(preds, yb).item() * Xb.size(0)
            test_loss /= len(test_loader.dataset)
            
            # 更新輸出訊息
            print(f"Epoch {epoch}/{EPOCHS}  |  Train Loss: {train_loss:.4f}  |  "
                  f"Val Loss: {val_loss:.4f}  |  Test Loss: {test_loss:.4f}  |  "
                  f"Val MAE: {mae:.2f}s  |  Val RMSE: {rmse:.2f}s")
            
            records.append({'Epoch': epoch, 'Train Loss': train_loss, 'Val Loss': val_loss, 'Test Loss': test_loss})
            metrics_records.append({'Epoch': epoch, 'MAE': mae, 'RMSE': rmse})

        # ===== 修改點 7: 更新存檔邏輯 =====
        # 寫檔 (Loss & Metrics)
        pd.DataFrame(records).to_csv(f'result/GRU_TravelTime_loss_{HORIZON}min_v{model_version}.csv', index=False)
        pd.DataFrame(metrics_records).to_csv(f'result/GRU_TravelTime_metrics_{HORIZON}min_v{model_version}.csv', index=False)

        # 產生測試集預測結果
        model.eval()
        y_pred_test = []
        with torch.no_grad():
            for Xb, _ in test_loader:
                Xb = Xb.to(device)
                preds = model(Xb)
                y_pred_test.extend(preds.cpu().numpy().flatten())
        
        df_pred = pd.DataFrame({
            'SectionID': ids_test,
            'CurrentTime': times_test,
            'TrueTravelTime': y_test,
            'PredictedTravelTime': y_pred_test
        })

        # ===== 新增點 2: 在最終預測的 DataFrame 中增加正確性判斷欄位 =====
        df_pred['IsCorrect_60s'] = np.abs(df_pred['PredictedTravelTime'] - df_pred['TrueTravelTime']) <= 60
        
        df_pred.to_csv(f'result/GRU_TravelTime_predictions_{HORIZON}min_v{model_version}.csv', index=False)

        # 輸出成純文字格式 (更新文字內容)
        output_filename = f'result/GRU_TravelTime_predictions_{HORIZON}min_v{model_version}.txt'
        with open(output_filename, 'w', encoding='utf-8') as fout:
            for sec, ts, pred_tt in zip(ids_test, times_test, y_pred_test, y_test):
                ts_pd = pd.to_datetime(ts)
                ts_str = ts_pd.strftime('%Y-%m-%d %H:%M:%S')
                line = (f"Section:{sec}, 當前時間:{ts_str}, "
                        f"預測{HORIZON}分鐘後該路段旅行時間: {pred_tt:.2f}秒\n")
                fout.write(line)

        print(f"已將測試結果輸出到：{output_filename}")