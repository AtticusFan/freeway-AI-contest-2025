import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
### functions區
from function.merged_data import load_and_merge_traffic_weather

if __name__ == '__main__':
    # 固定隨機種子
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    print("讀取資料並進行預處理...")
    traffic_file = 'datasets/vd_livetraffic_data_0501_0521.csv'
    weather_file = 'datasets/weather_data_0501_0521.csv'
    df = load_and_merge_traffic_weather(traffic_file, weather_file)
    df = df.sort_values(['SectionID', 'Timestamp'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp')

    # 中位數聚合設定
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

    FEATURES = [
        'Occupancy',
        'VehicleType_S_Volume','VehicleType_S_Speed',
        'VehicleType_L_Volume','VehicleType_L_Speed',
        # 'VehicleType_T_Volume','VehicleType_T_Speed',
        #'TravelSpeed',
        'StnPres','Temperature','RH','WS','WD','WSGust','WDGust','Precip'
    ]
    # 目標變數改成 TravelTime（秒）
    TARGET = 'TravelTime'

    # 使用過去 SEQ_LEN 分鐘的資料來預測 HORIZON 分鐘後的旅行時間
    SEQ_LEN = 30
    HORIZONS = [5, 15, 30]
    model_version = input("請輸入模型版本號: ")

    for HORIZON in HORIZONS:
        print(f"\n========== 訓練 HORIZON = {HORIZON} 分鐘 ==========")

        sequences, targets, section_ids, times = [], [], [], []
        for section, grp in df_resampled.groupby(level=0):
            arr = grp[FEATURES + [TARGET]].values  # 最後一維是 TravelTime
            time_index = grp.index.get_level_values('Timestamp').to_numpy()
            for i in range(len(arr) - SEQ_LEN - HORIZON + 1):
                # 序列輸入 X：前 SEQ_LEN 行的 FEATURES
                sequences.append(arr[i : i + SEQ_LEN, :-1])
                # 目標 y：SEQ_LEN + HORIZON - 1 對應的 TravelTime
                targets.append(arr[i + SEQ_LEN + HORIZON - 1, -1])
                section_ids.append(section)
                times.append(time_index[i + SEQ_LEN - 1])

        X = np.stack(sequences).astype(np.float32)  # 形狀 (N, SEQ_LEN, n_features)
        y = np.array(targets, dtype=np.float32)     # 形狀 (N,)
        section_ids = np.array(section_ids)
        times = np.array(times)

        # 正規化 X（對每個 feature 做 MinMaxScaler）
        nsamples, _, nfeatures = X.shape
        X_flat = X.reshape(-1, nfeatures)
        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X_flat)
        X = X_scaled.reshape(nsamples, SEQ_LEN, nfeatures)

        # 正規化 y（TravelTime）以提升收斂
        y = y.reshape(-1, 1)
        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y).reshape(-1)

        # 切分 Train/Val/Test，同時帶入 section_ids 與 times
        X_temp, X_test, y_temp, y_test, ids_temp, ids_test, times_temp, times_test = train_test_split(
            X, y_scaled, section_ids, times,
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

        # 建立 DataLoader
        def create_loader(X, y, batch_size=128, shuffle=False):
            ds = TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape (batch, 1)
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        train_loader = create_loader(X_train, y_train, shuffle=True)
        val_loader   = create_loader(X_val,   y_val)
        test_loader  = create_loader(X_test,  y_test)

        # 定義 GRU 回歸模型
        class GRURegressor(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, drop_prob=0.2):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, num_layers,
                                  batch_first=True, dropout=drop_prob)
                self.fc = nn.Linear(hidden_size, 1)
            def forward(self, x):
                out, _ = self.gru(x)           # out 的 shape 為 (batch, SEQ_LEN, hidden_size)
                return self.fc(out[:, -1, :])  # 只取最後一個時間步的 hidden state

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GRURegressor(len(FEATURES), 64, 2).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        EPOCHS = 20

        for epoch in range(1, EPOCHS+1):
            model.train()
            train_loss = 0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * Xb.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    pred = model(Xb)
                    val_loss += criterion(pred, yb).item() * Xb.size(0)
            val_loss /= len(val_loader.dataset)

            test_loss = 0
            with torch.no_grad():
                for Xb, yb in test_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    pred = model(Xb)
                    test_loss += criterion(pred, yb).item() * Xb.size(0)
            test_loss /= len(test_loader.dataset)

            print(f"Epoch {epoch}/{EPOCHS}  Train MSE: {train_loss:.4f}  Val MSE: {val_loss:.4f}  Test MSE: {test_loss:.4f}")
            
            # 寫入訓練過程數值到CSV檔案
            log_filename = f'result/travelTime/{HORIZON}min/GRU_TravelTime_{HORIZON}min_training_log_v{model_version}.csv'
            log_data = pd.DataFrame({
                'Epoch': [epoch],
                'Train_MSE': [train_loss],
                'Val_MSE': [val_loss],
                'Test_MSE': [test_loss]
            })
            
            if epoch == 1:
                log_data.to_csv(log_filename, index=False)
            else:
                log_data.to_csv(log_filename, mode='a', header=False, index=False)

        # 儲存模型參數
        #torch.save(model.state_dict(), f'model/GRU_TravelTime_{HORIZON}min_v{model_version}.pt')

        # 在測試集上做預測
        model.eval()
        y_pred_scaled = []
        y_true_scaled = []
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb = Xb.to(device)
                pred = model(Xb)
                y_pred_scaled.extend(pred.cpu().numpy().reshape(-1))
                y_true_scaled.extend(yb.cpu().numpy().reshape(-1))

        # 逆向轉回原始尺度
        y_pred = scaler_y.inverse_transform(np.array(y_pred_scaled).reshape(-1,1)).reshape(-1)
        y_true = scaler_y.inverse_transform(np.array(y_true_scaled).reshape(-1,1)).reshape(-1)

        # # 計算一般回歸指標（原始尺度）
        # mse = mean_squared_error(y_true, y_pred)
        # mae = mean_absolute_error(y_true, y_pred)
        #print(f"HORIZON {HORIZON} 分鐘  Test MSE (原始): {mse:.4f}  MAE: {mae:.4f}")

        # ===== 計算預測正確率 =====
        # |y_pred - y_true| <= 60
        diff = np.abs(y_pred - y_true)
        correct_60 = np.sum(diff <= 60)
        total_samples = len(diff)
        acc_within_60 = correct_60 / total_samples
        print(f"預測{HORIZON}分鐘後旅行時間的Accuracy {acc_within_60:.4f}")
        
        # 將指標結果寫入CSV檔案
        metrics_data = pd.DataFrame({
            'Accuracy': [acc_within_60]
        })
        
        metrics_filename = f'result/travelTime/{HORIZON}min/GRU_TravelTime_{HORIZON}min_metrics_v{model_version}.csv'
        metrics_data.to_csv(metrics_filename, index=False)

        # 封裝成 DataFrame，方便輸出 CSV
        df_pred = pd.DataFrame({
            'SectionID': ids_test,
            'CurrentTime': times_test,
            'TrueTravelTime': y_true,
            'PredictedTravelTime': y_pred,
            'Within30': (diff <= 60).astype(int)
        })
        df_pred.to_csv(f'result/travelTime/{HORIZON}min/GRU_TravelTime_{HORIZON}min_predictions_v{model_version}.csv', index=False)

        # 將結果以純文字格式輸出
        output_filename = f'result/travelTime/{HORIZON}min/GRU_TravelTime_{HORIZON}min_predict_info_v{model_version}.txt'
        with open(output_filename, 'w', encoding='utf-8') as fout:
            for section, ts, pred_value in zip(ids_test, times_test, y_pred):
                ts_pd = pd.to_datetime(ts)
                current_time = ts_pd.strftime('%Y-%m-%d %H:%M')
                line = (
                    f"Section:{section}，當前時間：{current_time}，"
                    f"{HORIZON} 分鐘後旅行時間，預測值：{pred_value:.2f}秒\n"
                )
                fout.write(line)

        print(f"HORIZON {HORIZON} 分鐘 旅行時間預測檔案已輸出：{output_filename}")
