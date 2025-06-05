import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
### functions區
from function.merged_data import load_and_merge_traffic_weather

if __name__ == '__main__':
    # 固定隨機種子
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # DATA_PATH = '../datasets/merged_traffic_weather_0501_0521.csv'
    # df = pd.read_csv(DATA_PATH, parse_dates=['Timestamp'])
    traffic_file = 'datasets/vd_livetraffic_data_0501_0521.csv'
    weather_file = 'datasets/weather_data_0501_0521.csv'
    df = load_and_merge_traffic_weather(traffic_file, weather_file)
    df = df.sort_values(['SectionID', 'Timestamp'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp')
    # 中位數
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
        'VehicleType_T_Volume','VehicleType_T_Speed',
        'TravelTime','TravelSpeed',
        'StnPres','Temperature','RH','WS','WD','WSGust','WDGust','Precip'
    ]
    TARGET = 'CongestionLevel'
    # 預測未來 5 分鐘的壅塞等級，使用過去 60 分鐘的資料
    SEQ_LEN = 30
    #HORIZON = 30
    # 預測未來 5、15、30 分鐘的壅塞等級
    HORIZONS = [5, 15, 30]
    model_version = input("請輸入版本號: ")
    for HORIZON in HORIZONS:
        print(f"\n========== 訓練 HORIZON = {HORIZON} 分鐘 ==========")
    

        sequences, targets, section_ids, times = [], [], [], []
        for sec, grp in df_resampled.groupby(level=0):
            arr = grp[FEATURES + [TARGET]].values
            #time_index = grp.index.to_numpy()
            time_index = grp.index.get_level_values('Timestamp').to_numpy()
            for i in range(len(arr) - SEQ_LEN - HORIZON + 1):
                sequences.append(arr[i : i + SEQ_LEN, :-1])
                targets.append(arr[i + SEQ_LEN + HORIZON - 1, -1])
                section_ids.append(sec)
                #times.append(time_index[i + SEQ_LEN - 1])
                times.append(time_index[i + SEQ_LEN - 1])

        X = np.stack(sequences)
        y = np.array(targets, dtype=int)
        section_ids = np.array(section_ids)
        times = np.array(times)

        # 正規化 X
        nsamples, _, nfeatures = X.shape
        X_flat = X.reshape(-1, nfeatures)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_flat)
        X = X_scaled.reshape(nsamples, SEQ_LEN, nfeatures)

        # 切分 Train/Val/Test，同時帶入 section_ids 與 times
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

        # 建立 DataLoader（與原本相同）
        def create_loader(X, y, batch_size=128, shuffle=False):
            ds = TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long)
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        train_loader = create_loader(X_train, y_train, shuffle=True)
        val_loader   = create_loader(X_val,   y_val)
        test_loader  = create_loader(X_test,  y_test)

        # 定義 GRU 分類模型
        class GRUPredictor(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, num_classes, drop_prob=0.2):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=drop_prob)
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

        #訓練與評估
        records, metrics_records = [], []
        for epoch in range(1, EPOCHS+1):
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

            test_loss = 0
            with torch.no_grad():
                for Xb, yb in test_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    logits = model(Xb)
                    test_loss += criterion(logits, yb).item() * Xb.size(0)
            test_loss /= len(test_loader.dataset)

            print(f"Epoch {epoch}/{EPOCHS}  Train Loss: {train_loss:.4f}  "
                f"Val Loss: {val_loss:.4f}  Test Loss: {test_loss:.4f}  "
                f"Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")
            records.append({'Epoch': epoch, 'Train Loss': train_loss, 'Val Loss': val_loss, 'Test Loss': test_loss})
            metrics_records.append({'Epoch': epoch, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1})

        # 寫檔
        
        pd.DataFrame(records).to_csv(f'result/GRU_0501_0521_loss_{HORIZON}min_v{model_version}.csv', index=False)
        pd.DataFrame(metrics_records).to_csv(f'result/GRU_0501_0521_metrics_{HORIZON}min_v{model_version}.csv', index=False)

        model.eval()
        y_pred_test = []
        with torch.no_grad():
            for Xb, _ in test_loader:
                Xb = Xb.to(device)
                logits = model(Xb)
                y_pred_test.extend(logits.argmax(dim=1).cpu().numpy())
        # print("times_test   :", np.array(times_test).shape)
        # ids_arr   = np.array(ids_test).ravel()
        # times_arr = np.array(times_test).ravel()
        # y_true    = np.array(y_test).ravel()
        # y_pred    = np.array(y_pred_test).ravel()
        # print("壓平後維度檢查：")
        # print("ids_arr      :", ids_arr.shape)
        # print("times_arr    :", times_arr.shape)
        # print("y_true       :", y_true.shape)
        # print("y_pred       :", y_pred.shape)
        # 封裝成 DataFrame，方便輸出 CSV
        df_pred = pd.DataFrame({
            'SectionID': ids_test,
            'CurrentTime': times_test,
            'TrueCongestionLevel': y_test,
            'PredictedCongestionLevel': y_pred_test
        })
        df_pred.to_csv(f'result/GRU_0501_0521_predictions_{HORIZON}min_v{model_version}.csv', index=False)

        # ====== 依照需求，將結果輸出成純文字格式 ======
        output_filename = f'result/GRU_0501_0521_predictions_{HORIZON}min_v{model_version}.txt'
        with open(output_filename, 'w', encoding='utf-8') as fout:
            for sec, ts, pred in zip(ids_test, times_test, y_pred_test):
                ts_pd = pd.to_datetime(ts)
                ts_str = ts_pd.strftime('%Y-%m-%d %H:%M:%S')
                line = f"Section:{sec}，當前時間：{ts_str}，{HORIZON}分鐘後壅塞程度：{pred}\n"
                fout.write(line)

        print(f"已將測試結果以純文字格式輸出到：{output_filename}")