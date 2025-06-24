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

if __name__ == '__main__':
    # 固定隨機種子
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    DATA_PATH = 'datasets/0401_0610/merged_traffic_weather_geometry_0401_0610.csv'
    print(f"=====讀取資料集：{DATA_PATH}=====")
    # 讀檔、排序、設定 Timestamp 索引
    df = pd.read_csv(DATA_PATH, parse_dates=['Timestamp'])
    df = df.sort_values(['SectionID','Timestamp'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp')
    print(f"=====資料前處理=====")
    # 1. 靜態幾何特徵聚合
    geom_numeric = [
        'geom_LaneWidth','geom_TotalWidth','geom_LaneCount',
        'geom_Lane1Width','geom_Lane2Width','geom_Lane3Width','geom_Lane4Width',
        'geom_ChannelizationWidth','geom_InnerShoulderWidth',
        'geom_OuterShoulderWidth','geom_AuxLane1Width','geom_AuxLane2Width','geom_AuxLane3Width',
        'geom_BayArea','geom_CurveRadius','geom_LongitudinalSlope','geom_LateralSlope'
    ]
    geom_categorical = [
        'geom_PavementType','geom_Channelization',
        'geom_InnerShoulder','geom_OuterShoulder',
        'geom_AuxLane1','geom_AuxLane2','geom_AuxLane3'
    ]
    geom_num_agg = df.groupby('SectionID')[geom_numeric].median()
    geom_cat_agg = df.groupby('SectionID')[geom_categorical].first()
    geom_static = pd.concat([geom_num_agg, geom_cat_agg], axis=1)

    # 2. 動態特徵重取樣聚合並計算 median_speed
    agg_cfg = {
        'Occupancy':'median',
        'VehicleType_S_Volume':'median','VehicleType_S_Speed':'median',
        'VehicleType_L_Volume':'median','VehicleType_L_Speed':'median',
        'StnPres':'median','Temperature':'median','RH':'median',
        'WS':'median','WD':'median','WSGust':'median','WDGust':'median',
        'Precip':'median','CongestionLevel':'median'
    }
    df_dyn = (
        df.groupby('SectionID')
        .resample('1min')
        .agg(agg_cfg)
        .dropna(how='all')
    )
    df_dyn['median_speed'] = df_dyn[['VehicleType_S_Speed','VehicleType_L_Speed']].median(axis=1)

    # 3. 合併靜態幾何回到動態表
    df_resampled = (
        df_dyn
        .reset_index()
        .merge(geom_static, on='SectionID')
        .set_index(['SectionID','Timestamp'])
    )
    for col in geom_categorical:
        df_resampled[col] = df_resampled[col].astype('category').cat.codes
        
    # 4. 更新 FEATURES 清單
    FEATURES = [
        'Occupancy','VehicleType_S_Volume','VehicleType_L_Volume','median_speed',
        *geom_numeric, *geom_categorical,
        'StnPres','Temperature','RH','WS','WD','WSGust','WDGust','Precip'
    ]
    TARGET = 'CongestionLevel'
    
    print("=====NaN 值處理=====")
    # <<< 修改點：將 .apply() 替換為 .transform() >>>
    df_resampled[FEATURES] = df_resampled.groupby(level=0)[FEATURES].transform(lambda x: x.ffill().bfill())
    
    # 填充後若仍有 NaN (可能整個 SectionID 都沒有資料)，則刪除這些行
    df_resampled.dropna(subset=FEATURES + [TARGET], inplace=True)
    print("NaN 處理後剩餘資料筆數:", len(df_resampled))
    
    # 預測未來 5、15、30 分鐘的壅塞等級
    SEQ_LEN = 30
    HORIZONS = [5, 15, 30]
    
    model_version = input("請輸入版本號: ")

    print("=====依 SectionID 切分資料集=====")
    unique_sections = df_resampled.index.get_level_values('SectionID').unique()
    ids_temp, ids_test = train_test_split(unique_sections, test_size=0.15, random_state=42)
    val_ratio = 0.15 / 0.85
    ids_train, ids_val = train_test_split(ids_temp, test_size=val_ratio, random_state=42)

    print(f"訓練 Section 數量: {len(ids_train)}")
    print(f"驗證 Section 數量: {len(ids_val)}")
    print(f"測試 Section 數量: {len(ids_test)}")
    
    train_sections_set = set(ids_train)
    val_sections_set = set(ids_val)
    test_sections_set = set(ids_test)
    
    for HORIZON in HORIZONS:
        print(f"\n========== 訓練 HORIZON = {HORIZON} 分鐘 ==========")
        
        def generate_sequences(target_section_ids, df_full):
            sequences, targets, section_ids, times = [], [], [], []
            for sec, grp in df_full.groupby(level=0):
                if sec not in target_section_ids:
                    continue
                arr = grp[FEATURES + [TARGET]].values
                time_index = grp.index.get_level_values('Timestamp').to_numpy()
                for i in range(len(arr) - SEQ_LEN - HORIZON + 1):
                    sequences.append(arr[i : i + SEQ_LEN, :-1])
                    targets.append(arr[i + SEQ_LEN + HORIZON - 1, -1])
                    section_ids.append(sec)
                    times.append(time_index[i + SEQ_LEN - 1])
            
            # 處理分組後沒有產生任何序列的狀況
            if not sequences:
                return np.array([]), np.array([]), np.array([]), np.array([])

            X = np.stack(sequences)
            y = np.array(targets, dtype=int)
            s_ids = np.array(section_ids)
            t_stamps = np.array(times)
            return X, y, s_ids, t_stamps
        
        # 分別為 train, val, test 建立序列
        X_train, y_train, ids_train_seq, times_train = generate_sequences(train_sections_set, df_resampled)
        X_val, y_val, ids_val_seq, times_val = generate_sequences(val_sections_set, df_resampled)
        X_test, y_test, ids_test_seq, times_test = generate_sequences(test_sections_set, df_resampled)

        if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
            print(f"HORIZON={HORIZON} 時，資料不足以切分訓練、驗證或測試集，跳過此輪訓練。")
            continue

        print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}")
        
        # 正規化 X
        nsamples_train, _, nfeatures = X_train.shape
        X_train_flat = X_train.reshape(-1, nfeatures)
        
        scaler = MinMaxScaler()
        X_train_scaled_flat = scaler.fit_transform(X_train_flat)
        X_train = X_train_scaled_flat.reshape(nsamples_train, SEQ_LEN, nfeatures)
        
        nsamples_val, _, _ = X_val.shape
        X_val_flat = X_val.reshape(-1, nfeatures)
        X_val_scaled_flat = scaler.transform(X_val_flat)
        X_val = X_val_scaled_flat.reshape(nsamples_val, SEQ_LEN, nfeatures)

        nsamples_test, _, _ = X_test.shape
        X_test_flat = X_test.reshape(-1, nfeatures)
        X_test_scaled_flat = scaler.transform(X_test_flat)
        X_test = X_test_scaled_flat.reshape(nsamples_test, SEQ_LEN, nfeatures)

        scaler_dir = 'result'
        os.makedirs(scaler_dir, exist_ok=True)
        scaler_path = f'{scaler_dir}/GRU_congestionLevel_{HORIZON}min_scaler_v{model_version}.pkl'
        joblib.dump(scaler, scaler_path)
        print(f"Scaler 已儲存至：{scaler_path}")
        
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
                self.gru = nn.GRU(input_size, hidden_size, num_layers,
                                  batch_first=True, dropout=drop_prob)
                self.fc = nn.Linear(hidden_size, num_classes)
            def forward(self, x):
                out, _ = self.gru(x)
                return self.fc(out[:, -1, :])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_classes = len(np.unique(df_resampled[TARGET].astype(int)))
        model = GRUPredictor(len(FEATURES), 64, 2, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        EPOCHS = 20

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
            
            print(f"Epoch {epoch}/{EPOCHS}  Train Loss: {train_loss:.4f}  "
                  f"Val Loss: {val_loss:.4f}  "
                  f"Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")
            records.append({'Epoch': epoch, 'Train Loss': train_loss, 'Val Loss': val_loss})
            metrics_records.append({'Epoch': epoch, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1})
            
        output_dir = f'result/congestionLevel/{HORIZON}min'
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(records).to_csv(f'{output_dir}/GRU_congestionLevel_{HORIZON}min_training_log_v{model_version}.csv', index=False)
        pd.DataFrame(metrics_records).to_csv(f'{output_dir}/GRU_congestionLevel_{HORIZON}min_metrics_v{model_version}.csv', index=False)

        model.eval()
        y_pred_test = []
        with torch.no_grad():
            for Xb, _ in test_loader:
                Xb = Xb.to(device)
                logits = model(Xb)
                y_pred_test.extend(logits.argmax(dim=1).cpu().numpy())

        df_pred = pd.DataFrame({
            'SectionID': ids_test_seq,
            'CurrentTime': times_test,
            'TrueCongestionLevel': y_test,
            'PredictedCongestionLevel': y_pred_test
        })
        df_pred.to_csv(f'{output_dir}/GRU_congestionLevel_{HORIZON}min_predictions_v{model_version}.csv', index=False)
        
        model_path = f'{output_dir}/GRU_congestionLevel_{HORIZON}min_model_v{model_version}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"已將模型儲存至：{model_path}")

        output_filename = f'{output_dir}/GRU_congestionLevel_{HORIZON}min_predict_info_v{model_version}.txt'
        with open(output_filename, 'w', encoding='utf-8') as fout:
            for sec, ts, pred in zip(ids_test_seq, times_test, y_pred_test):
                ts_pd = pd.to_datetime(ts)
                ts_str = ts_pd.strftime('%Y-%m-%d %H:%M:%S')
                line = f"Section:{sec}，當前時間：{ts_str}，{HORIZON}分鐘後壅塞程度：{pred}\n"
                fout.write(line)
        print(f"已將測試結果以純文字格式輸出到：{output_filename}")