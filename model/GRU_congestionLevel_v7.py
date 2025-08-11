import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import gc

# ==================== 主要修正點 1: 建立自定義 Dataset ====================
# 為處理大規模資料、避免記憶體爆炸，建立一個自定義的 Dataset 類別。
# 它會根據索引即時地從 DataFrame 中生成訓練樣本，而不是一次性載入所有資料。
class TrafficDataset(Dataset):
    def __init__(self, df, features_tensor, target_tensor, pointers, seq_len, horizon, is_test=False):
        """
        初始化 Dataset。
        :param df: 預處理後的原始 DataFrame，用於獲取元數據 (SectionID, Timestamp)。
        :param features_tensor: 包含所有特徵且已標準化的 PyTorch 張量。
        :param target_tensor: 包含所有目標值的 PyTorch 張量。
        :param pointers: 一個列表，其中每個元素是一個字典，指向一個有效序列的起始位置。
        :param seq_len: 輸入序列長度。
        :param horizon: 預測時間範圍。
        :param is_test: 布林值，指示是否為測試集（測試集需要返回額外的元數據）。
        """
        self.df = df
        self.features_tensor = features_tensor
        self.target_tensor = target_tensor
        self.pointers = pointers
        self.seq_len = seq_len
        self.horizon = horizon
        self.is_test = is_test

    def __len__(self):
        # 返回資料集的總樣本數
        return len(self.pointers)

    def __getitem__(self, idx):
        # 根據索引 idx 獲取一個樣本
        pointer = self.pointers[idx]
        start_iloc = pointer['start_iloc']
        
        # 定義輸入序列和目標值的索引位置
        sequence_end_iloc = start_iloc + self.seq_len
        target_iloc = start_iloc + self.seq_len + self.horizon - 1
        
        # 從預先轉換好的張量中切片，效率高
        sequence = self.features_tensor[start_iloc : sequence_end_iloc]
        # 將目標值從 [1, 5] 轉換為 [0, 4]
        target = self.target_tensor[target_iloc] - 1
        
        if self.is_test:
            # 如果是測試集，額外返回 SectionID 和時間戳記以供後續分析
            current_time_iloc = start_iloc + self.seq_len - 1
            meta_info = self.df.iloc[current_time_iloc]
            section_id = meta_info.name[0]  # 從 MultiIndex ('SectionID', 'Timestamp') 獲取
            current_time = meta_info.name[1]
            return sequence, target, section_id, current_time
        else:
            return sequence, target

# 為測試集的 DataLoader 定義一個自定義的 collate_fn
# 因為 __getitem__ 返回的資料類型不一 (tensors, int, str, timestamp)，需要手動打包
def test_collate_fn(batch):
    sequences, targets, section_ids, current_times = zip(*batch)
    sequences = torch.stack(sequences, 0)
    targets = torch.tensor(targets, dtype=torch.long)
    return sequences, targets, list(section_ids), list(current_times)

if __name__ == '__main__':
    # 固定隨機種子
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # 設定 CUDA_LAUNCH_BLOCKING=1 有助於獲得更精確的錯誤堆疊追蹤
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    DATA_PATH = 'datasets/0401_0610/merged_traffic_weather_geometry_0401_0610.csv'
    print("正在讀取資料...")
    df = pd.read_csv(DATA_PATH, parse_dates=['Timestamp'])
    
    print("正在進行資料預處理...")
    # 對類別型道路幾何特徵進行 One-Hot Encoding
    categorical_geom_features = [
        'geom_PavementType', 'geom_Channelization', 'geom_InnerShoulder',
        'geom_OuterShoulder', 'geom_AuxLane1', 'geom_AuxLane2', 'geom_AuxLane3', 'geom_BayArea'
    ]
    for col in categorical_geom_features:
        df[col] = df[col].fillna('None').astype(str).str.strip()
        df.loc[df[col] == '', col] = 'None'

    df = pd.get_dummies(df, columns=categorical_geom_features, prefix=categorical_geom_features)
    
    df = df.sort_values(['SectionID', 'Timestamp'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp')
    
    numerical_geom_features = [
        'geom_LaneWidth', 'geom_TotalWidth', 'geom_LaneCount',
        'geom_Lane1Width', 'geom_Lane2Width', 'geom_Lane3Width', 'geom_Lane4Width',
        'geom_ChannelizationWidth', 'geom_InnerShoulderWidth', 'geom_OuterShoulderWidth',
        'geom_AuxLane1Width', 'geom_AuxLane2Width', 'geom_AuxLane3Width',
        'geom_CurveRadius', 'geom_LongitudinalSlope', 'geom_LateralSlope'
    ]
    one_hot_geom_features = [c for c in df.columns if any(c.startswith(prefix) for prefix in categorical_geom_features)]
    df[numerical_geom_features] = df[numerical_geom_features].fillna(0)

    agg_cfg = {
        'Occupancy':'median','VehicleType_S_Volume':'median','VehicleType_S_Speed':'median',
        'VehicleType_L_Volume':'median','VehicleType_L_Speed':'median',
        'TravelTime':'median','TravelSpeed':'median',
        'StnPres':'median','Temperature':'median','RH':'median','WS':'median','WD':'median',
        'WSGust':'median','WDGust':'median','Precip':'median','CongestionLevel':'median'
    }
    for feat in numerical_geom_features + one_hot_geom_features:
        agg_cfg[feat] = 'first'
        
    print("正在重採樣資料...")
    df_resampled = (
        df.groupby('SectionID')
        .resample('1min')
        .agg(agg_cfg)
        .dropna(subset=['CongestionLevel'])
    )
    df_resampled = df_resampled.groupby(level=0).ffill().groupby(level=0).bfill().dropna()
    df_resampled['median_speed'] = df_resampled[['VehicleType_S_Speed','VehicleType_L_Speed']].median(axis=1)
    
    # 釋放原始 DataFrame 記憶體以節省資源
    del df
    gc.collect()

    BASE_FEATURES = [
        'Occupancy', 'VehicleType_S_Volume', 'VehicleType_L_Volume', 'median_speed',
        'StnPres','Temperature','RH','WS','WD','WSGust','WDGust','Precip'
    ]
    FEATURES = BASE_FEATURES + numerical_geom_features + one_hot_geom_features
    TARGET = 'CongestionLevel'
    
    SEQ_LEN = 30
    HORIZONS = [5, 15, 30]
    model_version = input("請輸入版本號: ")
    
    print("正在進行特徵標準化...")
    scaler = MinMaxScaler()
    df_resampled[FEATURES] = scaler.fit_transform(df_resampled[FEATURES])
    
    # 將 DataFrame 轉換為 PyTorch 張量，這將是 Dataset 的資料來源
    print("正在將資料轉換為張量...")
    features_tensor = torch.tensor(df_resampled[FEATURES].values, dtype=torch.float32)
    target_tensor = torch.tensor(df_resampled[TARGET].values, dtype=torch.long)
    
    for HORIZON in HORIZONS:
        print(f"\n========== 訓練 HORIZON = {HORIZON} 分鐘 ==========")
        
        result_dirs = [f'result/congestionLevel/{HORIZON}min', 'result/model']
        for d in result_dirs:
            os.makedirs(d, exist_ok=True)
        
        scaler_path = f'result/model/GRU_congestionLevel_{HORIZON}min_scaler_v{model_version}.pkl'
        joblib.dump(scaler, scaler_path)
        print(f"Scaler 已儲存至：{scaler_path}")
        
        # 建立序列指標，而不是生成實際的序列資料
        print("正在建立序列指標...")
        valid_pointers = []
        grouped = df_resampled.groupby(level=0)
        for sec_name, grp in grouped:
            num_rows = len(grp)
            start_iloc_global = df_resampled.index.get_loc(grp.index[0])
            for i in range(num_rows - SEQ_LEN - HORIZON + 1):
                valid_pointers.append({
                    'section': sec_name, 
                    'start_iloc': start_iloc_global + i
                })
        print(f"共找到 {len(valid_pointers)} 個有效序列。")
        
        # 對指標列表進行分層抽樣
        sections_for_stratify = [p['section'] for p in valid_pointers]
        indices = list(range(len(valid_pointers)))
        
        temp_indices, test_indices, _, _ = train_test_split(
            indices, sections_for_stratify, test_size=0.15, shuffle=True, stratify=sections_for_stratify, random_state=42
        )
        temp_sections = [sections_for_stratify[i] for i in temp_indices]
        val_ratio = 0.15 / 0.85
        train_indices, val_indices, _, _ = train_test_split(
            temp_indices, temp_sections, test_size=val_ratio, shuffle=True, stratify=temp_sections, random_state=42
        )

        pointers_train = [valid_pointers[i] for i in train_indices]
        pointers_val = [valid_pointers[i] for i in val_indices]
        pointers_test = [valid_pointers[i] for i in test_indices]

        # 實例化自定義 Dataset
        train_dataset = TrafficDataset(df_resampled, features_tensor, target_tensor, pointers_train, SEQ_LEN, HORIZON)
        val_dataset = TrafficDataset(df_resampled, features_tensor, target_tensor, pointers_val, SEQ_LEN, HORIZON)
        test_dataset = TrafficDataset(df_resampled, features_tensor, target_tensor, pointers_test, SEQ_LEN, HORIZON, is_test=True)

        # 建立 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, collate_fn=test_collate_fn)
        
        # 定義模型
        class GRUPredictor(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, num_classes, drop_prob=0.2):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=drop_prob)
                self.fc = nn.Linear(hidden_size, num_classes)
            def forward(self, x):
                out, _ = self.gru(x)
                return self.fc(out[:, -1, :])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_classes = int(df_resampled[TARGET].max())
        model = GRUPredictor(len(FEATURES), 64, 2, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        EPOCHS = 20

        records, metrics_records = [], []
        print(f"開始訓練模型 (Horizon: {HORIZON} min)...")
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

        print("正在對測試集進行預測...")
        model.eval()
        all_preds, all_trues, all_ids, all_times = [], [], [], []
        with torch.no_grad():
            for Xb, yb, id_b, time_b in test_loader:
                Xb = Xb.to(device)
                logits = model(Xb)
                preds = logits.argmax(dim=1).cpu()
                
                all_preds.extend(preds.numpy())
                all_trues.extend(yb.numpy())
                all_ids.extend(id_b)
                all_times.extend(time_b)
        
        # 還原標籤 [0, 4] -> [1, 5]
        y_pred_test_restored = np.array(all_preds) + 1
        y_true_test_restored = np.array(all_trues) + 1
        
        df_pred = pd.DataFrame({
            'SectionID': all_ids,
            'CurrentTime': all_times,
            'TrueCongestionLevel': y_true_test_restored,
            'PredictedCongestionLevel': y_pred_test_restored
        })
        df_pred.to_csv(f'result/congestionLevel/{HORIZON}min/GRU_congestionLevel_{HORIZON}min_predictions_v{model_version}.csv', index=False)
        
        model_path = f'result/model/GRU_congestionLevel_{HORIZON}min_model_v{model_version}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"已將模型儲存至：{model_path}")
        
        output_filename = f'result/congestionLevel/{HORIZON}min/GRU_congestionLevel_{HORIZON}min_predict_info_v{model_version}.txt'
        with open(output_filename, 'w', encoding='utf-8') as fout:
            for sec, ts, pred in zip(all_ids, all_times, y_pred_test_restored):
                ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
                line = f"Section:{sec}，當前時間：{ts_str}，{HORIZON}分鐘後壅塞程度：{pred}\n"
                fout.write(line)
        print(f"已將測試結果以純文字格式輸出到：{output_filename}")

    # 清理記憶體
    del df_resampled, features_tensor, target_tensor, valid_pointers, train_loader, val_loader, test_loader, model
    gc.collect()