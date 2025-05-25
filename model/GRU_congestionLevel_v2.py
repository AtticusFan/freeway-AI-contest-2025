import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. 讀取並預處理資料
DATA_PATH = '../datasets/merged_traffic_weather_0501_0521.csv'
df = pd.read_csv(DATA_PATH, parse_dates=['Timestamp'])

df = df.sort_values(['SectionID', 'Timestamp'])
# 1.1 設定 Timestamp 為索引
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.set_index('Timestamp')

# 2. 以 SectionID 分群 + 10 分鐘重取樣 (取中位數)
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
      .resample('10T')
      .agg(agg_cfg)
      .dropna()
)

# 3. 建立時序序列 (sliding window)
FEATURES = [
    'Status','LaneID','LaneType','LaneSpeed','Occupancy',
    'VehicleType_S_Volume','VehicleType_S_Speed',
    'VehicleType_L_Volume','VehicleType_L_Speed',
    'VehicleType_T_Volume','VehicleType_T_Speed',
    'TravelTime','TravelSpeed',
    'StnPres','Temperature','RH','WS','WD','WSGust','WDGust','Precip'
]
TARGET = 'CongestionLevel'
SEQ_LEN = 6  # 前6步 (60分鐘)
HORIZON = 1  # 預測下一10分鐘

sequences, targets = [], []
for sec, grp in df_resampled.groupby(level=0):
    arr = grp[FEATURES + [TARGET]].values
    for i in range(len(arr) - SEQ_LEN - HORIZON + 1):
        sequences.append(arr[i:i+SEQ_LEN, :-1])
        targets.append(arr[i+SEQ_LEN+HORIZON-1, -1])
X = np.stack(sequences)  # [N, SEQ_LEN, F]
y = np.array(targets, dtype=int)    # [N], 類別標籤

# 4. 正規化
nsamples, _, nfeatures = X.shape
X_flat = X.reshape(-1, nfeatures)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_flat)
X = X_scaled.reshape(nsamples, SEQ_LEN, nfeatures)

# 5. 切分訓練/驗證/測試集 (70% train, 15% val, 15% test)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
val_ratio = 0.15 / 0.85
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, shuffle=False)

# 6. Dataset & DataLoader
from torch.utils.data import TensorDataset

def create_loader(X, y, batch_size=128, shuffle=False):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = create_loader(X_train, y_train, shuffle=True)
val_loader   = create_loader(X_val,   y_val)
test_loader  = create_loader(X_test,  y_test)

# 7. 定義 GRU 模型 (分類任務)
class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, drop_prob=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# 8. 訓練參數與裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(np.unique(y))
model = GRUPredictor(len(FEATURES), 64, 2, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 50

# 9. 訓練迴圈，計算 Loss 與 Metrics
metrics_records = []
records = []
for epoch in range(1, EPOCHS+1):
    # train
    model.train()
    train_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_loss += loss.item() * Xb.size(0)
    train_loss /= len(train_loader.dataset)
    # val
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
    # test loss per epoch
    test_loss = 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            test_loss += criterion(logits, yb).item() * Xb.size(0)
    test_loss /= len(test_loader.dataset)
    # record
    print(f"Epoch {epoch}/{EPOCHS}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Test Loss: {test_loss:.4f}")
    records.append({'Epoch': epoch, 'Train Loss': train_loss, 'Val Loss': val_loss, 'Test Loss': test_loss})
    metrics_records.append({'Epoch': epoch, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1})

# 10. 寫檔
model_version = input("請輸入模型版本號: ")
pd.DataFrame(records).to_csv(f'GRU_0501_0521_data_v{model_version}.csv', index=False)
pd.DataFrame(metrics_records).to_csv(f'GRU_0501_0521_data_metrics_v{model_version}.csv', index=False)

