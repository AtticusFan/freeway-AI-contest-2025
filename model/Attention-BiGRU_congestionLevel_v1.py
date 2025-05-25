import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. 讀取並預處理資料
DATA_PATH = '../datasets/merged_traffic_weather_0501_0521.csv'
df = pd.read_csv(DATA_PATH, parse_dates=['Timestamp'])
df = df.sort_values(['SectionID', 'Timestamp'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.set_index('Timestamp')

# 2. SectionID 分群 + 10 分鐘重取樣 (取中位數)
agg_cols = [
    'Status','LaneID','LaneType','LaneSpeed','Occupancy',
    'VehicleType_S_Volume','VehicleType_S_Speed',
    'VehicleType_L_Volume','VehicleType_L_Speed',
    'VehicleType_T_Volume','VehicleType_T_Speed',
    'TravelTime','TravelSpeed','StnPres','Temperature','RH',
    'WS','WD','WSGust','WDGust','Precip','CongestionLevel'
]
agg_cfg = {col: 'median' for col in agg_cols}
df_resampled = df.groupby('SectionID').resample('10T').agg(agg_cfg).dropna()

# 3. 建立時序序列 (sliding window)
FEATURES = [c for c in agg_cols if c != 'CongestionLevel']
TARGET = 'CongestionLevel'
SEQ_LEN, HORIZON = 6, 1

sequences, targets = [], []
for sec, grp in df_resampled.groupby(level=0):
    arr = grp[FEATURES + [TARGET]].values
    for i in range(len(arr) - SEQ_LEN - HORIZON + 1):
        sequences.append(arr[i:i+SEQ_LEN, :-1])
        targets.append(int(arr[i+SEQ_LEN+HORIZON-1, -1]))
X = np.stack(sequences)
y = np.array(targets, dtype=int)

# 4. 正規化
nsamples, _, nfeatures = X.shape
X_flat = X.reshape(-1, nfeatures)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_flat)
X = X_scaled.reshape(nsamples, SEQ_LEN, nfeatures)

# 5. 切分 train/val/test (70/15/15)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
val_ratio = 0.15 / 0.85
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, shuffle=False)

# 6. DataLoader
batch_size = 128
def create_loader(X, y, shuffle=False):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
train_loader = create_loader(X_train, y_train, shuffle=True)
val_loader   = create_loader(X_val,   y_val)
test_loader  = create_loader(X_test,  y_test)

# 7. 定義 Bidirectional GRU + Attention 模型
class BiGRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, drop_prob=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size,
                          hidden_size,
                          num_layers,
                          batch_first=True,
                          dropout=drop_prob,
                          bidirectional=True)
        # 注意力對接受 2*hidden_size 輸出計分
        self.attn_fc = nn.Linear(hidden_size*2, 1)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, _ = self.gru(x)  # out: [batch, seq_len, 2*hidden_size]
        # attention scores
        scores = self.attn_fc(out)           # [batch, seq_len, 1]
        weights = torch.softmax(scores, dim=1)  # [batch, seq_len, 1]
        # context vector
        context = torch.sum(weights * out, dim=1)  # [batch, 2*hidden_size]
        return self.fc(context)  # [batch, num_classes]

# 8. 訓練參數與裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(np.unique(y))
model = BiGRUWithAttention(input_size=nfeatures,
                           hidden_size=64,
                           num_layers=2,
                           num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 50

# 9. 訓練與評估
records, metrics_records = [], []
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

    # val & metrics
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

    # test
    test_loss = 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            test_loss += criterion(logits, yb).item() * Xb.size(0)
    test_loss /= len(test_loader.dataset)

    print(f"Epoch {epoch}/{EPOCHS}  "
          f"Train {train_loss:.4f}  Val {val_loss:.4f}  Test {test_loss:.4f}  "
          f"Acc {acc:.4f}  Prec {prec:.4f}  Rec {rec:.4f}  F1 {f1:.4f}")
    records.append({'Epoch': epoch,
                    'Train Loss': train_loss,
                    'Test Loss': test_loss,
                    'Val Loss': val_loss
                    })
    metrics_records.append({'Epoch': epoch,
                            'Accuracy': acc,
                            'Precision': prec,
                            'Recall': rec,
                            'F1': f1})

# 10. 寫檔
version = input("請輸入模型版本號: ")
file_name_loss = "Attention-BiGRU_0501_0521_data_v" + version
file_name_metrics = "Attention-BiGRU_0501_0521_data_metrics_v" + version
pd.DataFrame(records).to_csv(f'{file_name_loss}.csv', index=False)
pd.DataFrame(metrics_records).to_csv(f'{file_name_metrics}.csv', index=False)

# 11. 儲存模型
#torch.save(model.state_dict(), f'gru_model_v{version}.pth')
