import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. 讀取並預處理資料
DATA_PATH = '../datasets/merged_traffic_weather_0501_0521.csv'
df = pd.read_csv(DATA_PATH, parse_dates=['Timestamp'])
df = df.sort_values(['SectionID', 'Timestamp'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.set_index('Timestamp')

# 2. SectionID 分群 + 5 分鐘重取樣 (取中位數)
agg_cols = [
    'Status','LaneID','LaneType','LaneSpeed','Occupancy',
    'VehicleType_S_Volume','VehicleType_S_Speed',
    'VehicleType_L_Volume','VehicleType_L_Speed',
    'VehicleType_T_Volume','VehicleType_T_Speed',
    'TravelTime','TravelSpeed','StnPres','Temperature','RH',
    'WS','WD','WSGust','WDGust','Precip','CongestionLevel'
]
agg_cfg = {col: 'median' for col in agg_cols}
df_resampled = df.groupby('SectionID').resample('5min').agg(agg_cfg).dropna()

# 3. 建立時序序列 (sliding window)
FEATURES = [c for c in agg_cols if c != 'CongestionLevel']
TARGET = 'CongestionLevel'
# 使用過去60分鐘 (12個5分鐘窗口) 預測30分鐘後 (6個5分鐘窗口) 的壅塞等級
SEQ_LEN, HORIZON = 12, 6

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

# 7. 定義單向 GRU + Attention + 正則化模型
class GRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, drop_prob=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=drop_prob,
            bidirectional=False
        )
        self.attn_fc = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        scores = self.attn_fc(out)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * out, dim=1)
        context = self.layer_norm(context)
        context = self.dropout(context)
        return self.fc(context)

# 8. 訓練參數與裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(np.unique(y))
model = GRUWithAttention(
    input_size=nfeatures,
    hidden_size=64,
    num_layers=2,
    num_classes=num_classes
).to(device)
EPOCHS = 50
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2,eta_min=1e-6)


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
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
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
            y_true.extend(yb.cpu().numpy()); y_pred.extend(preds.cpu().numpy())
    val_loss /= len(val_loader.dataset)

    scheduler.step(epoch + (0))

    # metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # test
    test_loss = 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            test_loss += criterion(model(Xb), yb).item() * Xb.size(0)
    test_loss /= len(test_loader.dataset)

    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}/{EPOCHS}  Train: {train_loss:.4f}  Val: {val_loss:.4f}  Test: {test_loss:.4f}  "
          f"Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}  LR: {lr:.6f}")
    records.append({'Epoch': epoch, 'Train Loss': train_loss,
                    'Val Loss': val_loss, 'Test Loss': test_loss, 'LR': lr})
    metrics_records.append({'Epoch': epoch, 'Accuracy': acc,
                            'Precision': prec, 'Recall': rec, 'F1': f1})

# 10. 儲存結果與模型
version = input("請輸入模型版本號: ")
ume_losses = pd.DataFrame(records)
umetrics = pd.DataFrame(metrics_records)
ume_losses.to_csv(f'Attention-GRU_0501_0521_data_v{version}.csv', index=False)
umetrics.to_csv(f'Attention-GRU_0501_0521_data_metrics_v{version}.csv', index=False)
torch.save(model.state_dict(), f'gru_model_v{version}.pth')
