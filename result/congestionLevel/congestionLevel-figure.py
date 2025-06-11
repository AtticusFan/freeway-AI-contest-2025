import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
# 讀取模型效能指標資料
minutes = input("請輸入時間(5、15 或 30 分鐘):")
df = pd.read_csv(f'{minutes}min/GRU_congestionLevel_{minutes}min_metrics_v2.csv')

# 繪製各項指標隨 Epoch 的變化
x = df['Epoch']

plt.figure()
plt.plot(x, df['Accuracy'], label='Accuracy')
plt.plot(x, df['Precision'], label='Precision')
plt.plot(x, df['Recall'], label='Recall')
plt.plot(x, df['F1'], label='F1')
plt.xticks(x.astype(int))
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
plt.ylim(0.4, 1.0)
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title(f'Using GRU to Predict Congestion Level After {minutes} Minutes')
plt.grid()
plt.legend()
plt.savefig(f'{minutes}min/GRU_congestionLevel_{minutes}min_metrics_plot.png')
print(f'=====已儲存圖表至 {minutes}min/GRU_congestionLevel_{minutes}min_metrics_plot.png=====')
