import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# 讀取 Accuracy 數值
acc5 = pd.read_csv('5min/GRU_TravelTime_5min_metrics_v3.csv')['Accuracy'].iloc[0]
acc15 = pd.read_csv('15min/GRU_TravelTime_15min_metrics_v3.csv')['Accuracy'].iloc[0]
acc30 = pd.read_csv('30min/GRU_TravelTime_30min_metrics_v3.csv')['Accuracy'].iloc[0]

labels = ['5min', '15min', '30min']
accuracies = [acc5, acc15, acc30]
colors = ['skyblue', 'lightcoral', 'lightgreen']

# 建立圖形與座標軸
fig, ax = plt.subplots()

# 繪製長條圖並設定 zorder 讓 bars 在 grid 上方
bars = ax.bar(labels, accuracies, color=colors, zorder=3)

# 顯示 Y 軸網格線，並設定 zorder 於 bars 之下
ax.yaxis.grid(True, zorder=0)

# Y 軸百分比與範圍設定
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.set_ylim(0.4, 1.0)

# 標籤與標題
ax.set_xlabel('Minutes After')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Using GRU to Predict Travel Time')

# 儲存圖表
plt.tight_layout()
fig.savefig('GRU_TravelTime_accuracy_plot_v3.png', dpi=300)
print('===== 已儲存圖表至 GRU_TravelTime_accuracy_plot_v3.png =====')
