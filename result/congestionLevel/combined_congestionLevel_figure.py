import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# 定義時間間隔和指標
minutes_list = [5, 15, 30]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
dataframes = {}

# 讀取每個時間間隔的資料
for minutes in minutes_list:
    file_path = f'{minutes}min/GRU_congestionLevel_{minutes}min_metrics_v5.csv'
    try:
        dataframes[minutes] = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"錯誤：在 {file_path} 找不到檔案")
        # 適當地退出或處理錯誤
        exit()

# 建立一個圖表
plt.figure(figsize=(14, 10))
#plt.title('GRU 模型在不同預測時間下的效能指標', fontsize=16)

# 定義顏色和線條樣式
colors = {'Accuracy': 'red', 'Precision': 'lightblue', 'Recall': 'gold', 'F1': 'black'}
markers = {5: 'o', 15: 's', 30: '^'}
linestyles = {5: '-', 15: '-', 30: '-'}

# 繪製每個指標和時間間隔的線條
for metric in metrics:
    for minutes in minutes_list:
        df = dataframes[minutes]
        x = df['Epoch']
        label = f'{metric} ({minutes} min)'
        plt.plot(x, df[metric], label=label, color=colors[metric], 
                linestyle=linestyles[minutes], marker=markers[minutes], 
                markersize=10, markeredgewidth=0.5, markeredgecolor='black')

# 設定圖表屬性
df_5min = dataframes[5]
plt.xticks(df_5min['Epoch'].astype(int),fontsize=20)
plt.ylim(0.4, 1.0)
yticks = [i * 0.05 for i in range(8, 21)]  # 0.4~1.0
yticklabels = [str(int(y * 100)) for y in yticks]  # 40~100
plt.yticks(yticks, yticklabels, fontsize=20)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Accuracy(%)', fontsize=20)

plt.grid(True,alpha=0.3)
plt.legend(loc='lower right', fontsize=20, ncol=2)
# 調整佈局並儲存圖表
plt.tight_layout()
output_filename_png = 'combined_GRU_congestionLevel_metrics_plot_single.png'
output_filename_eps = 'combined_GRU_congestionLevel_metrics_plot_single.eps'
plt.savefig(output_filename_png)
plt.savefig(output_filename_eps)

print(f'=====已儲存圖表至 {output_filename_png}=====')
print(f'=====已儲存圖表至 {output_filename_eps}=====')

plt.show()
