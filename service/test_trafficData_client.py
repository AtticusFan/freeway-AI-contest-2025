import requests
import json
import time
from datetime import datetime, timedelta
import numpy as np
import sys
import random  # --- 新增匯入 ---

# --- 可配置的參數 ---
# 您的 Flask 伺服器運行的端點
API_ENDPOINT = "http://127.0.0.1:5001/api/traffic-data"

# 每隔幾秒傳送一筆資料
SEND_INTERVAL_SECONDS = 5

# --- 要隨機選取的路段 ID 列表 ---
SECTION_IDS = [str(i) for i in range(23, 32)]

# --- 主程式 ---
def run_test_client():
    """
    執行測試客戶端，持續傳送模擬的交通資料到 API 伺服器。
    此版本會隨機選擇 SectionID 進行傳送。
    """
    print("=" * 50)
    print("      交通資料模擬傳送客戶端 (隨機版)      ")
    print("=" * 50)
    print(f"目標伺服器: {API_ENDPOINT}")
    print(f"傳送間隔: {SEND_INTERVAL_SECONDS} 秒")
    print(f"隨機路段ID範圍: {', '.join(SECTION_IDS)}")
    print("\n按下 Ctrl+C 來停止傳送。\n")

    current_timestamp = datetime.now() - timedelta(minutes=10)
    data_point_index = 0

    while True:
        try:
            # --- 修改部分：隨機選取 SectionID ---
            # 從 SECTION_IDS 列表中隨機選取一個元素
            selected_section_id = random.choice(SECTION_IDS)

            # 1. 產生一筆新的模擬交通資料
            payload = {
                "Timestamp": current_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "SectionID": selected_section_id,  # 使用隨機選出的 SectionID
                "Occupancy": round(0.3 + 0.2 * np.random.random(), 4),
                "VehicleType_S_Volume": int(150 + 50 * np.random.random()),
                "VehicleType_L_Volume": int(20 + 15 * np.random.random()),
                "median_speed": round(60 + 25 * np.random.random(), 2)
            }

            print(f"--- [{datetime.now().strftime('%H:%M:%S')}] 準備傳送第 {data_point_index + 1} 筆資料 ---")
            print(json.dumps(payload, indent=2, ensure_ascii=False))

            # 2. 發送 POST 請求
            response = requests.post(
                API_ENDPOINT,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )

            response.raise_for_status()

            # 3. 顯示伺服器的回應
            print(f"\n>>> 伺服器回應 (狀態碼: {response.status_code}):")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
            print("-" * 50 + "\n")

            # 更新下一次的時間戳與索引
            current_timestamp += timedelta(seconds=SEND_INTERVAL_SECONDS)
            data_point_index += 1

            # 4. 等待指定的間隔時間
            time.sleep(SEND_INTERVAL_SECONDS)

        except requests.exceptions.ConnectionError:
            print("\n錯誤：無法連線至伺服器。請確認伺服器腳本正在運行，且 API 端點正確。\n", file=sys.stderr)
            time.sleep(10)
        except requests.exceptions.RequestException as e:
            print(f"\n請求時發生錯誤: {e}\n", file=sys.stderr)
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n\n偵測到使用者中斷操作，正在關閉測試客戶端...")
            break
        except Exception as e:
            print(f"\n發生未預期的錯誤: {e}", file=sys.stderr)
            break

if __name__ == "__main__":
    run_test_client()