# api_client.py
import requests
import json
from typing import Dict, Any

class APIClient:
    """
    與即時預測 API 伺服器進行通訊的客戶端。
    """
    def __init__(self, base_url: str = "http://localhost:5000/api"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })

    def _make_request(self, method: str, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        發送 HTTP 請求的通用方法。
        (此函式與您提供的版本相同，保持不變)
        """
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == 'POST':
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"不支援的 HTTP 方法: {method}")
            
            result = response.json()
            
            if not response.ok:
                raise requests.exceptions.HTTPError(
                    f"HTTP {response.status_code}: {result.get('message', '請求失敗')}"
                )
            return result
        except requests.exceptions.RequestException as e:
            print(f"請求錯誤: {e}")
            raise
        except json.JSONDecodeError:
            print(f"JSON 解析錯誤，伺服器原始回應: {response.text}")
            raise

    def get_prediction(self, traffic_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        發送單筆路況資料到伺服器以取得預測結果。

        Args:
            traffic_data (Dict[str, Any]): 包含路況資料的字典。
                                           必須包含 'TimeStamp', 'Occupancy', 'Vehicle_Median_Speed',
                                           'VehicleType_S_Volume', 'VehicleType_L_Volume'。
        Returns:
            Dict[str, Any]: 伺服器的 JSON 回應。
        """
        # 呼叫伺服器的 /api/predict 端點
        return self._make_request('POST', '/predict', traffic_data)