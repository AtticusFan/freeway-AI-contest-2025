# server.py
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # 允許跨域請求

# 資料儲存檔案路徑
TRAFFIC_DATA_FILE = 'traffic_data.json'

# 記憶體中的資料快取，作為即時存取使用
traffic_records = []

def load_traffic_data():
    """從檔案載入路況資料到記憶體"""
    global traffic_records
    if os.path.exists(TRAFFIC_DATA_FILE):
        try:
            with open(TRAFFIC_DATA_FILE, 'r', encoding='utf-8') as f:
                traffic_records = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"警告: 無法讀取或解析 {TRAFFIC_DATA_FILE}。將以空列表開始。錯誤: {e}")
            traffic_records = []
    else:
        traffic_records = []

def save_traffic_data():
    """將記憶體中的路況資料儲存到檔案"""
    try:
        with open(TRAFFIC_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(traffic_records, f, ensure_ascii=False, indent=4)
    except IOError as e:
        print(f"錯誤: 無法寫入資料到 {TRAFFIC_DATA_FILE}。錯誤: {e}")

# POST - 新增路況資料
@app.route('/api/traffic', methods=['POST'])
def add_traffic_record():
    """接收並儲存一筆新的路況資料"""
    data = request.get_json()
    
    if not data:
        return jsonify({
            "success": False,
            "message": "請求內文必須是 JSON 格式"
        }), 400
        
    required_fields = ["TimeStamp", "Occupancy", "VehicleType_S_Volume", "VehicleType_L_Volume","median_speed"]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({
            "success": False,
            "message": f"缺少必要欄位: {', '.join(missing_fields)}"
        }), 400
        
    try:
        record = {
            "TimeStamp": data['TimeStamp'],
            "Occupancy": float(data['Occupancy']),
            "Vehicle_Median_Speed": float(data['Vehicle_Median_Speed']),
            "VehicleType_S_Volume": int(data['VehicleType_S_Volume']),
            "VehicleType_L_Volume": int(data['VehicleType_L_Volume'])
        }
        # 驗證時間戳格式
        datetime.fromisoformat(record["TimeStamp"].replace(" ", "T"))
    except (ValueError, TypeError) as e:
        return jsonify({
            "success": False,
            "message": f"欄位型別錯誤或時間格式不正確 (應為 YYYY-MM-DD HH:MM:SS): {e}"
        }), 400

    # 4. 新增紀錄並存檔
    traffic_records.append(record)
    save_traffic_data()
    
    return jsonify({
        "success": True,
        "data": record,
        "message": "成功新增路況紀錄"
    }), 201

# GET - 取得所有路況資料
@app.route('/api/traffic', methods=['GET'])
def get_all_traffic_records():
    """回傳所有已儲存的路況資料"""
    return jsonify({
        "success": True,
        "count": len(traffic_records),
        "data": traffic_records,
        "message": "成功取得所有路況紀錄"
    })

# 錯誤處理
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "message": "API 端點不存在"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "message": "伺服器內部錯誤"
    }), 500

if __name__ == '__main__':
    load_traffic_data()  # 伺服器啟動時載入資料
    
    print("Flask API Server 啟動中...")
    print("API 端點:")
    print("  POST   /api/traffic       - 新增路況資料")
    print("  GET    /api/traffic       - 取得所有路況資料")
    print(f"\n路況資料將會儲存於: {os.path.abspath(TRAFFIC_DATA_FILE)}")
    print("伺服器運行在 http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)