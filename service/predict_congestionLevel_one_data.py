import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import joblib
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque

class GRUPredictor(nn.Module):
    """
    Gated Recurrent Unit (GRU) model for time series classification.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, drop_prob=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.gru(x)
        # Decode the hidden state of the last time step
        return self.fc(out[:, -1, :])

class TrafficPredictionClient:
    """
    A client to handle traffic prediction. 
    This version is stateful and designed for real-time, point-by-point data ingestion.
    """
    def __init__(self, model_path: str, scaler_path: str, config: Dict, base_url: str = "http://localhost:5000/api"):
        self.base_url = base_url
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the pre-trained model and scaler
        self.scaler = joblib.load(scaler_path)
        self.model = GRUPredictor(
            input_size=len(config['features']),
            hidden_size=64,
            num_layers=2,
            num_classes=config['num_classes']
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Configure HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })

        # Initialize a fixed-length data queue to manage state
        self.data_queue = deque(maxlen=self.config['seq_len'])

    def add_and_predict(self, new_data_point: Dict) -> Optional[Dict[str, Any]]:
        """
        Receives a single new data point, updates the internal queue, 
        and triggers a prediction when enough data is available.
        
        Args:
            new_data_point: A dictionary representing a single time step of data.
        
        Returns:
            A dictionary with the prediction result if the data queue is full, otherwise None.
        """
        self.data_queue.append(new_data_point)

        if len(self.data_queue) == self.config['seq_len']:
            sequence_to_predict = list(self.data_queue)
            prediction_result = self.predict_single(sequence_to_predict)
            return prediction_result
        else:
            print(f"Data collection in progress... {len(self.data_queue)}/{self.config['seq_len']} points.")
            return None

    def predict_single(self, sequence_data: List[Dict]) -> Dict[str, Any]:
        """
        Performs prediction on a single, complete sequence of data.
        This method is called internally by add_and_predict.
        """
        seq_len = self.config['seq_len']
        features = self.config['features']
        horizon = self.config['horizon']
        
        if len(sequence_data) != seq_len:
            raise ValueError(f"Input sequence length must be {seq_len}, but got {len(sequence_data)}")
        
        # Convert to DataFrame for easier manipulation and sorting
        df = pd.DataFrame(sequence_data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values(by='Timestamp').reset_index(drop=True)
        
        current_time = df['Timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
        section_id = df['SectionID'].iloc[0] if 'SectionID' in df.columns else "Unknown"
        
        # Prepare features for the model
        features_data = df[features]
        scaled_data = self.scaler.transform(features_data.values)
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Perform prediction
        with torch.no_grad():
            logits = self.model(input_tensor)
            prediction = logits.argmax(dim=1).item()
        
        prediction_result = {
            "Section": section_id,
            "當前時間": current_time,
            f"{horizon}分鐘後壅塞程度": prediction,
        }
        return prediction_result
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Generic method for making HTTP requests."""
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == 'GET':
                response = self.session.get(url)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            result = response.json()
            if not response.ok:
                raise requests.exceptions.HTTPError(
                    f"HTTP {response.status_code}: {result.get('message', 'Request failed')}"
                )
            return result
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            raise
    
    def send_prediction(self, prediction_result: Dict) -> Dict[str, Any]:
        """Sends a prediction result to the server."""
        return self._make_request('POST', '/predictions', prediction_result)

def pretty_print(data: Dict[str, Any]) -> None:
    """Pretty-prints a dictionary."""
    print(json.dumps(data, ensure_ascii=False, indent=2))

def real_time_demo():
    """Demonstrates the client in a real-time, point-by-point simulation."""
    MODEL_VERSION = '4'
    HORIZON = 15
    
    CONFIG = {
        'seq_len': 30,
        'horizon': HORIZON,
        'num_classes': 5, 
        'features': [
            'Occupancy', 'VehicleType_S_Volume', 'VehicleType_L_Volume', 
            'median_speed', 'StnPres', 'Temperature', 'RH', 'WS', 'WD', 
            'WSGust', 'WDGust', 'Precip'
        ]
    }

    # NOTE: These paths are placeholders. You must provide the actual paths to your files.
    SCALER_PATH = f'../result/model/GRU_congestionLevel_{HORIZON}min_scaler_v{MODEL_VERSION}.pkl'
    MODEL_PATH = f'../result/model/GRU_congestionLevel_{HORIZON}min_model_v{MODEL_VERSION}.pth'
    
    try:
        # 1. Initialize the stateful client
        stateful_client = TrafficPredictionClient(MODEL_PATH, SCALER_PATH, CONFIG)
        print("Stateful client initialized. Waiting for data...\n")

        # 2. Simulate data arriving one point at a time
        start_time = datetime.now() - timedelta(minutes=200)
        for i in range(40): # Simulate receiving 40 data points
            new_timestamp = start_time + timedelta(minutes=i * 5)
            
            # Generate a single mock data point
            new_data_point = {
                'Timestamp': new_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'SectionID': '23',
                'Occupancy': 0.2 + 0.15 * np.random.random(),
                'VehicleType_S_Volume': 120 + 60 * np.random.random(),
                'VehicleType_L_Volume': 15 + 15 * np.random.random(),
                'median_speed': 70 + 20 * np.random.random(),
                'StnPres': 1010 + 10 * np.random.random(),
                'Temperature': 28 + 5 * np.random.random(),
                'RH': 70 + 15 * np.random.random(),
                'WS': 4 + 4 * np.random.random(),
                'WD': 150 + 100 * np.random.random(),
                'WSGust': 7 + 6 * np.random.random(),
                'WDGust': 180 + 100 * np.random.random(),
                'Precip': 0.05 * np.random.random()
            }
            
            print(f"--- Received new data point for Timestamp: {new_data_point['Timestamp']} ---")
            
            # Feed the new data to the client and attempt to predict
            prediction = stateful_client.add_and_predict(new_data_point)
            
            # 4. Check if a prediction was returned
            if prediction:
                print(">>> Prediction triggered!")
                pretty_print(prediction)
                
                # Optional: Send the prediction to a server
                # try:
                #     server_response = stateful_client.send_prediction(prediction)
                #     print("Server response:")
                #     pretty_print(server_response)
                # except Exception as e:
                #     print(f"Failed to send prediction to server (This is expected in a demo environment): {e}")
            print("-" * 20)

    except FileNotFoundError as e:
        print(f"\nERROR: A required file was not found. Please ensure the model and scaler paths are correct.")
        print(f"Missing file details: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during execution: {e}")

if __name__ == "__main__":
    real_time_demo()