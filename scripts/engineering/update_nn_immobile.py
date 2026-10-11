import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import json
import os

# Define the NN class
class MarketValueNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.2):
        super(MarketValueNN, self).__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, 1))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

def get_nn_pred(player_id, nn_dir="models/nn", data_dir="data/processed/ml_ready"):
    with open(f"{nn_dir}/metadata_nn.json", "r") as f:
        nn_meta = json.load(f)
    
    model = MarketValueNN(len(nn_meta['features']), nn_meta['best_params']['hidden_dims'], nn_meta['best_params']['dropout'])
    model.load_state_dict(torch.load(f"{nn_dir}/market_value_nn.pth", map_location=torch.device('cpu')))
    model.eval()
    
    scaler = joblib.load(f"{nn_dir}/scaler_nn.joblib")
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    
    p_data = test_df[test_df['player_id'] == player_id].sort_values('month_date', ascending=False).iloc[[0]]
    if p_data.empty:
        return None
        
    X_nn_raw = p_data[nn_meta['features']].values
    X_nn = scaler.transform(X_nn_raw)
    
    with torch.no_grad():
        pred = model(torch.FloatTensor(X_nn)).item()
    
    return pred

if __name__ == "__main__":
    pid = 105521
    pred = get_nn_pred(pid)
    print(f"NN Prediction for {pid}: {pred}")
    
    # Save to temp file to be picked up by generate_player_shaps.py
    if pred is not None:
        with open("nn_preds_temp.json", "r") as f:
            nn_preds = json.load(f)
        nn_preds[str(pid)] = pred
        with open("nn_preds_temp.json", "w") as f:
            json.dump(nn_preds, f, indent=4)
        print("Updated nn_preds_temp.json")
