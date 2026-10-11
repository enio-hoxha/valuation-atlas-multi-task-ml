import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import os
from src.models.train_nn import MarketValueNN

def load_data(data_dir="data/processed/ml_ready"):
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df.fillna(0, inplace=True)
    return test_df

def main():
    test_df = load_data()
    nn_dir = "models/nn"
    os.makedirs("temp_preds", exist_ok=True)
    
    with open(f"{nn_dir}/metadata_nn.json", "r") as f:
        nn_meta = json.load(f)
    
    nn_model = MarketValueNN(len(nn_meta['features']), nn_meta['best_params']['hidden_dims'], nn_meta['best_params']['dropout'])
    nn_model.load_state_dict(torch.load(f"{nn_dir}/market_value_nn.pth"))
    nn_model.eval()
    nn_scaler = joblib.load(f"{nn_dir}/scaler_nn.joblib")
    
    X_nn = nn_scaler.transform(test_df[nn_meta['features']].values)
    with torch.no_grad():
        preds = nn_model(torch.FloatTensor(X_nn)).cpu().numpy().flatten()
    
    np.save("temp_preds/nn_preds.npy", preds)
    print("NN preds saved.")

if __name__ == "__main__":
    main()
