import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.models.train_nn import MarketValueNN

def load_data(data_dir="data/processed/ml_ready"):
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df.fillna(0, inplace=True)
    return test_df

def main():
    test_df = load_data()
    nn_dir = "models/nn"
    try:
        with open(f"{nn_dir}/metadata_nn.json", "r") as f:
            nn_meta = json.load(f)
        
        nn_model = MarketValueNN(len(nn_meta['features']), nn_meta['best_params']['hidden_dims'], nn_meta['best_params']['dropout'])
        nn_model.load_state_dict(torch.load(f"{nn_dir}/market_value_nn.pth"))
        nn_model.eval()
        
        nn_scaler = joblib.load(f"{nn_dir}/scaler_nn.joblib")
        
        X_nn = nn_scaler.transform(test_df[nn_meta['features']].values)
        y_true = test_df['log_target_market_value_next_month']
        
        with torch.no_grad():
            y_pred = nn_model(torch.FloatTensor(X_nn)).cpu().numpy().flatten()
            
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        print(f"NeuralNetwork | R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    except Exception as e:
        print(f"Error evaluating NN: {e}")

if __name__ == "__main__":
    main()
