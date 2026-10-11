import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_models():
    indep_dir = "models/independent"
    nn_dir = "models/nn"
    ensemble_dir = "models/ensemble"
    
    # XGBoost models
    xgb_models = {}
    for persona in ['Market', 'Sponsor', 'Coach', 'Agent']:
        print(f"Loading {persona}...")
        xgb_models[persona] = joblib.load(f"{indep_dir}/{persona}_model.joblib")
        
    # NN model
    import torch
    import torch.nn as nn
    
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

    with open(f"{nn_dir}/metadata_nn.json", "r") as f:
        nn_meta = json.load(f)
    nn_model = MarketValueNN(len(nn_meta['features']), nn_meta['best_params']['hidden_dims'], nn_meta['best_params']['dropout'])
    nn_model.load_state_dict(torch.load(f"{nn_dir}/market_value_nn.pth", map_location='cpu'))
    nn_model.eval()
    nn_scaler = joblib.load(f"{nn_dir}/scaler_nn.joblib")
    
    # Meta-Learner (Ridge)
    meta_model = joblib.load(f"{ensemble_dir}/stacking_meta_model.joblib")
    
    return xgb_models, nn_model, nn_scaler, nn_meta['features'], meta_model

def get_stacking_features(df, xgb_models, nn_model, nn_scaler, nn_features):
    # Ensure num_threads is 1 for stability on M-series
    import torch
    torch.set_num_threads(1)
    
    # Use the feature names from the first XGBoost model
    first_model = list(xgb_models.values())[0]
    xgb_features = first_model.get_booster().feature_names
    
    X_xgb = df[xgb_features]
    preds = {}
    for name, model in xgb_models.items():
        preds[f"{name.lower()}"] = model.predict(X_xgb)
        
    # Scaling is ALREADY handled here by nn_scaler
    X_nn = nn_scaler.transform(df[nn_features].fillna(0).values)
    with torch.no_grad():
        preds["nn"] = nn_model(torch.FloatTensor(X_nn)).cpu().numpy().flatten()
        
    return pd.DataFrame(preds)

def main():
    test_df = pd.read_parquet("data/processed/ml_ready/test.parquet")
    
    # Fill NaNs as done in scripts
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df.fillna(0, inplace=True)
    
    xgb_models, nn_model, nn_scaler, nn_features, meta_model = load_models()
    
    X_stack_test = get_stacking_features(test_df, xgb_models, nn_model, nn_scaler, nn_features)
    y_test_log = test_df['log_target_market_value_next_month']
    y_test_real = test_df['target_market_value_next_month']
    
    # Meta-Learner predictions (log scale)
    final_preds_log = meta_model.predict(X_stack_test)
    
    # Inverse transform to get real € values
    final_preds_real = np.expm1(final_preds_log)
    
    # Metrics
    r2 = r2_score(y_test_log, final_preds_log)
    mae_eur = mean_absolute_error(y_test_real, final_preds_real) / 1_000_000
    rmse_eur = np.sqrt(mean_squared_error(y_test_real, final_preds_real)) / 1_000_000
    
    print("\nTable 1: The Generalisation Performance Matrix (Corrected)")
    print("| Metric | Value | Interpretation |")
    print("| :--- | :--- | :--- |")
    print(f"| R^2 (Coefficient of Determination) | {r2:.4f} | Proportion of variance explained |")
    print(f"| MAE (Mean Absolute Error) | ±€{mae_eur:.2f}M | Average financial error (Engineered) |")
    print(f"| RMSE (Root Mean Squared Error) | €{rmse_eur:.2f}M | Sensitivity to fat tails |")

if __name__ == "__main__":
    main()
