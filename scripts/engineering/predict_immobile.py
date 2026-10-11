import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
import joblib
import os
import json
import duckdb
import torch
import torch.nn as nn
import sys

# Define the NN class as it was in train_nn.py
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

def load_all_models(indep_dir="models/independent", ensemble_dir="models/ensemble", nn_dir="models/nn"):
    print("Loading XGBoost models...")
    xgb_models = {}
    for persona in ['Market', 'Sponsor', 'Coach', 'Agent']:
        xgb_models[persona] = joblib.load(f"{indep_dir}/{persona}_model.joblib")
    
    print("Loading NN metadata...")
    with open(f"{nn_dir}/metadata_nn.json", "r") as f:
        nn_meta = json.load(f)
    
    print("Initializing NN model...")
    nn_model = MarketValueNN(len(nn_meta['features']), nn_meta['best_params']['hidden_dims'], nn_meta['best_params']['dropout'])
    print("Loading NN weights...")
    nn_model.load_state_dict(torch.load(f"{nn_dir}/market_value_nn.pth", map_location=torch.device('cpu')))
    nn_model.eval()
    
    print("Loading NN scaler...")
    nn_scaler = joblib.load(f"{nn_dir}/scaler_nn.joblib")
    
    print("Loading Meta-model...")
    meta_model = joblib.load(f"{ensemble_dir}/stacking_meta_model.joblib")
    
    print("Loading Ensemble metadata...")
    with open(f"{ensemble_dir}/ensemble_metadata.json", "r") as f:
        ensemble_meta = json.load(f)
    
    return xgb_models, nn_model, nn_scaler, nn_meta['features'], meta_model, ensemble_meta

def predict_player(player_id):
    xgb_models, nn_model, nn_scaler, nn_features, meta_model, ensemble_meta = load_all_models()
    
    print("Loading test data...")
    test_df = pd.read_parquet("data/processed/ml_ready/test.parquet")
    
    p_data = test_df[test_df['player_id'] == player_id].sort_values('month_date', ascending=False).iloc[[0]]
    if p_data.empty:
        print("No data for player")
        return

    # 1. XGBoost Persona Preds
    print("Running XGBoost predictions...")
    persona_map = {'Market': 'market', 'Sponsor': 'sponsor', 'Coach': 'coach', 'Agent': 'agent'}
    xgb_preds = {}
    for persona, model_key in persona_map.items():
        m = xgb_models[persona]
        m_feats = m.get_booster().feature_names
        p_data_fixed = p_data[m_feats]
        xgb_preds[model_key] = m.predict(p_data_fixed)[0]
    
    # 2. NN Prediction
    print("Running NN prediction...")
    X_nn_raw = p_data[nn_features].values
    X_nn = nn_scaler.transform(X_nn_raw)
    with torch.no_grad():
        pred_nn = nn_model(torch.FloatTensor(X_nn)).item()
    
    # 3. Ensemble Prediction (Stacking)
    print("Running Ensemble prediction...")
    # Input order: market, sponsor, coach, agent, nn
    input_data = pd.DataFrame([[
        xgb_preds['market'],
        xgb_preds['sponsor'],
        xgb_preds['coach'],
        xgb_preds['agent'],
        pred_nn
    ]], columns=['market', 'sponsor', 'coach', 'agent', 'nn'])
    
    final_log_val = meta_model.predict(input_data)[0]
    final_val = np.exp(final_log_val)
    
    print(f"\nSUCCESS!")
    print(f"Player: Ciro Immobile (ID: {player_id})")
    print(f"Observation Date: {p_data['month_date'].values[0]}")
    print(f"Current Market Value: €{p_data['current_market_value'].values[0]:,.0f}")
    print(f"Ensemble Predicted Value: €{final_val:,.0f}")
    print("-" * 30)

if __name__ == "__main__":
    predict_player(105521)
