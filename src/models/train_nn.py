import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import json
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def load_processed_data(data_dir="data/processed/ml_ready"):
    """Load train, val, and test parquets."""
    print(f"Loading data from {data_dir}...")
    train = pd.read_parquet(f"{data_dir}/train.parquet")
    val = pd.read_parquet(f"{data_dir}/val.parquet")
    test = pd.read_parquet(f"{data_dir}/test.parquet")
    return train, val, test

def clean_data(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

def get_features_and_target(df):
    target_col = 'log_target_market_value_next_month'
    
    # Exclude other targets and meta
    exclude = [
        'player_id', 'month_date', 'target_market_value_next_month', 
        'target_minutes_next_month', 'target_wiki_views_next_month',
        'log_target_wiki_views_next_month', 'log_target_market_value_next_month',
        'weekly_wage', 'current_market_value', 'wiki_views'
    ]
    
    feature_cols = [c for c in df.columns if c not in exclude]
    # Ensure log_current_market_value is present (adamant requirement)
    if 'log_current_market_value' not in feature_cols:
        print("WARNING: log_current_market_value not found in features. Adding it back.")
        feature_cols.append('log_current_market_value')
        
    return feature_cols, target_col

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

def train_nn(X_train, y_train, X_val, y_val, params, device):
    """Train the NN with early stopping."""
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).view(-1, 1))
    
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=params['batch_size'])
    
    model = MarketValueNN(X_train.shape[1], params['hidden_dims'], params['dropout']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 5
    trigger_times = 0
    
    for epoch in range(10): # Quick run
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            best_model_state = model.state_dict()
        else:
            trigger_times += 1
            if trigger_times >= patience:
                break
                
    model.load_state_dict(best_model_state)
    return model, best_val_loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_df, val_df, test_df = load_processed_data()
    train_df = clean_data(train_df)
    val_df = clean_data(val_df)
    test_df = clean_data(test_df)
    
    features, target = get_features_and_target(train_df)
    print(f"Training on {len(features)} features.")
    
    X_train_raw, y_train = train_df[features].values, train_df[target].values
    X_val_raw, y_val = val_df[features].values, val_df[target].values
    X_test_raw, y_test = test_df[features].values, test_df[target].values
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    
    tuning_space = [
        {'hidden_dims': [128, 64], 'dropout': 0.1, 'lr': 0.001, 'batch_size': 128, 'weight_decay': 1e-5},
    ]
    
    best_overall_loss = float('inf')
    best_params = None
    best_model = None
    
    print("Starting optimization...")
    for i, params in enumerate(tuning_space):
        model, val_loss = train_nn(X_train, y_train, X_val, y_val, params, device)
        print(f"Trial {i+1} | Params: {params} | Val Loss: {val_loss:.6f}")
        if val_loss < best_overall_loss:
            best_overall_loss = val_loss
            best_params = params
            best_model = model
            
    best_model.eval()
    with torch.no_grad():
        preds = best_model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    
    print(f"\nFinal Neural Network Results:")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    model_dir = "models/nn"
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(best_model.state_dict(), f"{model_dir}/market_value_nn.pth")
    joblib.dump(scaler, f"{model_dir}/scaler_nn.joblib")
    
    metadata = {
        "features": features,
        "best_params": best_params,
        "test_rmse": float(rmse),
        "test_mae": float(mae)
    }
    with open(f"{model_dir}/metadata_nn.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"✅ Neural Network saved to {model_dir}")
