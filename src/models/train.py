import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

def load_processed_data(data_dir="data/processed/ml_ready"):
    """Load train, val, and test parquets."""
    print(f"Loading data from {data_dir}...")
    train = pd.read_parquet(f"{data_dir}/train.parquet")
    val = pd.read_parquet(f"{data_dir}/val.parquet")
    test = pd.read_parquet(f"{data_dir}/test.parquet")
    return train, val, test

def train_multitask_model(train_df, val_df):
    """Train a Multi-Task XGBoost Regressor with standardized features."""
    
    # Define Targets
    target_cols = [
        'log_target_market_value_next_month',
        'target_minutes_next_month',
        'log_target_wiki_views_next_month'
    ]
    
    # Define Features (Exclude targets and non-numeric metadata)
    exclude = target_cols + [
        'player_id', 'month_date', 'target_market_value_next_month', 
        'target_wiki_views_next_month'
    ]
    # Also exclude the raw versions of logged features
    exclude += ['weekly_wage', 'current_market_value', 'wiki_views']
    
    feature_cols = [c for c in train_df.columns if c not in exclude]
    
    print(f"Training on {len(feature_cols)} features...")
    if 'tactical_cluster' in feature_cols:
        print("✅ tactical_cluster detected in feature set.")
    
    # Standardizing features
    print("Standardizing features using StandardScaler...")
    scaler = StandardScaler()
    X_train_raw = train_df[feature_cols]
    y_train = train_df[target_cols]
    
    X_val_raw = val_df[feature_cols]
    y_val = val_df[target_cols]
    
    # Fit on train, transform both
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    
    # Initialize Multi-Task XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        tree_method='hist',
        early_stopping_rounds=50
    )
    
    print("Fitting model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    return model, feature_cols, scaler

def evaluate_tasks(model, X_test, y_test, target_names):
    """Calculate RMSE for each task independently."""
    preds = model.predict(X_test)
    
    results = {}
    for i, name in enumerate(target_names):
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], preds[:, i]))
        results[name] = float(rmse)
        print(f"Task: {name} | RMSE: {rmse:.4f}")
    
    return results

if __name__ == "__main__":
    train_df, val_df, test_df = load_processed_data()
    
    # Clean up NaNs from momentum features
    for df in [train_df, val_df, test_df]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
    
    model, features, scaler = train_multitask_model(train_df, val_df)
    
    # Evaluate
    target_cols = [
        'log_target_market_value_next_month',
        'target_minutes_next_month',
        'log_target_wiki_views_next_month'
    ]
    # Apply same scaler to test set
    X_test = scaler.transform(test_df[features])
    y_test = test_df[target_cols]
    
    metrics = evaluate_tasks(model, X_test, y_test, target_cols)
    
    # Save Model and Scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/multitask_xgb_v1.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    print(f"✅ Model and Scaler saved to models/")

    
    # Save Model & Metadata
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = f"{model_dir}/multitask_xgb_v1.joblib"
    joblib.dump(model, model_path)
    
    metadata = {
        "features": features,
        "targets": target_cols,
        "metrics": metrics
    }
    with open(f"{model_dir}/metadata_v1.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"✅ Model saved to {model_path}")
