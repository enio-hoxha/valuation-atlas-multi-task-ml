import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import json
import random
from tqdm import tqdm

def load_processed_data(data_dir="data/processed/ml_ready"):
    """Load train, val, and test parquets."""
    print(f"Loading data from {data_dir}...")
    train = pd.read_parquet(f"{data_dir}/train.parquet")
    val = pd.read_parquet(f"{data_dir}/val.parquet")
    test = pd.read_parquet(f"{data_dir}/test.parquet")
    return train, val, test

def clean_data(df):
    """Clean NaNs and calculate derived Agent target if missing."""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Calculate value_change (Agent target) if not explicitly present as a target
    if 'target_value_change' not in df.columns:
        # Shifted diff to get next month's change
        df = df.sort_values(['player_id', 'month_date'])
        df['target_value_change'] = df.groupby('player_id')['current_market_value'].shift(-1) - df['current_market_value']
    
    df.fillna(0, inplace=True)
    return df

def get_features(df):
    # Standard excluded columns
    targets = [
        'log_target_market_value_next_month',
        'target_minutes_next_month',
        'log_target_wiki_views_next_month',
        'target_value_change'
    ]
    meta = ['player_id', 'month_date', 'target_market_value_next_month', 'target_wiki_views_next_month']
    raw_logged = ['weekly_wage', 'current_market_value', 'wiki_views']
    
    exclude = targets + meta + raw_logged
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols

def tune_single_model(X_train, y_train, X_val, y_val, persona_name, n_trials=20):
    """Tune a single XGBoost model for a specific persona."""
    print(f"\n>>> Tuning {persona_name} model...")
    
    space = {
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6, 7],
        'min_child_weight': [1, 5, 10, 20],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 1, 10],
        'reg_lambda': [0.1, 1, 10, 100]
    }
    
    best_score = float('inf')
    best_params = None
    best_model = None
    
    for i in range(n_trials):
        params = {k: random.choice(v) for k, v in space.items()}
        
        model = xgb.XGBRegressor(
            n_estimators=1500,
            **params,
            n_jobs=-1,
            random_state=42,
            tree_method='hist',
            early_stopping_rounds=50
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        preds = model.predict(X_val)
        # Use MSE for tuning
        score = mean_squared_error(y_val, preds)
        
        if score < best_score:
            best_score = score
            best_params = params
            best_model = model
            print(f"Trial {i+1} | New Best Score: {score:.6f}")

    return best_model, best_params

if __name__ == "__main__":
    train_df, val_df, test_df = load_processed_data()
    
    train_df = clean_data(train_df)
    val_df = clean_data(val_df)
    test_df = clean_data(test_df)
    
    features = get_features(train_df)
    print(f"Training on {len(features)} features")
    
    personas = {
        'Market': 'log_target_market_value_next_month',
        'Sponsor': 'log_target_wiki_views_next_month',
        'Coach': 'target_minutes_next_month',
        'Agent': 'target_value_change'
    }
    
    model_dir = "models/independent"
    os.makedirs(model_dir, exist_ok=True)
    
    final_metadata = {}
    
    for name, target in personas.items():
        X_tr, y_tr = train_df[features], train_df[target]
        X_va, y_va = val_df[features], val_df[target]
        X_te, y_te = test_df[features], test_df[target]
        
        # Tune
        model, params = tune_single_model(X_tr, y_tr, X_va, y_va, name, n_trials=15)
        
        # Evaluate on Test
        preds = model.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        mae = mean_absolute_error(y_te, preds)
        
        print(f"--- {name} Results ---")
        print(f"Best Params: {params}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        
        # Save
        model_path = f"{model_dir}/{name}_model.joblib"
        joblib.dump(model, model_path)
        
        final_metadata[name] = {
            "target": target,
            "best_params": params,
            "test_rmse": float(rmse),
            "test_mae": float(mae),
            "best_iteration": int(model.best_iteration)
        }
    
    # Save overall metadata
    with open(f"{model_dir}/metadata.json", "w") as f:
        json.dump(final_metadata, f, indent=4)
        
    print(f"\n✅ All 4 independent models trained and saved to {model_dir}")
