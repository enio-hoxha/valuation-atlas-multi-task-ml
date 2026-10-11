import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import json

def load_processed_data(data_dir="data/processed/ml_ready"):
    train = pd.read_parquet(f"{data_dir}/train.parquet")
    val = pd.read_parquet(f"{data_dir}/val.parquet")
    test = pd.read_parquet(f"{data_dir}/test.parquet")
    return train, val, test

def get_raw_features_and_target(df):
    target_col = 'log_target_market_value_next_month'
    
    # EXTREME EXCLUSION: No current value, no wages, no previous deltas
    exclude = [
        'player_id', 'month_date', 'target_market_value_next_month', 
        'target_minutes_next_month', 'target_wiki_views_next_month',
        'log_target_wiki_views_next_month', 'log_target_market_value_next_month',
        'weekly_wage', 'current_market_value', 'wiki_views', 'target_value_change',
        'log_current_market_value', 'log_weekly_wage', 'relative_wage',
        'lag1_delta', 'lag2_delta', 'mv_momentum_3m', 'value_per_minute', 'hype_to_value_ratio'
    ]
    
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols, target_col

if __name__ == "__main__":
    train_df, val_df, test_df = load_processed_data()
    
    # Clean
    for df in [train_df, val_df, test_df]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
    
    features, target = get_raw_features_and_target(train_df)
    print(f"Training 'Raw' Model on {len(features)} performance/context features...")
    print(f"Features: {features}")
    
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    # Train Raw XGBoost
    model = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7, # Allow more depth to find complex performance patterns
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=10,
        random_state=42,
        tree_method='hist',
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    # Evaluate
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print("\n" + "="*30)
    print("Raw Fundamental Model Results:")
    print(f"Test RMSE (Log): {rmse:.4f}")
    print(f"Test R2: {r2:.4f}")
    
    # Save
    model_dir = "models/raw"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/raw_fundamental_model.joblib")
    
    metadata = {
        "features": features,
        "test_rmse": float(rmse),
        "test_r2": float(r2)
    }
    with open(f"{model_dir}/metadata_raw.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"✅ Raw model saved to {model_dir}")
