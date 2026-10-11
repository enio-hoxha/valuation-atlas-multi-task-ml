import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

def clean_data(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

def get_features_and_residual_target(df):
    # Calculate the Delta: Target Log MV - Current Log MV
    df['target_log_mv_delta'] = df['log_target_market_value_next_month'] - df['log_current_market_value']
    
    target_col = 'target_log_mv_delta'
    
    # Exclude all target-related columns and raw metadata
    exclude = [
        'player_id', 'month_date', 'target_market_value_next_month', 
        'target_minutes_next_month', 'target_wiki_views_next_month',
        'log_target_wiki_views_next_month', 'log_target_market_value_next_month',
        'weekly_wage', 'current_market_value', 'wiki_views', 'target_value_change',
        'target_log_mv_delta'
    ]
    
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols, target_col

if __name__ == "__main__":
    train_df, val_df, test_df = load_processed_data()
    
    train_df = clean_data(train_df)
    val_df = clean_data(val_df)
    test_df = clean_data(test_df)
    
    # 1. Calculate Persistence Baseline (The "Naïve" Prediction)
    # This is the RMSE if we just predicted next month = this month
    persistence_rmse = np.sqrt(mean_squared_error(
        test_df['log_target_market_value_next_month'], 
        test_df['log_current_market_value']
    ))
    print(f"\n[BASELINE] Persistence RMSE (No-Change assumption): {persistence_rmse:.4f}")
    
    # 2. Prepare Residual Data
    features, target = get_features_and_residual_target(train_df)
    _, _ = get_features_and_residual_target(val_df)
    _, _ = get_features_and_residual_target(test_df)
    
    print(f"Training on {len(features)} features to predict the Value Delta...")
    
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    # 3. Train Residual XGBoost with Monotonic Constraints
    # Define constraints: 1 for positive correlation, -1 for negative, 0 for no constraint
    # We want: Goals, Assists, Minutes, ELO, Momentum to be >= 0 (positive impact on delta)
    # We want: Age to be <= 0 (negative impact on delta)
    
    constraints = {}
    for col in features:
        if any(x in col for x in ['goals', 'assists', 'minutes', 'elo', 'momentum', 'mv']):
            constraints[col] = 1
        elif 'age' in col:
            constraints[col] = -1
        else:
            constraints[col] = 0
            
    mono_const = tuple(constraints[c] for c in features)
    
    print(f"Applying Monotonic Constraints to {sum(x != 0 for x in mono_const)} features...")
    
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=10,
        monotone_constraints=mono_const, # Enforce logic
        objective='reg:squarederror', # Maximize R2 by minimizing variance
        random_state=42,
        tree_method='hist',
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    # 4. Evaluate Reconstructed RMSE
    # Final Prediction = Predicted Delta + Current Log MV
    delta_preds = model.predict(X_test)
    final_preds = delta_preds + test_df['log_current_market_value'].values
    
    reconstructed_rmse = np.sqrt(mean_squared_error(test_df['log_target_market_value_next_month'], final_preds))
    reconstructed_mae = mean_absolute_error(test_df['log_target_market_value_next_month'], final_preds)
    
    print("\n" + "="*30)
    print("Residual Learning Results:")
    print(f"Reconstructed Test RMSE: {reconstructed_rmse:.4f}")
    print(f"Reconstructed Test MAE: {reconstructed_mae:.4f}")
    print(f"Improvement over Persistence Baseline: {((persistence_rmse - reconstructed_rmse) / persistence_rmse * 100):.2f}%")
    
    # Save
    model_dir = "models/residual"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/market_residual_model.joblib")
    
    metadata = {
        "features": features,
        "persistence_rmse": float(persistence_rmse),
        "reconstructed_rmse": float(reconstructed_rmse),
        "reconstructed_mae": float(reconstructed_mae)
    }
    with open(f"{model_dir}/metadata_residual.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"✅ Residual model saved to {model_dir}")
