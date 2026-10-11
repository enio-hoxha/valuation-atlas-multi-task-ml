import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
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
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

def get_features_and_targets(train_df):
    target_cols = [
        'log_target_market_value_next_month',
        'target_minutes_next_month',
        'log_target_wiki_views_next_month'
    ]
    exclude = target_cols + [
        'player_id', 'month_date', 'target_market_value_next_month', 
        'target_wiki_views_next_month', 'weekly_wage', 'current_market_value', 'wiki_views'
    ]
    feature_cols = [c for c in train_df.columns if c not in exclude]
    return feature_cols, target_cols

def tune_multitask_model(X_train, y_train, X_val, y_val, n_trials=30):
    """Manual Random Search for XGBoost with Early Stopping."""
    
    # Define search space
    space = {
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'min_child_weight': [1, 3, 5, 10, 20],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2, 0.5],
        'reg_alpha': [0, 0.01, 0.1, 1, 10],
        'reg_lambda': [0.01, 0.1, 1, 10, 100]
    }
    
    # Normalization factors (approximate val RMSEs from base model)
    norm_factors = [1.27, 135.0, 0.47]
    
    best_score = float('inf')
    best_params = None
    best_model = None
    
    results = []
    
    print(f"Starting Random Search with {n_trials} trials...")
    
    for i in range(n_trials):
        params = {k: random.choice(v) for k, v in space.items()}
        print(f"\nTrial {i+1}/{n_trials} | Params: {params}")
        
        model = xgb.XGBRegressor(
            n_estimators=2000,
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
        
        # Evaluate on Val
        preds = model.predict(X_val)
        
        rmses = []
        norm_rmses = []
        for j in range(y_val.shape[1]):
            rmse = np.sqrt(mean_squared_error(y_val.iloc[:, j], preds[:, j]))
            rmses.append(rmse)
            norm_rmses.append(rmse / norm_factors[j])
        
        # Combined score: Mean of normalized RMSEs
        combined_score = np.mean(norm_rmses)
        
        print(f"RMSEs: {rmses}")
        print(f"Norm RMSEs: {norm_rmses}")
        print(f"Combined Normalized Score: {combined_score:.4f} (at iteration {model.best_iteration})")
        
        results.append({
            'params': params,
            'rmses': rmses,
            'combined_score': combined_score,
            'best_iteration': model.best_iteration
        })
        
        if combined_score < best_score:
            best_score = combined_score
            best_params = params
            best_model = model
            print("New Best Model!")

    return best_model, best_params, results

if __name__ == "__main__":
    train_df, val_df, test_df = load_processed_data()
    
    train_df = clean_data(train_df)
    val_df = clean_data(val_df)
    test_df = clean_data(test_df)
    
    features, targets = get_features_and_targets(train_df)
    
    X_train, y_train = train_df[features], train_df[targets]
    X_val, y_val = val_df[features], val_df[targets]
    X_test, y_test = test_df[features], test_df[targets]
    
    # Run tuning
    best_model, best_params, all_results = tune_multitask_model(X_train, y_train, X_val, y_val, n_trials=30)
    
    print("\n" + "="*30)
    print("Tuning Finished!")
    print(f"Best Params: {best_params}")
    
    # Evaluate Best Model
    preds = best_model.predict(X_test)
    test_metrics = {}
    for i, name in enumerate(targets):
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], preds[:, i]))
        test_metrics[name] = float(rmse)
        print(f"Test Task: {name} | RMSE: {rmse:.4f}")
        
    # Save Best Model & Results
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = f"{model_dir}/multitask_xgb_tuned.joblib"
    joblib.dump(best_model, model_path)
    
    metadata = {
        "best_params": best_params,
        "features": features,
        "targets": targets,
        "test_metrics": test_metrics,
        "tuning_results": all_results
    }
    
    with open(f"{model_dir}/tuning_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"✅ Tuned model saved to {model_path}")
