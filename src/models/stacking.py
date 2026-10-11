import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import json
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from train_nn import MarketValueNN # Import the NN class definition

def load_all_models(indep_dir="models/independent", nn_dir="models/nn"):
    models = {}
    # Load 4 XGBoost Models
    for persona in ['Market', 'Sponsor', 'Coach', 'Agent']:
        models[persona] = joblib.load(f"{indep_dir}/{persona}_model.joblib")
    
    # Load NN
    with open(f"{nn_dir}/metadata_nn.json", "r") as f:
        nn_meta = json.load(f)
    
    nn_model = MarketValueNN(len(nn_meta['features']), nn_meta['best_params']['hidden_dims'], nn_meta['best_params']['dropout'])
    nn_model.load_state_dict(torch.load(f"{nn_dir}/market_value_nn.pth"))
    nn_model.eval()
    
    nn_scaler = joblib.load(f"{nn_dir}/scaler_nn.joblib")
    
    return models, nn_model, nn_scaler, nn_meta['features']

def generate_stacking_features(df, xgb_models, nn_model, nn_scaler, nn_features):
    # Prepare features for XGB
    # Re-use logic from train_independent to get feature names if not in metadata
    # But usually it's better to be explicit.
    exclude = [
        'player_id', 'month_date', 'target_market_value_next_month', 
        'target_minutes_next_month', 'target_wiki_views_next_month',
        'log_target_wiki_views_next_month', 'log_target_market_value_next_month',
        'weekly_wage', 'current_market_value', 'wiki_views', 'target_value_change'
    ]
    xgb_features = [c for c in df.columns if c not in exclude]
    
    X_xgb = df[xgb_features]
    
    preds = {}
    for name, model in xgb_models.items():
        preds[f"pred_{name.lower()}"] = model.predict(X_xgb)
        
    # NN Preds
    X_nn = nn_scaler.transform(df[nn_features].values)
    with torch.no_grad():
        preds["pred_nn"] = nn_model(torch.FloatTensor(X_nn)).cpu().numpy().flatten()
        
    return pd.DataFrame(preds)

if __name__ == "__main__":
    from train_independent import clean_data # Reuse cleaner
    
    data_dir = "data/processed/ml_ready"
    train_df = pd.read_parquet(f"{data_dir}/train.parquet")
    val_df = pd.read_parquet(f"{data_dir}/val.parquet")
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    
    train_df = clean_data(train_df)
    val_df = clean_data(val_df)
    test_df = clean_data(test_df)
    
    xgb_models, nn_model, nn_scaler, nn_features = load_all_models()
    
    print("Generating stacking features...")
    # To avoid overfitting on training data, we use the validation set to train the stacker
    # and then evaluate on the test set. 
    # (A better way is K-fold OOF, but this is a solid start)
    X_stack_val = generate_stacking_features(val_df, xgb_models, nn_model, nn_scaler, nn_features)
    y_stack_val = val_df['log_target_market_value_next_month']
    
    X_stack_test = generate_stacking_features(test_df, xgb_models, nn_model, nn_scaler, nn_features)
    y_stack_test = test_df['log_target_market_value_next_month']
    
    print(f"Stacking features: {X_stack_val.columns.tolist()}")
    
    # Train Stacker (Meta-Learner)
    # Ridge is great for stacking as it handles correlations between model predictions
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_stack_val, y_stack_val)
    
    # Evaluate
    final_preds = meta_model.predict(X_stack_test)
    rmse = np.sqrt(mean_squared_error(y_stack_test, final_preds))
    mae = mean_absolute_error(y_stack_test, final_preds)
    
    print("\n" + "="*30)
    print("Stacking Ensemble Results:")
    print(f"Final Ensemble RMSE: {rmse:.4f}")
    print(f"Final Ensemble MAE: {mae:.4f}")
    
    # Compare with best single model (Market XGB)
    market_xgb_preds = X_stack_test['pred_market']
    base_rmse = np.sqrt(mean_squared_error(y_stack_test, market_xgb_preds))
    improvement = (base_rmse - rmse) / base_rmse * 100
    print(f"Baseline (Market XGB) RMSE: {base_rmse:.4f}")
    print(f"Ensemble Improvement: {improvement:.2f}%")
    
    # Save Stacker
    model_dir = "models/ensemble"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(meta_model, f"{model_dir}/stacking_meta_model.joblib")
    
    metadata = {
        "stacker_type": "Ridge",
        "input_models": X_stack_val.columns.tolist(),
        "test_rmse": float(rmse),
        "test_mae": float(mae),
        "coefficients": dict(zip(X_stack_val.columns, meta_model.coef_.tolist()))
    }
    with open(f"{model_dir}/ensemble_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"✅ Stacking ensemble saved to {model_dir}")
