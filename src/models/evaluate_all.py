import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.models.train_nn import MarketValueNN

def load_data(data_dir="data/processed/ml_ready"):
    """Load and clean test data."""
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df.fillna(0, inplace=True)
    return test_df

def evaluate_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred))
    }

def main():
    test_df = load_data()
    results = {}

    # 1. Independent XGBoost Models
    print("Evaluating Independent XGBoost Models...")
    indep_dir = "models/independent"
    personas = {
        'Market': 'log_target_market_value_next_month',
        'Sponsor': 'log_target_wiki_views_next_month',
        'Coach': 'target_minutes_next_month',
        'Agent': 'target_value_change'
    }
    
    # Identify feature columns for XGB (consistent with train_independent)
    exclude = [
        'player_id', 'month_date', 'target_market_value_next_month', 
        'target_minutes_next_month', 'target_wiki_views_next_month',
        'log_target_wiki_views_next_month', 'log_target_market_value_next_month',
        'weekly_wage', 'current_market_value', 'wiki_views', 'target_value_change'
    ]
    xgb_features = [c for c in test_df.columns if c not in exclude]

    for name, target in personas.items():
        try:
            model = joblib.load(f"{indep_dir}/{name}_model.joblib")
            # Use model's own feature names to avoid mismatch
            current_xgb_features = model.get_booster().feature_names
            
            if target not in test_df.columns and target == 'target_value_change':
                 test_df = test_df.sort_values(['player_id', 'month_date'])
                 test_df['target_value_change'] = test_df.groupby('player_id')['current_market_value'].shift(-1) - test_df['current_market_value']
                 test_df['target_value_change'] = test_df['target_value_change'].fillna(0)
            
            y_true = test_df[target]
            y_pred = model.predict(test_df[current_xgb_features])
            results[f"XGB_{name}"] = evaluate_metrics(y_true, y_pred)
            print(f"✅ XGB_{name} evaluated.")
        except Exception as e:
            print(f"❌ Error evaluating XGB_{name}: {e}")

    # 2. Neural Network
    print("\nEvaluating Neural Network...")
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
            
        results["NeuralNetwork"] = evaluate_metrics(y_true, y_pred)
        print("✅ Neural Network evaluated.")
    except Exception as e:
        print(f"❌ Error evaluating NN: {e}")

    # 3. Meta-Learner (Stacking Ensemble)
    print("\nEvaluating Stacking Ensemble...")
    ens_dir = "models/ensemble"
    try:
        with open(f"{ens_dir}/ensemble_metadata.json", "r") as f:
            ens_meta = json.load(f)
        
        meta_model = joblib.load(f"{ens_dir}/stacking_meta_model.joblib")
        
        # We need the predictions from base models to feed the stacker
        base_preds = {}
        for persona in ['Market', 'Sponsor', 'Coach', 'Agent']:
            model = joblib.load(f"{indep_dir}/{persona}_model.joblib")
            base_preds[f"pred_{persona.lower()}"] = model.predict(test_df[xgb_features])
        
        # NN Preds for stacker
        X_nn = nn_scaler.transform(test_df[nn_meta['features']].values)
        with torch.no_grad():
            base_preds["pred_nn"] = nn_model(torch.FloatTensor(X_nn)).cpu().numpy().flatten()
        
        X_stack = pd.DataFrame(base_preds)
        y_true = test_df['log_target_market_value_next_month']
        y_pred = meta_model.predict(X_stack)
        
        results["StackingEnsemble"] = evaluate_metrics(y_true, y_pred)
        print("✅ Stacking Ensemble evaluated.")
    except Exception as e:
        print(f"❌ Error evaluating Stacking: {e}")

    # Output Results
    print("\n" + "="*50)
    print(f"{'Model':<20} | {'R2':>8} | {'RMSE':>8} | {'MAE':>8}")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"{model_name:<20} | {metrics['R2']:>8.4f} | {metrics['RMSE']:>8.4f} | {metrics['MAE']:>8.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
