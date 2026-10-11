import pandas as pd
import numpy as np
import shap
import joblib
import os
import json
import matplotlib.pyplot as plt

def load_data_and_models(model_dir="models/independent", data_dir="data/processed/ml_ready"):
    """Load models and test data."""
    print(f"Loading metadata and test data...")
    with open(f"{model_dir}/metadata.json", "r") as f:
        metadata = json.load(f)
    
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    
    # Standard cleaning (same as training)
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if 'target_value_change' not in test_df.columns:
        test_df = test_df.sort_values(['player_id', 'month_date'])
        test_df['target_value_change'] = test_df.groupby('player_id')['current_market_value'].shift(-1) - test_df['current_market_value']
    test_df.fillna(0, inplace=True)
    
    return metadata, test_df

def run_treeshap(model_path, X_test, persona_name):
    """Run TreeSHAP for a single model."""
    print(f"Running TreeSHAP for {persona_name}...")
    model = joblib.load(model_path)
    
    # Initialize Explainer
    explainer = shap.TreeExplainer(model)
    
    # Sampling for speed if dataset is huge, but here we'll use a representative sample or the whole test set
    # Given ~18k test rows, we can do the whole thing if it's not too slow, or sample 2000
    if len(X_test) > 2000:
        X_sample = X_test.sample(2000, random_state=42)
    else:
        X_sample = X_test
        
    shap_values = explainer.shap_values(X_sample, check_additivity=False)
    
    # Calculate global importance (mean absolute SHAP)
    importance = np.abs(shap_values).mean(0)
    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return shap_values, X_sample, importance_df

if __name__ == "__main__":
    model_dir = "models/independent"
    metadata, test_df = load_data_and_models(model_dir)
    
    # Get features used during training
    # Based on train_independent.py logic:
    targets = [
        'log_target_market_value_next_month',
        'target_minutes_next_month',
        'log_target_wiki_views_next_month',
        'target_value_change'
    ]
    meta = ['player_id', 'month_date', 'target_market_value_next_month', 'target_wiki_views_next_month']
    raw_logged = ['weekly_wage', 'current_market_value', 'wiki_views']
    exclude = targets + meta + raw_logged
    feature_cols = [c for c in test_df.columns if c in metadata['Market']['features']] if 'features' in metadata['Market'] else [c for c in test_df.columns if c not in exclude]

    X_test = test_df[feature_cols]
    
    output_dir = "models/independent/explanations"
    os.makedirs(output_dir, exist_ok=True)
    
    importance_matrix = {}
    
    for persona in ['Market', 'Sponsor', 'Coach', 'Agent']:
        model_path = f"{model_dir}/{persona}_model.joblib"
        shap_vals, X_sample, imp_df = run_treeshap(model_path, X_test, persona)
        
        # Save importance CSV
        imp_df.to_csv(f"{output_dir}/{persona}_importance.csv", index=False)
        
        # Save SHAP values and sample for visualization later
        # (Using joblib for numpy arrays/dfs)
        joblib.dump(shap_vals, f"{output_dir}/{persona}_shap_values.joblib")
        joblib.dump(X_sample, f"{output_dir}/{persona}_shap_sample.joblib")
        
        importance_matrix[persona] = imp_df.set_index('feature')['importance'].to_dict()
        print(f"✅ {persona} SHAP analysis complete.")

    # Save overall importance matrix
    matrix_df = pd.DataFrame(importance_matrix).fillna(0)
    matrix_df.to_csv(f"{output_dir}/stakeholder_importance_matrix.csv")
    
    print(f"\n✅ All TreeSHAP analyses complete. Results saved in {output_dir}")
