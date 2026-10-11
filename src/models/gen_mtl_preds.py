import pandas as pd
import numpy as np
import joblib
import os
import json

def load_data(data_dir="data/processed/ml_ready"):
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df.fillna(0, inplace=True)
    return test_df

def main():
    test_df = load_data()
    os.makedirs("temp_preds", exist_ok=True)
    
    # Load Multi-Task Model and Metadata
    model_path = "models/multitask_xgb_v1.joblib"
    metadata_path = "models/metadata_v1.json"
    scaler_path = "models/scaler.joblib"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    print(f"Loading Multi-Task model from {model_path}...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    feature_cols = metadata['features']
    target_cols = metadata['targets']
    
    print(f"Generating predictions for {len(test_df)} rows...")
    X_test = scaler.transform(test_df[feature_cols])
    preds = model.predict(X_test)
    
    # Save each target's predictions
    # Index 0 is usually log_target_market_value_next_month
    for i, target in enumerate(target_cols):
        # Map target name to a simple filename
        short_name = target.split('_')[2] if 'market' in target else target.split('_')[1]
        if 'market' in target: short_name = 'market'
        elif 'minutes' in target: short_name = 'coach' # mapping to existing naming convention
        elif 'wiki' in target: short_name = 'sponsor'
        
        np.save(f"temp_preds/{short_name}_preds.npy", preds[:, i])
        print(f"✅ Saved {target} predictions to temp_preds/{short_name}_preds.npy")

if __name__ == "__main__":
    main()
