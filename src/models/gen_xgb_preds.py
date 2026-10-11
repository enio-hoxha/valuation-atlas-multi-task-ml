import pandas as pd
import numpy as np
import joblib
import os

def load_data(data_dir="data/processed/ml_ready"):
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df.fillna(0, inplace=True)
    return test_df

def main():
    test_df = load_data()
    indep_dir = "models/independent"
    os.makedirs("temp_preds", exist_ok=True)
    
    for persona in ['Market', 'Sponsor', 'Coach', 'Agent']:
        print(f"Generating preds for {persona}...")
        model = joblib.load(f"{indep_dir}/{persona}_model.joblib")
        current_xgb_features = model.get_booster().feature_names
        preds = model.predict(test_df[current_xgb_features])
        np.save(f"temp_preds/{persona.lower()}_preds.npy", preds)

if __name__ == "__main__":
    main()
