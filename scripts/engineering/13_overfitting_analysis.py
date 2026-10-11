import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

def load_split_data(data_dir="data/processed/ml_ready"):
    splits = ['train', 'val', 'test']
    dfs = {}
    for s in splits:
        df = pd.read_parquet(f"{data_dir}/{s}.parquet")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        dfs[s] = df
    return dfs

def main():
    dfs = load_split_data()
    model_path = "models/independent/Market_model.joblib"
    model = joblib.load(model_path)
    features = model.get_booster().feature_names
    target = 'log_target_market_value_next_month'
    
    results = []
    
    for split_name, df in dfs.items():
        y_true = df[target]
        y_pred = model.predict(df[features])
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        results.append({
            "Partition": split_name.capitalize(),
            "R2": r2,
            "MAE": mae,
            "Rows": len(df)
        })
    
    # Format the table
    print("\n" + "="*60)
    print("Table 2: The 'Gap Analysis' Table (Train vs. Val vs. Test)")
    print("="*60)
    print(f"{'Partition':<15} | {'R2':>10} | {'MAE':>10} | {'Samples':>10}")
    print("-" * 60)
    for res in results:
        print(f"{res['Partition']:<15} | {res['R2']:>10.4f} | {res['MAE']:>10.4f} | {res['Rows']:>10}")
    
    # Calculate Gaps
    train_r2 = results[0]['R2']
    test_r2 = results[2]['R2']
    gap = train_r2 - test_r2
    print("-" * 60)
    print(f"R2 Gap (Train - Test): {gap:.4f}")
    if gap < 0.05:
        print("Conclusion: Small gap detected. Model successfully avoided Overfitting.")
    else:
        print("Conclusion: Significant gap detected. Monitor for potential Overfitting.")
    print("="*60)

if __name__ == "__main__":
    main()
