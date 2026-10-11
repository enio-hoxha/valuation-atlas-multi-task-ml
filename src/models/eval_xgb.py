import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data(data_dir="data/processed/ml_ready"):
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
    indep_dir = "models/independent"
    personas = {
        'Market': 'log_target_market_value_next_month',
        'Sponsor': 'log_target_wiki_views_next_month',
        'Coach': 'target_minutes_next_month',
        'Agent': 'target_value_change'
    }
    
    for name, target in personas.items():
        try:
            model = joblib.load(f"{indep_dir}/{name}_model.joblib")
            current_xgb_features = model.get_booster().feature_names
            
            if target not in test_df.columns and target == 'target_value_change':
                 test_df = test_df.sort_values(['player_id', 'month_date'])
                 test_df['target_value_change'] = test_df.groupby('player_id')['current_market_value'].shift(-1) - test_df['current_market_value']
                 test_df['target_value_change'] = test_df['target_value_change'].fillna(0)
            
            y_true = test_df[target]
            y_pred = model.predict(test_df[current_xgb_features])
            m = evaluate_metrics(y_true, y_pred)
            print(f"XGB_{name} | R2: {m['R2']:.4f} | RMSE: {m['RMSE']:.4f} | MAE: {m['MAE']:.4f}")
        except Exception as e:
            print(f"Error evaluating XGB_{name}: {e}")

if __name__ == "__main__":
    main()
