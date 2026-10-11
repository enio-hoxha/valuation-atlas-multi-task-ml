import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import shap
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_all():
    """Load model, metadata and test data."""
    print("Loading model and data...")
    model = joblib.load("models/multitask_xgb_v1.joblib")
    with open("models/metadata_v1.json", "r") as f:
        meta = json.load(f)
    
    test_df = pd.read_parquet("data/processed/ml_ready/test.parquet")
    # Clean up test data for evaluation
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df.fillna(0, inplace=True)
    
    return model, meta, test_df

def comprehensive_evaluation(model, X_test, y_test, target_names):
    """Calculate detailed metrics for each task."""
    print("Calculating metrics...")
    preds = model.predict(X_test)
    
    eval_report = {}
    for i, name in enumerate(target_names):
        y_true = y_test.iloc[:, i]
        y_pred = preds[:, i]
        
        metrics = {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred))
        }
        eval_report[name] = metrics
        
    return eval_report, preds

def generate_shap_report(model, X_test, target_names):
    """Generate SHAP importance for the multi-task model."""
    print("Generating SHAP values (this may take a minute)...")
    
    # Take a sample for SHAP to speed up calculation if needed
    # Using 1000 samples for a robust but faster estimate
    X_sample = X_test.sample(min(1000, len(X_test)), random_state=42)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # SHAP returns a list of arrays for multi-output models
    # shap_values[0] = Market Value, [1] = Minutes, [2] = Wiki
    
    shap_importance = {}
    for i, name in enumerate(target_names):
        # Mean absolute SHAP value per feature
        vals = np.abs(shap_values[i]).mean(0)
        feature_importance = pd.DataFrame(list(zip(X_sample.columns, vals)), columns=['feature', 'importance'])
        feature_importance.sort_values(by=['importance'], ascending=False, inplace=True)
        shap_importance[name] = feature_importance.head(10).to_dict(orient='records')
        
    return shap_importance

if __name__ == "__main__":
    model, meta, test_df = load_all()
    
    X_test = test_df[meta['features']]
    y_test = test_df[meta['targets']]
    
    # 1. Metrics
    report, preds = comprehensive_evaluation(model, X_test, y_test, meta['targets'])
    
    # 2. SHAP
    shap_data = generate_shap_report(model, X_test, meta['targets'])
    
    # 3. Final Summary
    final_report = {
        "model_performance": report,
        "top_features_by_task": shap_data
    }
    
    with open("models/evaluation_report_v1.json", "w") as f:
        json.dump(final_report, f, indent=4)
        
    print("\n--- FINAL EVALUATION SUMMARY ---")
    for task, metrics in report.items():
        print(f"\nTask: {task}")
        print(f"  R2:   {metrics['R2']:.4f}")
        print(f"  MAE:  {metrics['MAE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        
    print("\n✅ Evaluation complete. Full report saved to models/evaluation_report_v1.json")
