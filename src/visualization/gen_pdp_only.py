import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence

def main():
    data_dir = "data/processed/ml_ready"
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df.fillna(0, inplace=True)
    
    # Load ONLY Market model for PDP
    model = joblib.load("models/independent/Market_model.joblib")
    features = model.get_booster().feature_names
    
    X = test_df[features]
    print("Generating PDP for age_at_obs...")
    features_idx = [list(X.columns).index('age_at_obs')]
    pdp_results = partial_dependence(model, X, features=features_idx, grid_resolution=50)
    
    plt.figure(figsize=(10, 6))
    plt.plot(pdp_results['grid_values'][0], pdp_results['average'][0], color='#2a9d8f', linewidth=3)
    plt.title('The Valuation Curve: Age vs. Market Value', fontsize=14, fontweight='bold')
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Partial Dependence (Log MV)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    peak_val = pdp_results['grid_values'][0][np.argmax(pdp_results['average'][0])]
    plt.axvline(x=peak_val, color='#e63946', linestyle='--', label=f'Theoretical Peak: ~{peak_val:.1f}')
    plt.legend()
    
    os.makedirs('Desc/plots', exist_ok=True)
    plt.savefig('Desc/plots/pdp_age.png', dpi=300)
    print("✅ Age PDP saved.")

if __name__ == "__main__":
    main()
