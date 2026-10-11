import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def main():
    # Load test data and predictions
    data_dir = "data/processed/ml_ready"
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    
    # We use the StackingEnsemble predictions for the best "Boardroom Synthesis" view
    # or the XGB_Market if it was the top performer. Given user's request for "The Model's Predicted",
    # and previous results showing XGB_Market (0.7838) is very strong, I will use that or the ensemble.
    # Let's use the ensemble predictions generated in the gen_xgb_preds turn if available, 
    # but since I ran the meta-learner evaluation separately, I'll use the Market XGB for clarity 
    # as it's the primary benchmark.
    
    y_true = test_df['log_target_market_value_next_month'].values
    y_pred = np.load('temp_preds/market_preds.npy') # Using Market XGB predictions
    
    # Sample for plotting to avoid overplotting (10,000 points is enough)
    indices = np.random.choice(len(test_df), min(10000, len(test_df)), replace=False)
    
    y_true_s = y_true[indices]
    y_pred_s = y_pred[indices]
    clusters_s = test_df['tactical_cluster'].values[indices]
    
    plt.figure(figsize=(10, 10))
    
    # Scatter plot color-coded by Tactical Archetype
    scatter = plt.scatter(y_true_s, y_pred_s, c=clusters_s, cmap='viridis', alpha=0.5, s=20)
    
    # Perfect 45-degree line
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # min of both axes
        np.max([plt.xlim(), plt.ylim()]),  # max of both axes
    ]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect Prediction (45°)')
    
    plt.xlabel('Ground Truth (Log Market Value)', fontsize=12)
    plt.ylabel('Predicted Valuation (Log Market Value)', fontsize=12)
    plt.title('Actual vs. Predicted Market Values\nColor-coded by Tactical Archetype', fontsize=14, fontweight='bold')
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Tactical Archetype (K-Means Cluster)', rotation=270, labelpad=15)
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add metrics as text box
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'$R^2$: {r2:.4f}', transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    os.makedirs('Desc/plots', exist_ok=True)
    plt.savefig('Desc/plots/actual_vs_predicted.png', dpi=300)
    print("✅ Plot saved to Desc/plots/actual_vs_predicted.png")

if __name__ == "__main__":
    main()
