import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
from scipy.stats import norm

def main():
    # Load test data and predictions
    data_dir = "data/processed/ml_ready"
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    
    y_true = test_df['log_target_market_value_next_month'].values
    y_pred = np.load('temp_preds/market_preds.npy') # Using Market XGB predictions
    
    # Calculate Residuals
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 7))
    
    # Plot Histogram with KDE
    sns.histplot(residuals, bins=50, kde=True, color='#457b9d', stat="density", alpha=0.6)
    
    # Overlay a Normal Distribution curve for comparison
    mu, std = norm.fit(residuals)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r', linewidth=2, label=f'Normal Fit ($\mu$={mu:.4f}, $\sigma$={std:.4f})')
    
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.8, label='Zero Error Line')
    
    plt.xlabel('Residual (Actual - Predicted)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Residual Analysis Distribution\nValidation of Unbiased Errors', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add text box with skewness and kurtosis
    from scipy.stats import skew, kurtosis
    s = skew(residuals)
    k = kurtosis(residuals)
    plt.text(0.05, 0.95, f'Skewness: {s:.4f}\nKurtosis: {k:.4f}', transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    os.makedirs('Desc/plots', exist_ok=True)
    plt.savefig('Desc/plots/residuals_distribution.png', dpi=300)
    print("✅ Plot saved to Desc/plots/residuals_distribution.png")

if __name__ == "__main__":
    main()
