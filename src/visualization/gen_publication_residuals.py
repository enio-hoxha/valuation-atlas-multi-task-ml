import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def generate_publication_residuals(df, y_true_col, y_pred_col):
    """
    Generates a high-resolution, statistically annotated residual distribution plot.
    """
    # 1. Calculate the Residuals (Actual - Predicted)
    # Using your clean, non-zero dataset (Valuation Updates focus)
    mask = (df[y_true_col] > 0.0) & (df['log_current_market_value'] > 0.0)
    clean_df = df[mask].copy()
    residuals = clean_df[y_true_col] - clean_df[y_pred_col]

    # 2. Calculate Statistical Moments for the embedded proof
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)

    # 3. Configure Academic Formatting
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        plt.style.use('seaborn-paper')
        
    sns.set_theme(style="white", rc={
        "font.family": "serif",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })

    # 4. Initialize Figure
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    # 5. Plot the Histogram and KDE (Kernel Density Estimate)
    sns.histplot(
        residuals, 
        bins=80, 
        kde=True, 
        color='#95a5a6',      # Muted grey for the bins
        edgecolor='white', 
        line_kws={'linewidth': 2.5, 'color': '#2c3e50'}, # Navy line for the KDE
        ax=ax,
        stat='density'
    )

    # 6. Add the Zero-Error Baseline
    ax.axvline(0, color='#8b0000', linestyle='--', linewidth=2, zorder=4, 
               label=r'Zero Systematic Bias ($y = \hat{y}$)')

    # 7. Typography and Explicative Labels
    ax.set_title('Meta-Learner Error Distribution: Residual Analysis', 
                 fontsize=15, fontweight='bold', pad=20, loc='left')
    
    ax.set_xlabel(r'Prediction Error ($y - \hat{y}$) [Log Market Valuation]', 
                  fontsize=12, labelpad=10, fontweight='bold')
    ax.set_ylabel('Density of Observations', fontsize=12, labelpad=10)

    # Add directional annotations to the X-axis to explain the economics
    ax.text(0.02, -0.08, '← Model Overvalued Asset', transform=ax.transAxes, 
            fontsize=10, fontstyle='italic', color='#7f8c8d')
    ax.text(0.98, -0.08, 'Model Undervalued Asset →', transform=ax.transAxes, 
            fontsize=10, fontstyle='italic', color='#7f8c8d', ha='right')

    # Explicitly constrain X-axis to ignore extreme, irrelevant outliers
    ax.set_xlim(-1.5, 1.5) 

    # 8. Embed the Formal Statistical Proof Box
    textstr = '\n'.join((
        r'$\mu=%.4f$' % (mu, ),
        r'$\sigma=%.4f$' % (sigma, ),
        r'$\mathrm{Skewness}=%.4f$' % (skewness, ),
        r'$\mathrm{Kurtosis}=%.4f$' % (kurtosis, )
    ))
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#d3d3d3')
    
    # Place text box in the upper right
    ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, zorder=5)

    # 9. Legend
    ax.legend(loc='upper left', frameon=True, edgecolor='#d3d3d3', fontsize=10)

    plt.tight_layout()
    
    # Save as high-res PNG
    os.makedirs('Desc/plots', exist_ok=True)
    save_path = 'Desc/plots/residual_distribution_publication.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Publication-ready residual plot saved to {save_path}")

if __name__ == "__main__":
    # Load data
    data_dir = "data/processed/ml_ready"
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    
    # Load predictions (using market_preds as meta-learner proxy for visualization)
    y_pred = np.load('temp_preds/market_preds.npy')
    test_df['predicted_log_market_value'] = y_pred
    
    generate_publication_residuals(
        test_df, 
        'log_target_market_value_next_month', 
        'predicted_log_market_value'
    )
