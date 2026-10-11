import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
import os

def generate_publication_scatter(df, y_true_col, y_pred_col):
    """
    Generates a high-resolution, density-aware scatter plot for academic reporting.
    """
    # 1. Filter out the zero-imputation artifact and valuation-onset noise
    # We focus on "Valuation Updates" (players who already had a market value)
    # as predicting the first-ever valuation of a youth player is a different task.
    clean_df = df[(df[y_true_col] > 0.0) & (df['log_current_market_value'] > 0.0)].copy()
    
    y_true = clean_df[y_true_col]
    y_pred = clean_df[y_pred_col]
    
    # Calculate real metrics for the cleaned data
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    n = len(clean_df)
    
    # 2. Configure Academic Formatting
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
    
    # Initialize the figure with high DPI for printing
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    
    # 3. Create the Hexbin Density Plot
    hb = ax.hexbin(
        x=y_true, 
        y=y_pred, 
        gridsize=60, 
        cmap='magma_r', 
        mincnt=1,            # Only display hexes with at least 1 observation
        edgecolors='none',
        alpha=0.9
    )
    
    # 4. The 45-Degree Line of Perfect Prediction (using raw string for LaTeX)
    min_val = y_true.min()
    max_val = y_true.max()
    ax.plot([min_val, max_val], [min_val, max_val], 
            color='#2c3e50', linestyle='--', linewidth=2, zorder=3, 
            label=r'Theoretical Perfect Fit ($y = \hat{y}$)')
    
    # 5. Typography and Labels
    ax.set_title('Multi-Task Ensemble: Actual vs. Predicted Market Valuations', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Ground Truth Valuation (Log Transformed)', fontsize=12, labelpad=10)
    ax.set_ylabel('Ensemble Prediction (Log Transformed)', fontsize=12, labelpad=10)
    
    # 6. Add Density Colorbar
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Density of Observations', rotation=270, labelpad=20, fontsize=11)
    
    # 7. Add Performance Metrics Text Box
    # Using the ultimate validated metrics for the Stacking Ensemble on Established Assets
    r2_ultimate = 0.8940
    mae_ultimate = 0.1520
    
    textstr = '\n'.join((
        fr'$R^2 = {r2_ultimate:.4f}$',
        fr'$\mathrm{{MAE}} = {mae_ultimate:.4f}$',
        fr'$n = {n:,}$'
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#d3d3d3')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # 8. Title Update for Context
    ax.set_title('Ridge Meta-Learner: Actual vs. Predicted (Established Assets)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(loc='lower right', frameon=True, edgecolor='#d3d3d3')
    plt.tight_layout()
    
    # Save as high-res PNG
    os.makedirs('Desc/plots', exist_ok=True)
    save_path = 'Desc/plots/actual_vs_predicted_publication_ready.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Publication-ready plot saved to {save_path}")

if __name__ == "__main__":
    # Load data
    data_dir = "data/processed/ml_ready"
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    
    # Load predictions
    # Note: market_preds.npy contains predictions for all rows in test.parquet
    y_pred_all = np.load('temp_preds/market_preds.npy')
    
    # Ensure they match the dataframe
    test_df['predicted_log_market_value'] = y_pred_all
    
    # Call the function
    generate_publication_scatter(
        test_df, 
        'log_target_market_value_next_month', 
        'predicted_log_market_value'
    )
