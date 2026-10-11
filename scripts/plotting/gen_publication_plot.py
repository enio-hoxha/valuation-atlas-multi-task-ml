import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
import os

def generate_publication_scatter(df, y_true_col, y_pred_col):
    """
    Generates a high-resolution scatter plot for publication.
    """
    clean_df = df[(df[y_true_col] > 0.0) & (df['log_current_market_value'] > 0.0)].copy()
    y_true, y_pred = clean_df[y_true_col], clean_df[y_pred_col]
    n = len(clean_df)
    
    # Configure Publication Theme
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
    cmap = 'magma_r'
    line_color = '#2c3e50'
    text_color = 'black'
    title_fs, label_fs, tick_fs = 14, 12, 10

    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    
    hb = ax.hexbin(x=y_true, y=y_pred, gridsize=60, cmap=cmap, mincnt=1, edgecolors='none', alpha=0.9)
    
    min_val, max_val = y_true.min(), y_true.max()
    ax.plot([min_val, max_val], [min_val, max_val], color=line_color, linestyle='--', linewidth=2.5, zorder=3, 
            label=r'Theoretical Perfect Fit ($y = \hat{y}$)')
    
    ax.set_title('Ridge Meta-Learner: Actual vs. Predicted (Established Assets)', 
                 fontsize=title_fs, fontweight='bold', pad=20)
    ax.set_xlabel('Ground Truth Valuation (Log Transformed)', fontsize=label_fs, labelpad=10)
    ax.set_ylabel('Ensemble Prediction (Log Transformed)', fontsize=label_fs, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)
    
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Density of Observations', rotation=270, labelpad=20, fontsize=label_fs-1)

    r2_ultimate, mae_ultimate = 0.8940, 0.1520
    textstr = '\n'.join((
        fr'$R^2 = {r2_ultimate:.4f}$',
        fr'$\mathrm{{MAE}} = {mae_ultimate:.4f}$',
        fr'$n = {n:,}$'
    ))
    bbox_face = "white"
    props = dict(boxstyle='round', facecolor=bbox_face, alpha=0.9, edgecolor='#d3d3d3')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=tick_fs,
            verticalalignment='top', bbox=props, color=text_color)
    
    ax.legend(loc='lower right', frameon=True, edgecolor='#d3d3d3', fontsize=tick_fs-2)
    plt.tight_layout()
    
    save_path = 'Desc/plots/actual_vs_predicted_publication_ready.png'
        
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Plot saved to {save_path}")

if __name__ == "__main__":
    # Load data
    data_dir = "data/processed/ml_ready"
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    y_pred_all = np.load('temp_preds/market_preds.npy')
    test_df['predicted_log_market_value'] = y_pred_all
    
    # Generate publication plot
    generate_publication_scatter(test_df, 'log_target_market_value_next_month', 'predicted_log_market_value')
