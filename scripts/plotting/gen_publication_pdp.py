import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import partial_dependence
from scipy.stats import gaussian_kde

def generate_publication_pdp():
    print("Loading model and data for PDP generation (Publication mode)...")
    data_dir = "data/processed/ml_ready"
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df.fillna(0, inplace=True)
    
    # Use the Independent Market model for clean PDP extraction
    model_path = "models/independent/Market_model.joblib"
    if not os.path.exists(model_path):
        model = joblib.load("models/multitask_xgb_v1.joblib")
    else:
        model = joblib.load(model_path)
    
    features = model.get_booster().feature_names
    X = test_df[features]
    
    print("Calculating Partial Dependence for age_at_obs...")
    X_sample = X.sample(min(5000, len(X)), random_state=42)
    
    features_idx = [list(X.columns).index('age_at_obs')]
    pdp_results = partial_dependence(model, X_sample, features=features_idx, grid_resolution=100)
    
    ages = pdp_results['grid_values'][0]
    pdp_values = pdp_results['average'][0]
    
    pdp_impact = pdp_values - np.mean(pdp_values)
    
    age_density_data = test_df['age_at_obs'].values
    kde = gaussian_kde(age_density_data)
    density_curve = kde(ages)
    
    y_min, y_max = pdp_impact.min(), pdp_impact.max()
    y_range = y_max - y_min
    density_base = y_min - (y_range * 0.2)
    density_scaled = (density_curve / density_curve.max()) * (y_range * 0.15) + density_base

    # 3. Configure Formatting
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        plt.style.use('seaborn-paper')
    sns.set_theme(style="white", rc={
        "font.family": "serif",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })
    line_color = '#2c3e50'
    density_color = '#95a5a6'
    baseline_color = '#e74c3c'
    peak_color = '#f1c40f'
    text_color = 'black'
    box_facecolor = 'white'
    tick_size, label_size, title_size = 11, 12, 15

    # 4. Initialize Figure
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    ax.yaxis.grid(True)

    # 5. Plot the PDP Line
    ax.plot(ages, pdp_impact, color=line_color, linewidth=3, label='Marginal Market Impact', zorder=4)
    
    # 6. Plot the Density Distribution at the bottom
    ax.fill_between(ages, density_base, density_scaled, color=density_color, alpha=0.3, zorder=2)
    ax.plot(ages, density_scaled, color=density_color, linewidth=1, linestyle=':', alpha=0.5, zorder=2)
    
    # 7. Add the Neutral Baseline (Y=0)
    ax.axhline(0, color=baseline_color, linestyle='-', linewidth=1.5, alpha=0.6, zorder=3)
    
    # 8. Add Narrative Highlight: Peak Valuation Window (Ages 25-27)
    ax.axvspan(25, 27, color=peak_color, alpha=0.15, zorder=1)

    # 9. Typography and Labels
    ax.set_title('The Biological Depreciation Curve:\nPartial Dependence of Age on Market Value', 
                 fontsize=title_size, fontweight='bold', pad=20, loc='left', color=text_color)
    
    ax.set_xlabel('Biological Age (Years)', fontsize=label_size, labelpad=10, fontweight='bold', color=text_color)
    ax.set_ylabel('Marginal Impact on Log Market Valuation\n(Relative Change from Model Mean)', 
                  fontsize=label_size, labelpad=15, fontweight='bold', color=text_color)
    
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    
    # Set axis limits
    ax.set_xlim(16, 40)
    ax.set_ylim(density_base, y_max * 1.2)

    # 10. Narrative Text Annotations
    props = dict(boxstyle='round,pad=0.4', facecolor=box_facecolor, alpha=0.9, edgecolor='#d3d3d3')
    
    # Label the Neutral Baseline
    ax.text(39.5, 0.02, 'Value Neutrality Threshold', color=baseline_color, 
            fontsize=tick_size-2, fontweight='bold', ha='right', va='bottom')
    
    # Label the Peak Window
    ax.text(26, y_max * 0.9, 'Peak Valuation Window\n(Ages 25-27)', 
            color='#b8860b', 
            fontsize=tick_size-1, fontweight='bold', ha='center', bbox=props, zorder=5)
    
    # Label the Density Distribution
    ax.text(16.5, density_base + (y_range * 0.05), 'Dataset Age Distribution (Density)', 
            color='#7f8c8d', 
            fontsize=tick_size-3, fontstyle='italic', ha='left', zorder=5)

    ax.legend(loc='upper right', frameon=True, facecolor=box_facecolor, edgecolor='#d3d3d3', fontsize=tick_size-3)

    plt.tight_layout()
    
    save_path = 'Desc/plots/biological_depreciation_pdp_publication.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Publication-ready PDP saved to {save_path}")

if __name__ == "__main__":
    generate_publication_pdp()
