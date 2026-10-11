import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_publication_quadrant(df, x_col, y_col):
    """
    Generates a high-resolution, density-aware quadrant plot for academic reporting.
    """
    # 1. Apply your correction: Filter out the zero-imputation artifacts
    clean_df = df[(df[x_col] > 0.0) & (df[y_col] > 0.0)].copy()
    
    # 2. Calculate Statistical Thresholds (Medians) for the Crosshairs
    x_median = clean_df[x_col].median()
    y_median = clean_df[y_col].median()

    # 3. Configure Academic Formatting
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        plt.style.use('seaborn-paper')
        
    sns.set_theme(style="white", rc={
        "font.family": "serif",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.grid": False # Turn off standard grid to emphasize the crosshairs
    })
    
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # 4. Create the Hexbin Density Plot
    # Using 'Blues' to visually separate this from the predictive performance plots
    hb = ax.hexbin(
        x=clean_df[x_col], 
        y=clean_df[y_col], 
        gridsize=50, 
        cmap='Blues', 
        mincnt=1, 
        edgecolors='none',
        alpha=0.85
    )
    
    # 5. Add the Statistical Crosshairs (The Quadrant Boundaries)
    ax.axvline(x_median, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.8, zorder=3)
    ax.axhline(y_median, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.8, zorder=3)
    
    # 6. Typography and Axis Formatting
    ax.set_title('Market Landscape: Commercial Reach vs. Structural Value', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel('Ground Truth Valuation (Log Transformed)', fontsize=12, labelpad=10)
    ax.set_ylabel('Commercial Reach / Hype (Log Wiki Views)', fontsize=12, labelpad=10)
    
    # Add Density Colorbar
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Density of Players', rotation=270, labelpad=20, fontsize=11)
    
    # 7. Add Formal Quadrant Annotations
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#d3d3d3')
    
    # Get axis limits for absolute text positioning
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Q2: Top-Left (High Hype, Low Value)
    ax.text(x_min + (x_median - x_min)*0.05, y_max - (y_max - y_median)*0.1, 
            'Speculative Breakouts\n(High Hype, Undervalued)', 
            fontsize=10, fontweight='bold', color='#2c3e50', bbox=props, zorder=4)
    
    # Q1: Top-Right (High Hype, High Value)
    ax.text(x_max - (x_max - x_median)*0.05, y_max - (y_max - y_median)*0.1, 
            'Established Elite\n(High Value & Reach)', 
            fontsize=10, fontweight='bold', color='#2c3e50', ha='right', bbox=props, zorder=4)
    
    # Q3: Bottom-Left (Low Hype, Low Value)
    ax.text(x_min + (x_median - x_min)*0.05, y_min + (y_median - y_min)*0.05, 
            'Baseline Assets\n(Developing Profiles)', 
            fontsize=10, fontweight='bold', color='#7f8c8d', bbox=props, zorder=4)
    
    # Q4: Bottom-Right (Low Hype, High Value)
    ax.text(x_max - (x_max - x_median)*0.05, y_min + (y_median - y_min)*0.05, 
            'Tactical Anchors\n(High Value, Low Commercial)', 
            fontsize=10, fontweight='bold', color='#2c3e50', ha='right', bbox=props, zorder=4)

    plt.tight_layout()
    
    # Save for the report
    os.makedirs('Desc/plots', exist_ok=True)
    save_path = 'Desc/plots/hype_vs_value_quadrant_publication.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Publication-ready quadrant plot saved to {save_path}")

if __name__ == "__main__":
    # Load data
    data_dir = "data/processed/ml_ready"
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    
    # log_current_market_value is the X axis (Structural Value)
    # log_wiki_views is the Y axis (Commercial Reach / Hype)
    generate_publication_quadrant(test_df, 'log_current_market_value', 'log_wiki_views')
