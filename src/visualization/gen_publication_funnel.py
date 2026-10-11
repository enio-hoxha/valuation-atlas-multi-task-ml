import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_publication_funnel():
    """
    Generates a high-resolution, statistically annotated Data Provenance Funnel.
    Uses actual counts from the master ML panel.
    """
    # 1. Pipeline Data (Actual row counts from current dataset)
    stages = [
        'Initial Roster Extraction\n(Transfermarkt & StatsBomb)',
        'Tactical Integration\n(Active Minutes > 0)',
        'Financial Intersection\n(Capology Wage Mapping)',
        'Commercial Reach Intersection\n(Wikipedia Pageviews)',
        'Terminal ML Panel\n(Dense Matrix for Induction)'
    ]
    
    # Values obtained from actual data audit:
    # Total: 246,053
    # Minutes > 0: 246,053
    # Wage > 0: 146,706
    # Wiki > 0: 63,748
    # Dense (All 3): 47,120
    values = np.array([246053, 246053, 146706, 63748, 47120])
    
    # 2. Calculate Attrition/Retention Rates
    retention_rates = [100.0] # First stage is 100% of baseline
    for i in range(1, len(values)):
        rate = (values[i] / values[i-1]) * 100 if values[i-1] > 0 else 0
        retention_rates.append(rate)

    # 3. Configure Academic Formatting
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        plt.style.use('seaborn-paper')
        
    sns.set_theme(style="white", rc={
        "font.family": "serif",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.grid": False
    })

    # 4. Initialize Figure
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

    # 5. Create the Symmetrical Funnel
    y_pos = np.arange(len(stages))[::-1] 
    left_edges = -values / 2

    # Academic color palette: Transitioning from a broad grey to a dense navy
    colors = sns.color_palette("light:b_r", n_colors=len(stages))

    bars = ax.barh(y_pos, values, left=left_edges, height=0.6, color=colors, edgecolor='white', linewidth=2)

    # 6. Add Statistical Annotations
    for i, (bar, val, rate) in enumerate(zip(bars, values, retention_rates)):
        # Absolute count (n) inside the bar
        ax.text(0, bar.get_y() + bar.get_height()/2, 
                f'n = {val:,.0f}', 
                va='center', ha='center', 
                color='white' if i < 3 else '#2c3e50',
                fontsize=11, fontweight='bold')
        
        # Retention Rate annotation on the right side
        if i > 0:
            right_edge = val / 2
            ax.text(right_edge + (values[0]*0.02), bar.get_y() + bar.get_height()/2, 
                    f'↳ {rate:.1f}% Retained', 
                    va='center', ha='left', 
                    color='#7f8c8d', fontsize=11, fontstyle='italic')

    # 7. Typography and Layout
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages, fontsize=11, fontweight='bold', color='#2c3e50')
    ax.set_xticks([])
    
    ax.set_title('ETL Pipeline Data Provenance & Resolution Attrition', 
                 fontsize=15, fontweight='bold', pad=30, loc='center')

    total_survival = (values[-1] / values[0]) * 100
    ax.text(0.5, 0.98, f'Total Pipeline Survival Rate: {total_survival:.1f}% of Initial Roster (Observation Level)', 
            transform=ax.transAxes, ha='center', va='top', fontsize=11, color='#7f8c8d')

    plt.tight_layout()
    
    os.makedirs('Desc/plots', exist_ok=True)
    save_path = 'Desc/plots/coverage_funnel_publication.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Publication-ready funnel plot saved to {save_path}")

if __name__ == "__main__":
    generate_publication_funnel()
