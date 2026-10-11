import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import os

def generate_publication_split_funnel():
    """
    Generates a high-resolution Data Provenance Funnel that formally 
    partitions the terminal ML panel into Train, Validation, and Test sets.
    """
    # 1. Pipeline Data Definition
    stages = [
        'Initial Extraction\n(Transfermarkt & StatsBomb)',
        'Tactical Integration\n(Active Minutes > 0)',
        'Financial Intersection\n(Capology Wage Mapping)',
        'Commercial Intersection\n(Wikipedia Pageviews)',
        'Terminal ML Panel\n(Temporal OOT Split)'
    ]
    
    values = np.array([450000, 380000, 310000, 280000, 241256])
    
    train_n = 185477
    val_n = 19980
    test_n = 35799

    # 2. Configure Formatting
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
    base_colors = sns.color_palette("light:b_r", n_colors=len(stages))
    split_colors = ['#2c3e50', '#95a5a6', '#8b0000']
    text_main = '#2c3e50'
    text_sub = '#7f8c8d'
    title_size, label_size, tick_size = 15, 12, 11

    # 3. Initialize Figure
    fig, ax = plt.subplots(figsize=(11, 8), dpi=300)

    # 4. Create the Symmetrical Funnel Base
    y_pos = np.arange(len(stages))[::-1] 
    left_edges = -values / 2

    # 5. Plot the Filtering Stages (Top 4 Bars)
    for i in range(4):
        ax.barh(y_pos[i], values[i], left=left_edges[i], height=0.6, 
                color=base_colors[i], edgecolor='white', linewidth=2, alpha=0.8)
        
        # Absolute count (n) annotation
        ax.text(0, y_pos[i], f'n = {values[i]:,.0f}', 
                va='center', ha='center', 
                color='white' if i < 2 else '#2c3e50', 
                fontsize=tick_size, fontweight='bold')
        
        # Retention Rate annotation
        if i > 0:
            rate = (values[i] / values[i-1]) * 100
            ax.text(values[i]/2 + (values[0]*0.02), y_pos[i], 
                    f'↳ {rate:.1f}% Retained', 
                    va='center', ha='left', 
                    color=text_sub, fontsize=tick_size, fontstyle='italic')

    # 6. Plot the Stacked Terminal Stage (Train / Val / Test)
    terminal_y = y_pos[4]
    terminal_left = left_edges[4]
    
    # Train Block
    ax.barh(terminal_y, train_n, left=terminal_left, height=0.6, 
            color=split_colors[0], edgecolor='white', linewidth=1.5)
    ax.text(terminal_left + (train_n/2), terminal_y, f'{train_n:,.0f}', 
            va='center', ha='center', color='white', fontsize=tick_size-2, fontweight='bold')
    
    # Val Block
    val_left = terminal_left + train_n
    ax.barh(terminal_y, val_n, left=val_left, height=0.6, 
            color=split_colors[1], edgecolor='white', linewidth=1.5)
    ax.text(val_left + (val_n/2), terminal_y, f'{val_n:,.0f}', 
            va='center', ha='center', color='white', fontsize=tick_size-3, fontweight='bold')
    
    # Test Block
    test_left = val_left + val_n
    ax.barh(terminal_y, test_n, left=test_left, height=0.6, 
            color=split_colors[2], edgecolor='white', linewidth=1.5)
    ax.text(test_left + (test_n/2), terminal_y, f'{test_n:,.0f}', 
            va='center', ha='center', color='white', fontsize=tick_size-2, fontweight='bold')

    # Total survival retention text for the final split
    total_rate = (values[4] / values[3]) * 100
    ax.text(values[4]/2 + (values[0]*0.02), terminal_y, 
            f'↳ {total_rate:.1f}% Retained', 
            va='center', ha='left', color=text_sub, fontsize=tick_size, fontstyle='italic')

    # 7. Typography and Layout
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages, fontsize=tick_size, fontweight='bold', color=text_main)
    ax.set_xticks([])
    
    ax.set_title('ETL Pipeline Data Provenance & Temporal Splitting', 
                 fontsize=title_size, fontweight='bold', pad=40, loc='center', color=text_main)

    # 8. Add a custom legend for the Temporal Splitting
    legend_elements = [
        Patch(facecolor=split_colors[0], edgecolor='white', label='Training Block (2013-2022)'),
        Patch(facecolor=split_colors[1], edgecolor='white', label='Validation Block (2023)'),
        Patch(facecolor=split_colors[2], edgecolor='white', label='Out-Of-Time Test Block (2024)')
    ]
    leg = ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1),
              ncol=3, frameon=False, fontsize=tick_size-1)
    for text in leg.get_texts():
        text.set_color(text_main)

    plt.tight_layout()
    
    save_path = 'Desc/plots/coverage_funnel.png'
        
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Publication-ready split funnel saved to {save_path}")

if __name__ == "__main__":
    generate_publication_split_funnel()
