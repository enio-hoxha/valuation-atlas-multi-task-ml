import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import os

def generate_publication_bar_chart():
    """
    Generates a high-resolution, categorical bar chart for the Comparative Performance analysis.
    Values updated from evaluation on forward-filled, valuation-update-only test set.
    """
    # 1. Actual Metrics from local evaluation (Established Asset Filter)
    data = {
        'Model': [
            'XGB_Agent (Volatility)', 
            'XGB_Coach (Utility)', 
            'XGB_Sponsor (Hype)', 
            'XGB_Market (Baseline)', 
            'FT-Transformer (Deep Learning)', 
            'Ridge Meta-Learner (Ensemble)'
        ],
        'R2': [0.1120, 0.1540, 0.3210, 0.8510, 0.8720, 0.8940],
        'Role': ['Persona', 'Persona', 'Persona', 'Base Learner', 'Base Learner', 'Synthesis']
    }
    
    df = pd.DataFrame(data)
    df = df.sort_values('R2', ascending=True).reset_index(drop=True)

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
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })
    color_map = {
        'Persona': '#95a5a6',      
        'Base Learner': '#2c3e50', 
        'Synthesis': '#8b0000'     
    }
    text_color = '#333333'
    tick_size, label_size, title_size = 11, 12, 15
    box_facecolor = 'white'

    bar_colors = [color_map[role] for role in df['Role']]

    # 4. Initialize Figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.xaxis.grid(True)
    
    bars = ax.barh(df['Model'], df['R2'], color=bar_colors, height=0.6, alpha=0.9)

    # 5. Add Exact Data Labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, 
                bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', 
                va='center', 
                ha='left', 
                fontsize=tick_size, 
                fontweight='bold',
                color=text_color)

    # 6. Typography and Axis Formatting
    ax.set_title('Comparative Generalization Performance ($R^2$)', 
                 fontsize=title_size, fontweight='bold', pad=20, loc='left', color=text_color)
    ax.set_xlabel('Coefficient of Determination ($R^2$)', fontsize=label_size, labelpad=10, color=text_color)
    
    ax.set_xlim(0, 1.0)
    ax.tick_params(axis='y', labelsize=tick_size, left=False)
    ax.tick_params(axis='x', labelsize=tick_size)

    # 7. Add Custom Legend
    legend_elements = [
        Patch(facecolor=color_map['Persona'], label='Specialized Persona Models'),
        Patch(facecolor=color_map['Base Learner'], label='Primary Base Learners'),
        Patch(facecolor=color_map['Synthesis'], label='Boardroom Synthesis (Meta-Learner)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, 
              facecolor=box_facecolor, edgecolor='#d3d3d3', fontsize=tick_size-3, borderpad=1)

    plt.tight_layout()
    
    save_path = 'Desc/plots/comparative_performance_publication.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Publication-ready bar chart saved to {save_path}")

if __name__ == "__main__":
    generate_publication_bar_chart()
