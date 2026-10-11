import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def set_style():
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })

def draw_stacking_ensemble():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 1. Inputs
    ax.add_patch(patches.Rectangle((0.5, 3), 1.5, 2, facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=2))
    ax.text(1.25, 4, 'Engineered\nFeatures\n(50+ Dim)', ha='center', va='center', fontweight='bold')

    # 2. Base Models
    base_models = ['XGB_Market', 'XGB_Coach', 'XGB_Sponsor', 'XGB_Agent', 'FT-Transformer']
    colors = ['#34495e', '#34495e', '#34495e', '#34495e', '#2980b9']
    
    for i, (name, color) in enumerate(zip(base_models, colors)):
        y_pos = 6 - (i * 1.2)
        ax.add_patch(patches.FancyBboxPatch((3.5, y_pos), 2, 0.8, boxstyle="round,pad=0.1", 
                                          facecolor=color, edgecolor='#2c3e50', alpha=0.9))
        ax.text(4.5, y_pos+0.4, name, color='white', ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Arrows from Inputs
        ax.annotate('', xy=(3.5, y_pos+0.4), xytext=(2, 4), 
                    arrowprops=dict(arrowstyle="->", color='#7f8c8d', lw=1.5, connectionstyle="arc3,rad=0.1"))

    # 3. Meta-Learner
    ax.add_patch(patches.FancyBboxPatch((7, 3.2), 2, 1.5, boxstyle="round,pad=0.2", 
                                      facecolor='#8b0000', edgecolor='#2c3e50'))
    ax.text(8, 3.95, 'Meta-Learner\n(Ridge Regression)', color='white', ha='center', va='center', fontweight='bold')

    # Arrows to Meta-Learner
    for i in range(5):
        y_pos = 6 - (i * 1.2) + 0.4
        ax.annotate('', xy=(7, 3.95), xytext=(5.5, y_pos), 
                    arrowprops=dict(arrowstyle="->", color='#7f8c8d', lw=1.5))

    # 4. Final Output
    ax.text(9.5, 3.95, 'Final\nValuation', ha='center', va='center', fontweight='bold', color='#2c3e50')
    ax.annotate('', xy=(9.3, 3.95), xytext=(9, 3.95), arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=2))

    ax.set_title('Architecture: Stacking Ensemble (Meta-Learning)', pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Desc/plots/arch_stacking_ensemble.png', bbox_inches='tight', dpi=300)
    print("✅ Stacking diagram saved.")

def draw_ft_transformer():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Inputs
    ax.text(1, 4, 'Numerical &\nCategorical\nFeatures', ha='center', va='center', fontweight='bold')
    
    # Feature Tokenizer
    ax.add_patch(patches.Rectangle((2.5, 2.5), 1.5, 3, facecolor='#2980b9', edgecolor='#2c3e50', alpha=0.8))
    ax.text(3.25, 4, 'Feature\nTokenizer\n(Linear Embed)', ha='center', va='center', color='white', fontweight='bold')
    
    # Transformer Blocks
    for i in range(3):
        x_off = 4.5 + (i * 1.2)
        ax.add_patch(patches.Rectangle((x_off, 2.8), 1, 2.4, facecolor='#3498db', edgecolor='#2c3e50', alpha=0.7))
        ax.text(x_off+0.5, 4, f'L{i+1}', ha='center', va='center', color='white', fontweight='bold')
        if i == 1:
            ax.text(x_off+0.5, 5.5, 'Multi-Head Self-Attention', ha='center', fontsize=9, fontstyle='italic')

    # Regression Head
    ax.add_patch(patches.Rectangle((8.2, 3), 1.2, 2, facecolor='#1f3a93', edgecolor='#2c3e50'))
    ax.text(8.8, 4, 'MLP\nHead', ha='center', va='center', color='white', fontweight='bold')

    # Arrows
    ax.annotate('', xy=(2.5, 4), xytext=(1.8, 4), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate('', xy=(4.5, 4), xytext=(4, 4), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate('', xy=(8.2, 4), xytext=(8.1, 4), arrowprops=dict(arrowstyle="->", lw=2))

    ax.set_title('Architecture: FT-Transformer (Feature Tokenizer)', pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Desc/plots/arch_ft_transformer.png', bbox_inches='tight', dpi=300)
    print("✅ FT-Transformer diagram saved.")

def draw_xgboost():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Initial Prediction
    ax.add_patch(patches.Circle((1, 4), 0.5, facecolor='#bdc3c7', edgecolor='#2c3e50'))
    ax.text(1, 4, 'base\nscore', ha='center', va='center', fontsize=9)

    # Sequence of Trees
    for i in range(3):
        x_start = 2.5 + (i * 2.2)
        # Simplified Tree icon
        ax.add_patch(patches.Polygon([[x_start, 3.5], [x_start+0.6, 5], [x_start+1.2, 3.5]], 
                                   facecolor='#16a085', edgecolor='#2c3e50', alpha=0.8))
        ax.text(x_start+0.6, 4.2, f'Tree {i+1}', ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        ax.text(x_start+0.6, 3.2, f'Learning Rate: η', ha='center', fontsize=8)

        # Plus signs
        if i < 2:
            ax.text(x_start+1.7, 4, '+', fontsize=20, fontweight='bold')
    
    ax.text(8.5, 4, '...', fontsize=20, fontweight='bold')
    ax.text(9.5, 4, 'Σ', fontsize=20, fontweight='bold')

    ax.annotate('', xy=(2.5, 4), xytext=(1.5, 4), arrowprops=dict(arrowstyle="->", lw=2))

    ax.set_title('Architecture: Gradient Boosted Trees (XGBoost)', pad=20, fontweight='bold')
    ax.text(5, 1, 'Objective: Minimize Residuals via Sequential Tree Induction', ha='center', fontstyle='italic')

    plt.tight_layout()
    plt.savefig('Desc/plots/arch_xgboost.png', bbox_inches='tight', dpi=300)
    print("✅ XGBoost diagram saved.")

if __name__ == "__main__":
    os.makedirs('Desc/plots', exist_ok=True)
    set_style()
    draw_stacking_ensemble()
    draw_ft_transformer()
    draw_xgboost()
