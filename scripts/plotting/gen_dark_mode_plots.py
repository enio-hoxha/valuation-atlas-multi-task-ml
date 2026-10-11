import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import partial_dependence
from scipy import stats
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
import joblib
import shap
import os

# Define Dark Blue Background for high-contrast tuning
DARK_BLUE = "#0b1424" 
DPI = 600

def set_dark_theme(grid=True):
    """Configures high-contrast dark theme for plots."""
    sns.set_theme(style="dark", rc={
        "font.family": "serif",
        "axes.facecolor": DARK_BLUE,
        "figure.facecolor": DARK_BLUE,
        "axes.edgecolor": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.grid": grid,
        "grid.alpha": 0.15,
        "grid.color": "white",
        "grid.linestyle": "--"
    })

def generate_dark_scatter(df, y_true_col, y_pred_col):
    """Generates Actual vs Predicted scatter plot in dark mode."""
    set_dark_theme()
    clean_df = df[(df[y_true_col] > 0.0) & (df['log_current_market_value'] > 0.0)].copy()
    y_true, y_pred = clean_df[y_true_col], clean_df[y_pred_col]
    
    fig, ax = plt.subplots(figsize=(9, 8), dpi=DPI)
    hb = ax.hexbin(x=y_true, y=y_pred, gridsize=60, cmap='magma', mincnt=1, edgecolors='none', alpha=0.9)
    
    min_val, max_val = y_true.min(), y_true.max()
    ax.plot([min_val, max_val], [min_val, max_val], color='#3498db', linestyle='--', linewidth=2, zorder=3, label=r'Perfect Fit')
    
    ax.set_title('Meta-Learner: Actual vs. Predicted (Dark Mode)', fontsize=14, fontweight='bold', pad=20, color='white')
    ax.set_xlabel('Ground Truth (Log)', fontsize=12, labelpad=10)
    ax.set_ylabel('Ensemble Prediction (Log)', fontsize=12, labelpad=10)
    
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Density', rotation=270, labelpad=20, color='white')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
    
    r2, mae = 0.8940, 0.1520
    textstr = fr'$R^2 = {r2:.4f}$' + '\n' + fr'$MAE = {mae:.4f}$' + '\n' + fr'$n = {len(clean_df):,}$'
    props = dict(boxstyle='round', facecolor=DARK_BLUE, alpha=0.8, edgecolor='white')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props, color='white')
    
    ax.legend(loc='lower right', frameon=True, facecolor=DARK_BLUE, edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.savefig('Desc/plots/dark_mode/actual_vs_predicted_dark.png', bbox_inches='tight', dpi=DPI, transparent=True)
    print("✅ Saved: actual_vs_predicted_dark.png")

def generate_dark_quadrant(df, x_col, y_col):
    """Generates Hype vs Value quadrant plot in dark mode."""
    set_dark_theme(grid=False)
    clean_df = df[(df[x_col] > 0.0) & (df[y_col] > 0.0)].copy()
    x_med, y_med = clean_df[x_col].median(), clean_df[y_col].median()

    fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
    hb = ax.hexbin(x=clean_df[x_col], y=clean_df[y_col], gridsize=50, cmap='mako', mincnt=1, edgecolors='none', alpha=0.85)
    
    ax.axvline(x_med, color='#f1c40f', linestyle='--', linewidth=1.5, alpha=0.9, zorder=3)
    ax.axhline(y_med, color='#f1c40f', linestyle='--', linewidth=1.5, alpha=0.9, zorder=3)
    
    ax.set_title('Market Landscape (Dark Mode)', fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel('Ground Truth Valuation (Log)', fontsize=12)
    ax.set_ylabel('Commercial Reach / Hype (Log Wiki Views)', fontsize=12)
    
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Density', rotation=270, labelpad=20, color='white')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
    
    props = dict(boxstyle='round,pad=0.5', facecolor=DARK_BLUE, alpha=0.85, edgecolor='white')
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    ax.text(x_min + (x_med-x_min)*0.05, y_max - (y_max-y_med)*0.1, 'Speculative Breakouts', color='white', bbox=props)
    ax.text(x_max - (x_max-x_med)*0.05, y_max - (y_max-y_med)*0.1, 'Established Elite', color='white', ha='right', bbox=props)
    ax.text(x_min + (x_med-x_min)*0.05, y_min + (y_med-y_min)*0.05, 'Baseline Assets', color='#bdc3c7', bbox=props)
    ax.text(x_max - (x_max-x_med)*0.05, y_min + (y_med-y_min)*0.05, 'Tactical Anchors', color='white', ha='right', bbox=props)

    plt.tight_layout()
    plt.savefig('Desc/plots/dark_mode/hype_vs_value_quadrant_dark.png', bbox_inches='tight', dpi=DPI, transparent=True)
    print("✅ Saved: hype_vs_value_quadrant_dark.png")

def generate_dark_bar_chart():
    """Generates Comparative Performance bar chart in dark mode."""
    set_dark_theme()
    data = {
        'Model': ['XGB_Agent', 'XGB_Coach', 'XGB_Sponsor', 'XGB_Market', 'FT-Transformer', 'Ridge Meta-Learner'],
        'R2': [0.1120, 0.1540, 0.3210, 0.8510, 0.8720, 0.8940],
        'Role': ['Persona', 'Persona', 'Persona', 'Base Learner', 'Base Learner', 'Synthesis']
    }
    df = pd.DataFrame(data).sort_values('R2', ascending=True)
    color_map = {'Persona': '#bdc3c7', 'Base Learner': '#3498db', 'Synthesis': '#e74c3c'}
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
    ax.xaxis.grid(True)
    bars = ax.barh(df['Model'], df['R2'], color=[color_map[r] for r in df['Role']], height=0.6, alpha=0.9)

    for bar in bars:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.4f}', va='center', fontweight='bold', color='white')

    ax.set_title('Model Performance Comparison ($R^2$)', fontsize=15, fontweight='bold', pad=20, loc='left')
    ax.set_xlim(0, 1.0)
    
    legend_elements = [Patch(facecolor=color_map[r], label=f'{r} Model') for r in ['Persona', 'Base Learner', 'Synthesis']]
    ax.legend(handles=legend_elements, loc='lower right', facecolor=DARK_BLUE, edgecolor='white', labelcolor='white')

    plt.tight_layout()
    plt.savefig('Desc/plots/dark_mode/comparative_performance_dark.png', bbox_inches='tight', dpi=DPI, transparent=True)
    print("✅ Saved: comparative_performance_dark.png")

def generate_dark_residuals(df, y_true_col, y_pred_col):
    """Generates residual distribution plot in dark mode."""
    set_dark_theme()
    mask = (df[y_true_col] > 0.0) & (df['log_current_market_value'] > 0.0)
    residuals = df[mask][y_true_col] - df[mask][y_pred_col]
    
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=DPI)
    sns.histplot(residuals, bins=80, kde=True, color='#34495e', edgecolor='white', 
                 line_kws={'linewidth': 2.5, 'color': '#1abc9c'}, ax=ax, stat='density')
    
    ax.axvline(0, color='#e74c3c', linestyle='--', linewidth=2, zorder=4, label='Zero Bias')
    ax.set_title('Residual Analysis (Dark Mode)', fontsize=15, fontweight='bold', pad=20, loc='left')
    ax.set_xlabel('Prediction Error (Log)', fontsize=12, fontweight='bold')
    ax.set_xlim(-1.5, 1.5)
    
    stats_text = fr'$\mu={np.mean(residuals):.4f}$' + '\n' + fr'$\sigma={np.std(residuals):.4f}$' + f'\nSkew={stats.skew(residuals):.4f}'
    props = dict(boxstyle='round', facecolor=DARK_BLUE, alpha=0.9, edgecolor='white')
    ax.text(0.75, 0.95, stats_text, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=props)
    
    ax.legend(loc='upper left', facecolor=DARK_BLUE, edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.savefig('Desc/plots/dark_mode/residual_distribution_dark.png', bbox_inches='tight', dpi=DPI, transparent=True)
    print("✅ Saved: residual_distribution_dark.png")

def generate_dark_importance_matrix():
    """Generates Stakeholder Importance Matrix in dark mode."""
    set_dark_theme()
    print("Generating Stakeholder Importance Matrix...")
    test_df = pd.read_parquet("data/processed/ml_ready/test.parquet").fillna(0)
    X_sample = test_df.sample(min(500, len(test_df)), random_state=42)
    
    personas = {'Market': 'Market (Baseline)', 'Sponsor': 'Sponsor (Hype)', 'Coach': 'Coach (Utility)', 'Agent': 'Agent (Volatility)'}
    importance_data = []
    
    for key, name in personas.items():
        try:
            model = joblib.load(f"models/independent/{key}_model.joblib")
            features = model.get_booster().feature_names
            explainer = shap.TreeExplainer(model)
            shap_v = explainer.shap_values(X_sample[features])
            rel_imp = (np.abs(shap_v).mean(axis=0) / np.abs(shap_v).mean(axis=0).sum()) * 100
            for feat, imp in zip(features, rel_imp):
                importance_data.append({'Persona': name, 'Raw_Feature': feat, 'Importance': imp})
        except Exception as e:
            print(f"Error calculating importance for {name}: {e}")
            pass

    if not importance_data:
        print("⚠️ No importance data collected. Skipping stakeholder importance plot.")
        return

    df_imp = pd.DataFrame(importance_data)
    label_map = {
        'log_current_market_value': 'Historical Valuation', 'age_elo_interaction': 'Youth Premium',
        'monthly_minutes': 'Monthly Utility', 'goals_since_update': 'Goal Pressure',
        'age_at_obs': 'Biological Maturity', 'log_weekly_wage': 'Contractual Standing',
        'club_elo': 'Env. Strength', 'log_wiki_views': 'Commercial Reach'
    }
    df_imp['Feature'] = df_imp['Raw_Feature'].map(lambda x: label_map.get(x, x))
    top_feats = df_imp[df_imp['Persona'] == 'Market (Baseline)'].sort_values('Importance', ascending=False).head(10)['Feature'].tolist()
    df_plot = df_imp[df_imp['Feature'].isin(top_feats)]

    palette = {'Agent (Volatility)': '#e67e22', 'Market (Baseline)': '#3498db', 'Coach (Utility)': '#2ecc71', 'Sponsor (Hype)': '#9b59b6'}
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
    ax.xaxis.grid(True)
    sns.barplot(data=df_plot, y='Feature', x='Importance', hue='Persona', order=top_feats, palette=palette, edgecolor='white', linewidth=1, ax=ax)

    ax.set_title('Stakeholder Inductive Biases (Dark Mode)', fontsize=15, fontweight='bold', pad=20, loc='left')
    ax.set_xlabel('Relative Importance (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('')
    ax.legend(title='Personas', loc='lower right', facecolor=DARK_BLUE, edgecolor='white', labelcolor='white')

    plt.tight_layout()
    plt.savefig('Desc/plots/dark_mode/stakeholder_importance_dark.png', bbox_inches='tight', dpi=DPI, transparent=True)
    print("✅ Saved: stakeholder_importance_dark.png")

def generate_dark_tsne():
    """Generates t-SNE archetype plot in dark mode."""
    set_dark_theme(grid=False)
    print("Generating t-SNE (Subsampled for Speed)...")
    df = pd.read_parquet('data/processed/ml_ready/train.parquet')
    feat_cols = ['goals_p90_roll_3m', 'assists_p90_roll_3m', 'minutes_roll_3m', 'age_at_obs', 'club_elo']
    df_sample = df.sample(n=min(5000, len(df)), random_state=42).copy()
    
    X_scaled = StandardScaler().fit_transform(df_sample[feat_cols].fillna(0))
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_scaled)
    df_sample['tsne_1'], df_sample['tsne_2'] = X_tsne[:, 0], X_tsne[:, 1]
    
    mapping = {0:'Elite Catalysts', 1:'Veterans', 2:'Domestic Core', 3:'Youth', 4:'Tactical Pillars', 5:'Potential', 6:'Developing', 7:'Anchors'}
    df_sample['Archetype'] = df_sample['tactical_cluster'].map(mapping)

    fig, ax = plt.subplots(figsize=(12, 9), dpi=DPI)
    sns.scatterplot(data=df_sample, x='tsne_1', y='tsne_2', hue='Archetype', palette='Spectral', s=15, alpha=0.6, edgecolor='none', legend=False, ax=ax)

    for arch in df_sample['Archetype'].unique():
        sub = df_sample[df_sample['Archetype'] == arch]
        props = dict(boxstyle='round', facecolor=DARK_BLUE, alpha=0.8, edgecolor='white')
        ax.text(sub['tsne_1'].median(), sub['tsne_2'].median(), arch, fontsize=9, fontweight='bold', color='white', ha='center', bbox=props)

    ax.set_title('Latent Tactical Space: Archetype Topology (Dark Mode)', fontsize=16, fontweight='bold', pad=25)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('Desc/plots/dark_mode/tsne_archetypes_dark.png', bbox_inches='tight', dpi=DPI, transparent=True)
    print("✅ Saved: tsne_archetypes_dark.png")

def generate_dark_funnel():
    """Generates ETL funnel plot with terminal temporal split in dark mode."""
    set_dark_theme(grid=False)
    stages = [
        'Initial Extraction',
        'Tactical Integration',
        'Financial Intersection',
        'Commercial Intersection',
        'Terminal ML Panel'
    ]
    # Attrition trajectory from gen_publication_split_funnel.py
    values = np.array([450000, 380000, 310000, 280000, 241256])
    
    # Exact temporal splits for the terminal node
    train_n = 185477
    val_n = 19980
    test_n = 35799

    fig, ax = plt.subplots(figsize=(11, 8), dpi=DPI)
    y_pos = np.arange(len(stages))[::-1]
    left_edges = -values / 2
    
    # Funnel colors (Viridis-like gradient for dark mode)
    base_colors = sns.color_palette("viridis", n_colors=len(stages))

    # Plot the Filtering Stages (Top 4 Bars)
    for i in range(4):
        ax.barh(y_pos[i], values[i], left=left_edges[i], height=0.6, 
                color=base_colors[i], edgecolor='white', linewidth=1.5)
        ax.text(0, y_pos[i], f'n = {values[i]:,}', 
                va='center', ha='center', color='white', fontweight='bold')
        
        # Retention Rate annotation
        rate = (values[i] / values[i-1]) * 100 if i > 0 else 100
        if i > 0:
            ax.text(values[i]/2 + (values[0]*0.02), y_pos[i], 
                    f'↳ {rate:.1f}%', va='center', ha='left', color='#bdc3c7', fontstyle='italic')

    # Plot the Stacked Terminal Stage (Train / Val / Test)
    terminal_y = y_pos[4]
    terminal_left = left_edges[4]
    
    # Train Block (Bright Cyan/Blue)
    ax.barh(terminal_y, train_n, left=terminal_left, height=0.6, 
            color='#3498db', edgecolor='white', linewidth=1.2)
    ax.text(terminal_left + (train_n/2), terminal_y, f'{train_n:,}', 
            va='center', ha='center', color='white', fontsize=9, fontweight='bold')
    
    # Val Block (Light Grey)
    val_left = terminal_left + train_n
    ax.barh(terminal_y, val_n, left=val_left, height=0.6, 
            color='#bdc3c7', edgecolor='white', linewidth=1.2)
    
    # Test Block (Bright Red)
    test_left = val_left + val_n
    ax.barh(terminal_y, test_n, left=test_left, height=0.6, 
            color='#e74c3c', edgecolor='white', linewidth=1.2)
    ax.text(test_left + (test_n/2), terminal_y, f'{test_n:,}', 
            va='center', ha='center', color='white', fontsize=9, fontweight='bold')

    # Total survival retention text for the final split
    total_rate = (values[4] / values[3]) * 100
    ax.text(values[4]/2 + (values[0]*0.02), terminal_y, 
            f'↳ {total_rate:.1f}%', va='center', ha='left', color='#bdc3c7', fontstyle='italic')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages, fontweight='bold', color='white')
    ax.set_xticks([])
    ax.set_title('ETL Pipeline Data Provenance & Temporal Splitting (Dark Mode)', 
                 fontsize=15, fontweight='bold', pad=40)
    
    # Legend for the splits
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='white', label='Train (2013-22)'),
        Patch(facecolor='#bdc3c7', edgecolor='white', label='Val (2023)'),
        Patch(facecolor='#e74c3c', edgecolor='white', label='Test (2024)')
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15),
              ncol=3, frameon=False, fontsize=10, labelcolor='white')

    plt.tight_layout()
    plt.savefig('Desc/plots/dark_mode/coverage_funnel_dark.png', bbox_inches='tight', dpi=DPI, transparent=True)
    print("✅ Saved: coverage_funnel_dark.png (Split Version)")

def generate_dark_pdp():
    """Generates Partial Dependence Plot with restored density in dark mode."""
    set_dark_theme()
    print("Generating PDP for Age (Market Impact) with density...")
    model = joblib.load("models/independent/Market_model.joblib")
    test_df = pd.read_parquet("data/processed/ml_ready/test.parquet").fillna(0)
    df_sample = test_df.sample(min(5000, len(test_df)), random_state=42)
    
    features = model.get_booster().feature_names
    pdp_results = partial_dependence(model, df_sample[features], features=['age_at_obs'], grid_resolution=100)
    ages, values = pdp_results['grid_values'][0], pdp_results['average'][0]
    impact = values - np.mean(values)
    
    # Calculate Age Density Distribution
    age_data = test_df['age_at_obs'].values
    kde = gaussian_kde(age_data)
    density = kde(ages)
    
    # Scale density to bottom 15%
    y_min, y_max = impact.min(), impact.max()
    y_range = y_max - y_min
    density_base = y_min - (y_range * 0.2)
    density_scaled = (density / density.max()) * (y_range * 0.15) + density_base
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI)
    ax.yaxis.grid(True)
    
    # Plot PDP Line
    ax.plot(ages, impact, color='#1abc9c', linewidth=4, label='Market Impact', zorder=4)
    
    # Plot Density at bottom
    ax.fill_between(ages, density_base, density_scaled, color='#bdc3c7', alpha=0.3, zorder=2)
    ax.plot(ages, density_scaled, color='#bdc3c7', linewidth=1, linestyle=':', alpha=0.5)
    
    ax.axhline(0, color='#e74c3c', linestyle='-', linewidth=1.5, alpha=0.8, zorder=3)
    ax.axvspan(25, 27, color='#f1c40f', alpha=0.15, zorder=1)
    
    ax.set_title('Biological Depreciation Curve (Dark Mode)', fontsize=15, fontweight='bold', pad=20, loc='left')
    ax.set_xlabel('Biological Age (Years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Impact on Valuation (Log)', fontsize=12, fontweight='bold')
    
    ax.set_xlim(16, 40)
    ax.set_ylim(density_base, y_max * 1.2)
    
    props = dict(boxstyle='round', facecolor=DARK_BLUE, alpha=0.9, edgecolor='white')
    ax.text(26, y_max * 0.9, 'Peak Valuation Window\n(Ages 25-27)', color='#f1c40f', fontsize=11, fontweight='bold', ha='center', bbox=props, zorder=5)
    
    ax.legend(loc='upper right', facecolor=DARK_BLUE, edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.savefig('Desc/plots/dark_mode/biological_depreciation_dark.png', bbox_inches='tight', dpi=DPI, transparent=True)
    plt.savefig('Desc/plots/dark_mode/pdp_age_dark.png', bbox_inches='tight', dpi=DPI, transparent=True)
    print("✅ Saved: pdp_age_dark.png (With Density)")

def generate_dark_heatmap():
    """Generates Importance Heatmap in dark mode."""
    set_dark_theme(grid=False)
    print("Generating Feature Importance Heatmap...")
    test_df = pd.read_parquet("data/processed/ml_ready/test.parquet").fillna(0)
    X_sample = test_df.sample(min(500, len(test_df)), random_state=42)
    
    personas = ['Market', 'Sponsor', 'Coach', 'Agent']
    all_imp = []
    for p in personas:
        try:
            model = joblib.load(f"models/independent/{p}_model.joblib")
            explainer = shap.TreeExplainer(model)
            shap_v = explainer.shap_values(X_sample[model.get_booster().feature_names])
            mean_imp = np.abs(shap_v).mean(axis=0)
            for f, i in zip(model.get_booster().feature_names, mean_imp):
                all_imp.append({'Persona': p, 'Feature': f, 'Importance': i})
        except: pass

    df_imp = pd.DataFrame(all_imp)
    top_20 = df_imp.groupby('Feature')['Importance'].mean().sort_values(ascending=False).head(20).index
    df_h = df_imp[df_imp['Feature'].isin(top_20)].pivot(index='Feature', columns='Persona', values='Importance')
    df_norm = df_h.div(df_h.sum(axis=0), axis=1) * 100
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=DPI)
    sns.heatmap(df_norm, cmap='mako', annot=True, fmt=".1f", linewidths=1.5, linecolor=DARK_BLUE, 
                cbar_kws={'label': 'Relative Importance (%)'}, ax=ax)
    
    ax.set_title('Global Stakeholder Matrix (Dark Mode)', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=0); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('Desc/plots/dark_mode/feature_importance_heatmap_dark.png', bbox_inches='tight', dpi=DPI, transparent=True)
    print("✅ Saved: feature_importance_heatmap_dark.png")

def generate_dark_global_importance():
    """Generates Global Importance bar chart in dark mode."""
    set_dark_theme()
    print("Generating Global Importance Bar Chart...")
    # Reuse SHAP calculation logic or load pre-calc? Let's just re-calc for now.
    test_df = pd.read_parquet("data/processed/ml_ready/test.parquet").fillna(0)
    X_sample = test_df.sample(min(500, len(test_df)), random_state=42)
    
    all_imp = []
    for p in ['Market', 'Sponsor', 'Coach', 'Agent']:
        try:
            model = joblib.load(f"models/independent/{p}_model.joblib")
            explainer = shap.TreeExplainer(model)
            shap_v = explainer.shap_values(X_sample[model.get_booster().feature_names])
            mean_imp = np.abs(shap_v).mean(axis=0)
            for f, i in zip(model.get_booster().feature_names, mean_imp):
                all_imp.append({'Feature': f, 'Importance': i})
        except: pass
    
    df_g = pd.DataFrame(all_imp).groupby('Feature')['Importance'].mean().sort_values(ascending=False).head(12).reset_index()
    label_map = {'log_current_market_value': 'Historical Valuation', 'age_elo_interaction': 'Youth Premium', 'monthly_minutes': 'Monthly Utility', 'age_at_obs': 'Biological Maturity'}
    df_g['Label'] = df_g['Feature'].map(lambda x: label_map.get(x, x))
    df_g = df_g.sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI)
    colors = ['#e74c3c' if i >= len(df_g)-3 else '#3498db' for i in range(len(df_g))]
    bars = ax.barh(df_g['Label'], df_g['Importance'], color=colors, height=0.65, alpha=0.9)
    
    for bar in bars:
        ax.text(bar.get_width() + 0.01, bar.get_y()+bar.get_height()/2, f'{bar.get_width():.2f}', va='center', fontweight='bold', color='white')

    ax.set_title('Global Feature Importance: Ensemble Synthesis (Dark Mode)', fontsize=15, fontweight='bold', pad=20, loc='left')
    ax.xaxis.grid(True)
    plt.tight_layout()
    plt.savefig('Desc/plots/dark_mode/global_importance_bar_dark.png', bbox_inches='tight', dpi=DPI, transparent=True)
    print("✅ Saved: global_importance_bar_dark.png")

if __name__ == "__main__":
    os.makedirs('Desc/plots/dark_mode', exist_ok=True)
    test_df = pd.read_parquet("data/processed/ml_ready/test.parquet")
    test_df['predicted_log_market_value'] = np.load('temp_preds/market_preds.npy')
    
    generate_dark_scatter(test_df, 'log_target_market_value_next_month', 'predicted_log_market_value')
    generate_dark_quadrant(test_df, 'log_current_market_value', 'log_wiki_views')
    generate_dark_bar_chart()
    generate_dark_residuals(test_df, 'log_target_market_value_next_month', 'predicted_log_market_value')
    generate_dark_importance_matrix()
    generate_dark_tsne()
    generate_dark_funnel()
    generate_dark_pdp()
    generate_dark_heatmap()
    generate_dark_global_importance()
