import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import seaborn as sns

def main():
    # Load test data
    data_dir = "data/processed/ml_ready"
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df.fillna(0, inplace=True)
    
    # Use a sample for SHAP calculation speed
    sample_size = 500
    X_sample = test_df.sample(min(sample_size, len(test_df)), random_state=42)
    
    personas = {
        'Market': 'Market (Baseline)',
        'Sponsor': 'Sponsor (Hype)',
        'Coach': 'Coach (Utility)',
        'Agent': 'Agent (Volatility)'
    }
    indep_dir = "models/independent"
    
    importance_data = []
    
    for key, name in personas.items():
        print(f"Calculating SHAP for {name}...")
        try:
            model = joblib.load(f"{indep_dir}/{key}_model.joblib")
            features = model.get_booster().feature_names
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample[features])
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            # THE FIX: Intra-Model Normalization
            # Force this model's importance to sum to 100%
            total_importance = np.sum(mean_abs_shap)
            relative_importance = (mean_abs_shap / total_importance) * 100
            
            for feat, imp in zip(features, relative_importance):
                importance_data.append({
                    'Persona': name,
                    'Raw_Feature': feat,
                    'Importance': imp
                })
        except Exception as e:
            print(f"Error with {name}: {e}")

    df_imp = pd.DataFrame(importance_data)
    
    # 1. Label Mapping for Academic Context
    label_mapping = {
        'log_current_market_value': 'Historical Valuation Gravity',
        'age_elo_interaction': 'Youth Premium (Age x Elo)',
        'monthly_minutes': 'Isolated Monthly Utility',
        'goals_since_update': 'Goal Accumulation Pressure',
        'PC2': 'Tactical Latent Feature (PC2)',
        'age_at_obs': 'Biological Maturity (Age)',
        'months_since_last_update': 'Time Decay Since Valuation',
        'value_per_minute': 'Efficiency (Value/Minute)',
        'hype_to_value_ratio': 'Attention Alpha (Wiki/Value)',
        'elo_momentum_3m': 'Team Momentum (3M Elo Δ)',
        'mv_momentum_3m': 'Price Momentum (3M Value Δ)',
        'minutes_roll_3m': 'Tactical Reliability (Mins)',
        'log_weekly_wage': 'Contractual Standing (Wage)',
        'club_elo': 'Environmental Strength (Elo)',
        'log_wiki_views': 'Commercial Reach (Log Views)',
        'dist_to_elite': 'Proximity to Elite Centroid'
    }
    
    df_imp['Feature'] = df_imp['Raw_Feature'].map(lambda x: label_mapping.get(x, x))
    
    # 2. Identify top features to keep the chart clean
    # Sort by the importance in the Market Baseline model
    top_features_list = df_imp[df_imp['Persona'] == 'Market (Baseline)'].sort_values('Importance', ascending=False).head(10)['Feature'].tolist()
    df_plot = df_imp[df_imp['Feature'].isin(top_features_list)]

    # 3. Configure Academic Formatting
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

    # 4. Define Categorical Palette
    palette = {
        'Agent (Volatility)': '#d35400', # Burnt Orange
        'Market (Baseline)':  '#2c3e50', # Deep Navy
        'Coach (Utility)':    '#27ae60', # Academic Green
        'Sponsor (Hype)':     '#8e44ad'  # Muted Purple
    }

    # 5. Initialize Figure and Plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.xaxis.grid(True) # Explicitly enable X-axis grid
    
    sns.barplot(
        data=df_plot, 
        y='Feature', 
        x='Importance', 
        hue='Persona',
        order=top_features_list,
        palette=palette,
        edgecolor='white',
        linewidth=1,
        ax=ax
    )

    # 6. Typography and Labels
    ax.set_title('Normalized Stakeholder Inductive Biases\n(Relative SHAP Feature Importance)', 
                 fontsize=15, fontweight='bold', pad=20, loc='left')
    
    ax.set_xlabel('Relative Contribution to Model Predictions (%)', fontsize=12, labelpad=10, fontweight='bold')
    ax.set_ylabel('')
    
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=11)

    ax.legend(title='Ensemble Personas', title_fontsize=11, fontsize=10, 
              loc='lower right', frameon=True, edgecolor='#d3d3d3')

    plt.tight_layout()
    
    os.makedirs('Desc/plots', exist_ok=True)
    save_path = 'Desc/plots/stakeholder_importance_matrix_publication.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Normalized stakeholder matrix saved to {save_path}")

if __name__ == "__main__":
    main()
