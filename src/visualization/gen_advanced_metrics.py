import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_importance_visuals():
    # Load test data
    data_dir = "data/processed/ml_ready"
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df.fillna(0, inplace=True)
    
    sample_size = 500
    X_sample = test_df.sample(min(sample_size, len(test_df)), random_state=42)
    
    personas = ['Market', 'Sponsor', 'Coach', 'Agent']
    indep_dir = "models/independent"
    
    all_importances = []
    
    for persona in personas:
        print(f"Calculating SHAP for {persona}...")
        try:
            model = joblib.load(f"{indep_dir}/{persona}_model.joblib")
            features = model.get_booster().feature_names
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample[features])
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            for feat, imp in zip(features, mean_abs_shap):
                all_importances.append({'Persona': persona, 'Feature': feat, 'Importance': imp})
        except Exception as e:
            print(f"Error with {persona}: {e}")

    df_imp = pd.DataFrame(all_importances)
    
    # 1. Feature Importance Heatmap (Scale-Normalized for Publication)
    top_20_features = df_imp.groupby('Feature')['Importance'].mean().sort_values(ascending=False).head(20).index
    df_heatmap = df_imp[df_imp['Feature'].isin(top_20_features)].pivot(index='Feature', columns='Persona', values='Importance')
    
    # Rename columns for the publication narrative
    rename_map = {
        'Market': 'Market (Baseline)',
        'Sponsor': 'Sponsor (Hype)',
        'Coach': 'Coach (Utility)',
        'Agent': 'Agent (Volatility)'
    }
    df_heatmap = df_heatmap.rename(columns=rename_map)
    
    # THE FIX: Column-wise Normalization (Relative Percentage)
    # This prevents the "Agent Domination" artifact where high-volatility financial targets
    # wash out the structural/utility targets of other personas.
    df_normalized = df_heatmap.div(df_heatmap.sum(axis=0), axis=1) * 100
    
    # Configure Academic Formatting
    plt.style.use('seaborn-v0_8-paper')
    sns.set_theme(style="white", rc={
        "font.family": "serif",
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    
    sns.heatmap(
        df_normalized, 
        cmap='YlGnBu', 
        annot=True,            # Show the numbers inside the tiles
        fmt=".1f",             # Format to 1 decimal place
        linewidths=1.5,        # Add clean white gridlines between tiles
        linecolor='white',
        cbar_kws={'label': 'Relative Feature Importance (%)', 'pad': 0.02},
        ax=ax
    )
    
    ax.set_title('Global Stakeholder Matrix: Relative Feature Importance', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Ensemble Persona Models', fontsize=12, labelpad=15)
    ax.set_ylabel('Engineered Features', fontsize=12, labelpad=15)

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('Desc/plots/feature_importance_heatmap.png', dpi=300)
    print("✅ Scale-normalized heatmap saved to Desc/plots/feature_importance_heatmap.png")
    
    # 2. Global Importance Bar Chart (Narrative Highlighting for Publication)
    df_global = df_imp.groupby('Feature')['Importance'].mean().sort_values(ascending=False).head(12).reset_index()
    
    # THE FIX: Label Mapping Dictionary
    label_mapping = {
        'log_current_market_value': 'Historical Valuation Gravity',
        'age_elo_interaction': 'The Youth Premium (Age x Elo)',
        'monthly_minutes': 'Isolated Monthly Utility',
        'goals_since_update': 'Goal Accumulation Pressure',
        'PC2': 'Tactical Latent Feature (PC2)',
        'age_at_obs': 'Biological Maturity (Age)',
        'months_since_last_update': 'Time Decay Since Valuation',
        'value_per_minute': 'Efficiency (Value/Minute)',
        'hype_to_value_ratio': 'Attention Alpha (Wiki/Value)',
        'elo_momentum_3m': 'Team Momentum (3M Elo Δ)',
        'mv_momentum_3m': 'Price Momentum (3M Value Δ)',
        'minutes_roll_3m': 'Tactical Reliability (Rolling Mins)',
        'log_weekly_wage': 'Contractual Standing (Wage)',
        'club_elo': 'Environmental Strength (Club Elo)',
        'hype_since_update': 'Hype Accumulation Pressure'
    }
    
    df_global['Clean_Label'] = df_global['Feature'].map(lambda x: label_mapping.get(x, x))
    
    # Sort for plotting (bottom up)
    df_global = df_global.sort_values('Importance', ascending=True)

    plt.figure(figsize=(10, 7), dpi=300)
    
    # Narrative Color Highlighting: Crimson for Top 3, Navy for the rest
    colors = ['#8b0000' if i >= len(df_global) - 3 else '#2c3e50' for i in range(len(df_global))]
    
    bars = plt.barh(df_global['Clean_Label'], df_global['Importance'], color=colors, height=0.65, alpha=0.9)
    
    # Add Exact Data Labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + (df_global['Importance'].max() * 0.01), 
                bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}', 
                va='center', ha='left', fontsize=10, fontweight='bold', color='#333333')

    plt.title('Global Feature Importance: Ensemble Synthesis', fontsize=15, fontweight='bold', pad=20, loc='left')
    plt.xlabel('Mean Absolute SHAP Value (Impact on Log Valuation)', fontsize=12, labelpad=10, fontweight='bold')
    
    # Final styling
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tick_params(axis='y', labelsize=11, left=False)
    
    # Custom Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#8b0000', label='Primary Valuation Drivers (Top 3)'),
        Patch(facecolor='#2c3e50', label='Secondary Contextual Drivers')
    ]
    plt.legend(handles=legend_elements, loc='lower right', frameon=True, edgecolor='#d3d3d3')

    plt.tight_layout()
    plt.savefig('Desc/plots/global_importance_bar.png', dpi=300)
    print("✅ Narrative-driven bar chart saved to Desc/plots/global_importance_bar.png")

def generate_infrastructure_visuals():
    coverage = {
        'Match Coverage': 1.0,
        'Wage Coverage': 0.76,
        'Wikipedia Coverage': 0.74,
        'Full ML Ready': 0.70
    }
    
    # 3. Coverage Funnel
    plt.figure(figsize=(10, 6))
    stages = list(coverage.keys())
    values = list(coverage.values())
    sns.barplot(x=values, y=stages, palette='viridis')
    for i, v in enumerate(values):
        plt.text(v + 0.01, i, f"{v*100:.1f}%", va='center', fontweight='bold')
    plt.title('Data Infrastructure Coverage Funnel (Top 5 Leagues)', fontsize=14, fontweight='bold')
    plt.xlim(0, 1.1)
    plt.tight_layout()
    plt.savefig('Desc/plots/coverage_funnel.png', dpi=300)

    # 4. League Benchmark Table
    print("Generating League Benchmark Table...")
    data_dir = "data/processed/ml_ready"
    df = pd.read_parquet(f"{data_dir}/test.parquet")
    
    # Join with id_mappings for competition info
    try:
        mappings = pd.read_csv('data/interim/id_mappings.csv')
        df = df.merge(mappings[['player_id', 'competition_id']], on='player_id', how='left')
        
        comp_map = {
            'GB1': 'Premier League',
            'ES1': 'La Liga',
            'IT1': 'Serie A',
            'L1': 'Bundesliga',
            'FR1': 'Ligue 1'
        }
        df['league'] = df['competition_id'].map(comp_map)
        df = df.dropna(subset=['league'])
        
        # Calculate benchmarks
        benchmarks = df.groupby('league').agg({
            'current_market_value': 'mean',
            'weekly_wage': 'mean'
        }).reset_index()
        
        benchmarks['wage_to_value_ratio'] = (benchmarks['weekly_wage'] * 52) / benchmarks['current_market_value']
        
        # Round and format
        benchmarks['current_market_value'] = benchmarks['current_market_value'] / 1e6 # to Millions
        benchmarks.columns = ['League', 'Avg MV (€M)', 'Avg Wage (€)', 'Wage/Value Ratio']
        benchmarks = benchmarks.round(3)
        
        print("\nLeague Benchmark Table:")
        print(benchmarks.to_string(index=False))
        
        # Save to CSV for the report to read
        benchmarks.to_csv('Desc/league_benchmarks.csv', index=False)
        print("✅ League benchmarks saved.")
        
    except Exception as e:
        print(f"Error generating league benchmarks: {e}")
    
if __name__ == "__main__":
    os.makedirs('Desc/plots', exist_ok=True)
    generate_importance_visuals()
    generate_infrastructure_visuals()
