import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from math import pi
import os
from sklearn.inspection import partial_dependence
from src.models.train_nn import MarketValueNN

def get_player_scores(player_id, test_df, xgb_models, nn_model, nn_scaler, nn_features):
    # Get last observation for player
    player_data = test_df[test_df['player_id'] == player_id].iloc[-1:]
    if player_data.empty:
        return None
    
    # Features for XGB
    exclude = [
        'player_id', 'month_date', 'target_market_value_next_month', 
        'target_minutes_next_month', 'target_wiki_views_next_month',
        'log_target_wiki_views_next_month', 'log_target_market_value_next_month',
        'weekly_wage', 'current_market_value', 'wiki_views', 'target_value_change',
        'pretty_name', 'competition_id', 'league' # metadata
    ]
    xgb_features = [c for c in test_df.columns if c not in exclude]
    
    scores = {}
    # Get raw predictions
    for name, model in xgb_models.items():
        # Ensure we use the model's expected features
        feat_names = model.get_booster().feature_names
        scores[name] = model.predict(player_data[feat_names])[0]
        
    # NN score
    X_nn = nn_scaler.transform(player_data[nn_features].values)
    with torch.no_grad():
        scores['DeepLearning'] = nn_model(torch.FloatTensor(X_nn)).cpu().numpy().flatten()[0]
        
    return scores

def plot_radar(player_name, scores, filename):
    # Number of variables
    categories = list(scores.keys())
    N = len(categories)
    
    # We should normalize scores to a 0-1 or 0-100 range for the radar
    # For now, let's just use the values as they are often in similar log-scales
    # or minutes (which might need scaling)
    # Actually, a better radar shows PERCENTILE or RANK relative to others
    
    values = list(scores.values())
    values += values[:1] # repeat first value to close the circle
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    plt.xticks(angles[:-1], categories, color='grey', size=12)
    ax.set_rlabel_position(0)
    
    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#e76f51')
    ax.fill(angles, values, '#e76f51', alpha=0.3)
    
    plt.title(f'Stakeholder Radar: {player_name}', size=20, color='#264653', y=1.1, fontweight='bold')
    
    os.makedirs('Desc/plots', exist_ok=True)
    plt.savefig(f'Desc/plots/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

def generate_pdp(model, X, feature='age_at_obs'):
    print(f"Generating PDP for {feature}...")
    # Use partial_dependence from sklearn
    # Note: XGBoost sklearn wrapper supports this
    features_idx = [list(X.columns).index(feature)]
    pdp_results = partial_dependence(model, X, features=features_idx, grid_resolution=50)
    
    plt.figure(figsize=(10, 6))
    plt.plot(pdp_results['grid_values'][0], pdp_results['average'][0], color='#2a9d8f', linewidth=3)
    plt.title(f'Partial Dependence Plot: {feature} vs. Valuation', fontsize=14, fontweight='bold')
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Partial Dependence (Log MV)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Highlight the "Peak"
    peak_val = pdp_results['grid_values'][0][np.argmax(pdp_results['average'][0])]
    plt.axvline(x=peak_val, color='#e63946', linestyle='--', label=f'Theoretical Peak: ~{peak_val:.1f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'Desc/plots/pdp_{feature}.png', dpi=300)
    plt.close()

def main():
    data_dir = "data/processed/ml_ready"
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df.fillna(0, inplace=True)
    
    # Load Models
    indep_dir = "models/independent"
    nn_dir = "models/nn"
    xgb_models = {p: joblib.load(f"{indep_dir}/{p}_model.joblib") for p in ['Market', 'Sponsor', 'Coach', 'Agent']}
    
    import json
    with open(f"{nn_dir}/metadata_nn.json", "r") as f:
        nn_meta = json.load(f)
    nn_model = MarketValueNN(len(nn_meta['features']), nn_meta['best_params']['hidden_dims'], nn_meta['best_params']['dropout'])
    nn_model.load_state_dict(torch.load(f"{nn_dir}/market_value_nn.pth"))
    nn_model.eval()
    nn_scaler = joblib.load(f"{nn_dir}/scaler_nn.joblib")

    # 1. Radar Charts for Case Studies
    # We'll use the players we found earlier: 467720 (Youth) and 27992 (Vet)
    # For meaningful radars, we normalize the scores
    
    def get_norm_scores(pid, df, xgb_models, nn_model, nn_scaler, nn_features):
        raw = get_player_scores(pid, df, xgb_models, nn_model, nn_scaler, nn_features)
        if not raw: return None
        # Normalization: simple log-market-value scale is already ~10-18. 
        # For minutes it's 0-100. Let's just scale everything to 0-10 relative to dataset max for visualization.
        norm = {
            'Market (Stability)': (raw['Market'] - 10) / 8 * 10,
            'Sponsor (Hype)': (raw['Sponsor'] - 10) / 8 * 10,
            'Coach (Utility)': (raw['Coach'] / 150) * 10 if raw['Coach'] > 0 else 0,
            'Agent (Growth)': (raw['Agent'] / 1e6) + 5, # Offset to show change
            'DeepLearning': (raw['DeepLearning'] - 10) / 8 * 10
        }
        return {k: max(0, min(10, v)) for k, v in norm.items()}

    scores_youth = get_norm_scores(467720, test_df, xgb_models, nn_model, nn_scaler, nn_meta['features'])
    if scores_youth: plot_radar("The Breakout Youth (ID 467720)", scores_youth, "radar_youth.png")
    
    scores_vet = get_norm_scores(27992, test_df, xgb_models, nn_model, nn_scaler, nn_meta['features'])
    if scores_vet: plot_radar("The Declining Veteran (ID 27992)", scores_vet, "radar_veteran.png")

    # 2. PDP for Age
    # Use Market Model for PDP as it's the primary valuation driver
    market_model = xgb_models['Market']
    feat_names = market_model.get_booster().feature_names
    generate_pdp(market_model, test_df[feat_names], 'age_at_obs')

if __name__ == "__main__":
    main()
