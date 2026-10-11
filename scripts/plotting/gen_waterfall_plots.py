import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

def generate_waterfall(player_id, title, filename, label_map):
    # Load test data and model
    data_dir = "data/processed/ml_ready"
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df.fillna(0, inplace=True)
    
    model = joblib.load("models/independent/Market_model.joblib")
    features = model.get_booster().feature_names
    
    # Get the specific player instance (last observation in test set)
    player_data = test_df[test_df['player_id'] == player_id].iloc[-1:]
    
    # SHAP calculation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(player_data[features])
    
    # Create the waterfall plot
    plt.figure(figsize=(12, 8))
    
    # We'll use the shap.plots.waterfall but customize the labels after
    # To do that, we modify the feature names in the shap_values object
    new_names = []
    for name in features:
        if name in label_map:
            new_names.append(f"{name} [{label_map[name]}]")
        else:
            new_names.append(name)
    
    shap_values.feature_names = new_names
    
    shap.plots.waterfall(shap_values[0], show=False)
    
    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    
    os.makedirs('Desc/plots', exist_ok=True)
    plt.savefig(f'Desc/plots/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Waterfall saved: {filename}")

def main():
    # 1. The Breakout Youth (ID: 467720)
    # Labels for Narrative Primitives
    labels_youth = {
        'club_elo': 'THE ELITE PREMIUM',
        'age_at_obs': 'THE PROMISE MULTIPLIER',
        'log_current_market_value': 'VALUATION ANCHOR',
        'is_breakout': 'BREAKOUT ALPHA'
    }
    generate_waterfall(467720, "Narrative Primitive: The Breakout Youth (ID 467720)", "waterfall_youth.png", labels_youth)
    
    # 2. The Declining Veteran (ID: 27992)
    labels_vet = {
        'age_at_obs': 'BIOLOGICAL DEPRECIATION',
        'log_current_market_value': 'MARKET GRAVITY',
        'minutes_roll_3m': 'UTILITY DECAY'
    }
    generate_waterfall(27992, "Narrative Primitive: The Declining Veteran (ID 27992)", "waterfall_veteran.png", labels_vet)

if __name__ == "__main__":
    main()
