import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from math import pi

def plot_radar(player_name, scores, filename):
    categories = list(scores.keys())
    N = len(categories)
    values = list(scores.values())
    values += values[:1]
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, color='grey', size=12)
    ax.set_rlabel_position(0)
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#e76f51')
    ax.fill(angles, values, '#e76f51', alpha=0.3)
    plt.title(f'Stakeholder Radar: {player_name}', size=18, color='#264653', y=1.1, fontweight='bold')
    os.makedirs('Desc/plots', exist_ok=True)
    plt.savefig(f'Desc/plots/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    data_dir = "data/processed/ml_ready"
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df.fillna(0, inplace=True)
    
    ids = [467720, 27992] # Youth and Vet
    player_names = {467720: "The Breakout Youth", 27992: "The Declining Veteran"}
    
    final_scores = {pid: {} for pid in ids}
    
    # Load models one by one to avoid segfault
    for p in ['Market', 'Sponsor', 'Coach', 'Agent']:
        model = joblib.load(f"models/independent/{p}_model.joblib")
        feats = model.get_booster().feature_names
        for pid in ids:
            pdata = test_df[test_df['player_id'] == pid].iloc[-1:]
            raw_score = model.predict(pdata[feats])[0]
            # Normalization logic
            if p == 'Coach':
                val = (raw_score / 150) * 10
            elif p == 'Agent':
                val = (raw_score / 1e6) + 5
            else:
                val = (raw_score - 10) / 8 * 10
            final_scores[pid][p] = max(0, min(10, val))
        del model
        
    for pid in ids:
        plot_radar(player_names[pid], final_scores[pid], f"radar_{pid}.png")
        print(f"✅ Radar saved for {pid}")

if __name__ == "__main__":
    main()
