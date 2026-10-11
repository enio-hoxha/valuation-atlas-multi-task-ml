import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
import joblib
import os
import json
import duckdb

def load_all_models(indep_dir="models/independent", ensemble_dir="models/ensemble"):
    models = {}
    for persona in ['Market', 'Sponsor', 'Coach', 'Agent']:
        models[persona] = joblib.load(f"{indep_dir}/{persona}_model.joblib")
    
    with open(f"{ensemble_dir}/ensemble_metadata.json", "r") as f:
        ensemble_meta = json.load(f)
    
    # Load pre-computed NN predictions
    with open("nn_preds_temp.json", "r") as f:
        nn_preds = json.load(f)
    
    return models, ensemble_meta, nn_preds

def get_player_metadata(player_ids):
    con = duckdb.connect('transfermarkt-datasets.duckdb')
    ids_str = ",".join(map(str, player_ids))
    query = f"""
    SELECT p.player_id, p.name, p.current_club_name, p.position, p.date_of_birth, 
           p.country_of_citizenship, p.market_value_in_eur
    FROM players p
    WHERE p.player_id IN ({ids_str})
    """
    df = con.execute(query).df()
    con.close()
    return df

def generate_html_report(p_name, metadata, top_features, top_shaps, final_val, output_dir):
    def clean_feat(f):
        return f.replace('_', ' ').title()

    drivers = []
    for f, s in zip(top_features, top_shaps):
        if np.abs(s) < 0.001: continue
        perc = (np.exp(s) - 1) * 100
        drivers.append({
            'name': clean_feat(f),
            'perc': f"{'+' if perc > 0 else ''}{perc:.1f}%",
            'is_pos': perc > 0
        })
    
    boosters = [d for d in drivers if d['is_pos']][::-1][:4]
    detractors = [d for d in drivers if not d['is_pos']][:4]
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{p_name} - Analytics</title>
        <style>
            body {{ font-family: 'Inter', sans-serif; background-color: #0d1117; color: #e6edf3; padding: 40px; }}
            .container {{ max-width: 1000px; margin: auto; background: #161b22; padding: 40px; border-radius: 16px; border: 1px solid #30363d; }}
            .header {{ display: flex; justify-content: space-between; border-bottom: 1px solid #30363d; padding-bottom: 20px; margin-bottom: 30px; }}
            h1 {{ color: #58a6ff; margin: 0; }}
            .val {{ color: #3fb950; font-size: 2em; font-weight: bold; }}
            .viz {{ background: #0d1117; padding: 20px; border-radius: 12px; margin-bottom: 30px; border: 1px solid #30363d; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            .card {{ background: #0d1117; padding: 20px; border-radius: 8px; border: 1px solid #30363d; }}
            .row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #21262d; }}
            .pos {{ color: #3fb950; }} .neg {{ color: #f85149; }}
            .back {{ color: #8b949e; text-decoration: none; margin-bottom: 20px; display: inline-block; }}
        </style>
    </head>
    <body>
        <div class="container">
            <a href="index.html" class="back">← Back to Dashboard</a>
            <div class="header">
                <div><h1>{p_name}</h1><p>{metadata['position']} • {metadata['current_club_name']}</p></div>
                <div style="text-align:right"><p style="color:#8b949e;margin:0">Predicted Market Value</p><div class="val">€{final_val/1e6:.1f}M</div></div>
            </div>
            <div class="viz"><img src="{p_name.replace(' ', '_')}_Waterfall.png" style="width:100%"></div>
            <div class="grid">
                <div class="card"><h3>Value Drivers</h3>{"".join([f'<div class="row"><span>{d["name"]}</span><span class="pos">{d["perc"]}</span></div>' for d in boosters])}</div>
                <div class="card"><h3>Risk Factors</h3>{"".join([f'<div class="row"><span>{d["name"]}</span><span class="neg">{d["perc"]}</span></div>' for d in detractors])}</div>
            </div>
        </div>
    </body>
    </html>
    """
    with open(f"{output_dir}/{p_name.replace(' ', '_')}.html", "w") as f:
        f.write(html_content)

def plot_waterfall_horizontal(p_name, feature_names, shap_values, baseline_log, nn_adj, output_path):
    idx = np.argsort(np.abs(shap_values))
    top_k = 10
    top_idx = idx[-top_k:]
    other_shap = np.sum(shap_values[idx[:-top_k]])
    
    display_shaps = [shap_values[i] for i in top_idx]
    display_names = [feature_names[i].replace('_', ' ').title() for i in top_idx]
    
    all_vals = [baseline_log] + display_shaps + [other_shap, nn_adj]
    all_names = ["Baseline (Model Expected)"] + display_names + ["Other Features", "Neural Net Correction"]
    
    plt.figure(figsize=(13, 8))
    plt.gca().set_facecolor('#0d1117')
    plt.gcf().set_facecolor('#0d1117')
    
    cumulative = np.cumsum(all_vals)
    for i in range(len(all_vals)):
        start = cumulative[i-1] if i > 0 else 0
        end = cumulative[i]
        
        if i == 0: color = '#58a6ff'
        elif i == len(all_vals) - 1: color = '#bc8cff'
        elif i == len(all_vals) - 2: color = '#8b949e'
        else: color = '#3fb950' if all_vals[i] > 0 else '#f85149'
        
        plt.barh(i, end - start, left=start, color=color, edgecolor='#30363d', height=0.7)
        
        val_m = np.exp(end) / 1e6
        plt.text(max(start, end) + 0.1, i, f"€{val_m:.1f}M", va='center', color='#c9d1d9', fontsize=9, fontweight='bold')

    plt.yticks(range(len(all_names)), all_names, color='#8b949e')
    plt.xticks(color='#8b949e')
    plt.title(f"Market Value Decomposition: {p_name}", color='#58a6ff', fontsize=16, pad=20)
    plt.grid(axis='x', linestyle='--', alpha=0.1)
    plt.tight_layout()
    plt.savefig(output_path, facecolor='#0d1117', dpi=120)
    plt.close()

def generate_index_html(players, output_dir):
    items = "".join([f'<li><a href="{p.replace(" ", "_")}.html">{p}</a></li>' for p in players])
    html = f"<html><body style='background:#0d1117;color:white;font-family:sans-serif;padding:50px;text-align:center;'><div style='max-width:500px;margin:auto;background:#161b22;padding:40px;border-radius:12px;border:1px solid #30363d;'><h1>Value Dashboard</h1><ul style='list-style:none;padding:0;text-align:left;'>{items}</ul></div></body></html>"
    with open(f"{output_dir}/index.html", "w") as f: f.write(html)

def generate_player_shaps():
    indep_dir, ensemble_dir, data_dir, output_dir = "models/independent", "models/ensemble", "data/processed/ml_ready", "TreeSHAPs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading models and data...")
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    xgb_models, ensemble_meta, nn_preds = load_all_models()
    meta_model = joblib.load(f"{ensemble_dir}/stacking_meta_model.joblib")
    
    exclude = ['player_id', 'month_date', 'target_market_value_next_month', 'target_minutes_next_month', 
               'log_target_wiki_views_next_month', 'target_value_change', 'weekly_wage', 'current_market_value', 'wiki_views']
    feature_cols = [c for c in test_df.columns if c not in exclude]
    
    top_players_df = test_df.sort_values('current_market_value', ascending=False).drop_duplicates('player_id').head(12)
    
    # Force include Ciro Immobile for the user request
    immobile_row = test_df[test_df['player_id'] == 105521].sort_values('month_date', ascending=False).head(1)
    if not immobile_row.empty:
        top_players_df = pd.concat([top_players_df, immobile_row])
    
    metadata_df = get_player_metadata(top_players_df['player_id'].tolist()).set_index('player_id')
    
    coeffs = ensemble_meta['coefficients']
    persona_map = {'Market': 'market', 'Sponsor': 'sponsor', 'Coach': 'coach', 'Agent': 'agent'}
    explainers = {p: shap.TreeExplainer(m) for p, m in xgb_models.items()}
    
    final_players = []
    for _, row in top_players_df.iterrows():
        p_id = row['player_id']
        if p_id not in metadata_df.index: continue
        p_name = metadata_df.loc[p_id, 'name']
        print(f"Processing {p_name}...")
        final_players.append(p_name)
        
        p_data = test_df.loc[[row.name]]
        
        f_shaps = np.zeros(len(feature_cols))
        intercept = meta_model.intercept_
        
        for persona, model_key in persona_map.items():
            if model_key in coeffs:
                w, exp = coeffs[model_key], explainers[persona]
                m_feats = xgb_models[persona].get_booster().feature_names
                p_data_fixed = p_data[m_feats]
                sv = exp.shap_values(p_data_fixed, check_additivity=False)
                if isinstance(sv, list): sv = sv[0]
                if len(sv.shape) == 2: sv = sv[0]
                
                intercept += w * exp.expected_value
                f_to_idx = {f: i for i, f in enumerate(feature_cols)}
                for i, f_name in enumerate(m_feats):
                    if f_name in f_to_idx: f_shaps[f_to_idx[f_name]] += w * sv[i]

        nn_adj = 0
        if str(p_id) in nn_preds:
            nn_adj = coeffs.get('nn', 0) * nn_preds[str(p_id)]

        final_val = np.exp(intercept + np.sum(f_shaps) + nn_adj)
        print(f"  Predicted: €{final_val/1e6:.1f}M")
        
        img_path = f"{output_dir}/{p_name.replace(' ', '_')}_Waterfall.png"
        plot_waterfall_horizontal(p_name, feature_cols, f_shaps, intercept, nn_adj, img_path)
        generate_html_report(p_name, metadata_df.loc[p_id], feature_cols, f_shaps, final_val, output_dir)

    generate_index_html(final_players, output_dir)
    print(f"\n🚀 Dashboard ready: {output_dir}/index.html")

if __name__ == "__main__":
    generate_player_shaps()
