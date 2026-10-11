import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns

def plot_pca_clusters(df):
    plt.figure(figsize=(10, 8))
    # Sample for performance if needed, but 200k points is manageable for scatter if alpha is low
    df_sample = df.sample(min(20000, len(df)), random_state=42)
    
    scatter = plt.scatter(df_sample['PC1'], df_sample['PC2'], 
                          c=df_sample['tactical_cluster'], 
                          cmap='tab10', alpha=0.6, s=15)
    
    plt.title('Tactical Archetypes in 2D Space (PCA Projection)', fontsize=14, fontweight='bold')
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    
    # Add a legend
    ax = plt.gca()
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
    ax.add_artist(legend1)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('Desc/plots/tactical_archetypes_pca.png', dpi=300)
    print("✅ PCA Plot saved.")

def plot_breakout_timeline(df, player_id):
    player_data = df[df['player_id'] == player_id].sort_values('month_date')
    
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Market Value on primary Y-axis
    color_mv = '#1d3557'
    ax1.set_xlabel('Timeline (Month-Year)')
    ax1.set_ylabel('Market Value (€)', color=color_mv, fontsize=12, fontweight='bold')
    ax1.plot(player_data['month_date'], player_data['current_market_value'], 
             color=color_mv, marker='o', linewidth=3, label='Market Value')
    ax1.tick_params(axis='y', labelcolor=color_mv)
    ax1.grid(True, alpha=0.3)

    # Valuation Pressure (Inverse of Elite Distance or Anomaly Score) on secondary Y-axis
    # Using 'dist_to_elite' as a proxy for pressure (lower distance = higher elite potential)
    # We'll plot 1/dist_to_elite or just invert the axis
    ax2 = ax1.twinx()
    color_pres = '#e63946'
    ax2.set_ylabel('Valuation Pressure (Stats)', color=color_pres, fontsize=12, fontweight='bold')
    
    # We can use 'dist_to_elite' inverted or 'is_breakout'
    # Let's use negative dist_to_elite to show "closeness to elite"
    pressure = -player_data['dist_to_elite']
    ax2.plot(player_data['month_date'], pressure, 
             color=color_pres, linestyle='--', linewidth=2, marker='x', label='Statistical Pressure')
    ax2.tick_params(axis='y', labelcolor=color_pres)

    plt.title(f'Breakout Detection Timeline: Player {player_id}\nPredicting the Valuation Spike', fontsize=14, fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig('Desc/plots/breakout_timeline.png', dpi=300)
    print("✅ Breakout Timeline Plot saved.")

def main():
    # Load all data to have a full view
    data_dir = "data/processed/ml_ready"
    train = pd.read_parquet(f"{data_dir}/train.parquet")
    val = pd.read_parquet(f"{data_dir}/val.parquet")
    test = pd.read_parquet(f"{data_dir}/test.parquet")
    df = pd.concat([train, val, test])
    df['month_date'] = pd.to_datetime(df['month_date'])
    
    os.makedirs('Desc/plots', exist_ok=True)
    
    plot_pca_clusters(df)
    plot_breakout_timeline(df, 730581)

if __name__ == "__main__":
    main()
