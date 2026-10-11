import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns

def plot_hype_vs_value(df):
    plt.figure(figsize=(12, 10))
    
    # Use log scales for both
    x = df['log_wiki_views']
    y = df['log_current_market_value']
    
    # Sample to avoid overplotting
    df_sample = df.sample(min(5000, len(df)), random_state=42)
    
    sns.scatterplot(data=df_sample, x='log_wiki_views', y='log_current_market_value', 
                    hue='is_breakout', palette={0: '#264653', 1: '#e76f51'}, alpha=0.6)
    
    # Draw quadrant lines at medians
    plt.axvline(x.median(), color='black', linestyle='--', alpha=0.5)
    plt.axhline(y.median(), color='black', linestyle='--', alpha=0.5)
    
    # Annotate quadrants
    plt.text(x.max()-1, y.max()-0.5, 'ESTABLISHED STARS', fontsize=12, fontweight='bold', color='grey')
    plt.text(x.min()+0.5, y.max()-0.5, 'UNDER-THE-RADAR\nWORKHORSES', fontsize=12, fontweight='bold', color='grey')
    plt.text(x.min()+0.5, y.min()+0.5, 'DEVELOPING TALENTS', fontsize=12, fontweight='bold', color='grey')
    plt.text(x.max()-1, y.min()+0.5, 'THE NEXT BIG THING\n(Breakout Candidates)', fontsize=12, fontweight='bold', color='#e76f51')
    
    plt.title('Hype vs. Value Quadrant Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Hype (Log Wiki Views)', fontsize=12)
    plt.ylabel('Market Value (Log Current MV)', fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('Desc/plots/hype_vs_value_quadrant.png', dpi=300)
    print("✅ Hype vs Value Plot saved.")

def plot_forward_fill_gap(df, player_id):
    player_data = df[df['player_id'] == player_id].sort_values('month_date')
    
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Market Value (The "Step-Ladder")
    ax1.step(player_data['month_date'], player_data['current_market_value'], 
             where='post', color='#1d3557', linewidth=3, label='Transfermarkt (Steps)')
    ax1.set_xlabel('Timeline')
    ax1.set_ylabel('Market Value (€)', color='#1d3557', fontsize=12, fontweight='bold')
    
    # Statistical Pressure (The "Smooth Lead")
    ax2 = ax1.twinx()
    # Use -dist_to_elite normalized to show growth
    pressure = -player_data['dist_to_elite']
    pressure = (pressure - pressure.min()) / (pressure.max() - pressure.min())
    
    ax2.plot(player_data['month_date'], pressure, 
             color='#e63946', linewidth=2, linestyle='-', marker='x', label='Model "Pressure" (Smooth)')
    ax2.set_ylabel('Predictive Pressure (Normalized)', color='#e63946', fontsize=12, fontweight='bold')
    
    # Highlight a "Gap"
    # Find a month where pressure rises but MV stays flat
    plt.title(f'The Forward-Fill Gap: Model vs. Market (Player {player_id})', fontsize=14, fontweight='bold')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig('Desc/plots/forward_fill_gap.png', dpi=300)
    print("✅ Forward-Fill Gap Plot saved.")

def plot_burst_calibration(df):
    plt.figure(figsize=(10, 8))
    
    # We want to show: High Pressure (low dist_to_elite) leads to high Value Change
    # Calculate value change for the next month
    df = df.sort_values(['player_id', 'month_date'])
    df['next_mv_change'] = df.groupby('player_id')['current_market_value'].shift(-1) - df['current_market_value']
    
    # Filter for significant jumps
    df_burst = df[df['next_mv_change'] > 0].copy()
    
    # Correlation between dist_to_elite and next_mv_change
    sns.regplot(data=df_burst.sample(min(2000, len(df_burst))), 
                x='dist_to_elite', y='next_mv_change', 
                scatter_kws={'alpha':0.3, 'color':'#2a9d8f'}, line_kws={'color':'#e63946'})
    
    plt.title('Burst Calibration Curve: Pressure vs. Value Jump', fontsize=14, fontweight='bold')
    plt.xlabel('Elite Distance (Lower = Higher Pressure)', fontsize=12)
    plt.ylabel('Next Month Value Jump (€)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Desc/plots/burst_calibration_curve.png', dpi=300)
    print("✅ Burst Calibration Plot saved.")

def main():
    data_dir = "data/processed/ml_ready"
    train = pd.read_parquet(f"{data_dir}/train.parquet")
    val = pd.read_parquet(f"{data_dir}/val.parquet")
    test = pd.read_parquet(f"{data_dir}/test.parquet")
    df = pd.concat([train, val, test])
    df['month_date'] = pd.to_datetime(df['month_date'])
    
    os.makedirs('Desc/plots', exist_ok=True)
    
    plot_hype_vs_value(df)
    plot_forward_fill_gap(df, 730581)
    plot_burst_calibration(df)

if __name__ == "__main__":
    main()
