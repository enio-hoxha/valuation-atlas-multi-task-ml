import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os

def generate_publication_tsne():
    """
    Generates a high-resolution, density-aware t-SNE projection of Tactical Archetypes.
    """
    print("Loading data for t-SNE projection...")
    data_path = 'data/processed/ml_ready/train.parquet'
    df = pd.read_parquet(data_path)
    
    # Define features used for clustering
    feature_cols = [
        'goals_p90_roll_3m', 'assists_p90_roll_3m', 'minutes_roll_3m', 
        'age_at_obs', 'club_elo'
    ]
    
    # Subsample for performance (10k points is the standard for legible t-SNE)
    n_samples = 10000
    df_sample = df.sample(n=min(n_samples, len(df)), random_state=42).copy()
    
    print(f"Performing t-SNE on {len(df_sample)} samples...")
    X_sample_scaled = StandardScaler().fit_transform(df_sample[feature_cols].fillna(0))
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_sample_scaled)
    
    df_sample['tsne_1'] = X_tsne[:, 0]
    df_sample['tsne_2'] = X_tsne[:, 1]

    # Map Numerical Clusters to Tactical Narrative Primitives (based on mean features)
    archetype_mapping = {
        0: 'Elite Offensive Catalysts',
        1: 'Substantial Veteran Presence',
        2: 'Standard Domestic Assets',
        3: 'Emerging Youth Prospects',
        4: 'Reliable Tactical Pillars',
        5: 'High-Potential Rotation',
        6: 'Developing Mid-Table Core',
        7: 'Experienced Defensive Anchors'
    }
    df_sample['Tactical_Archetype'] = df_sample['tactical_cluster'].map(archetype_mapping)

    # 3. Configure Academic Formatting
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        plt.style.use('seaborn-paper')
        
    sns.set_theme(style="white", rc={
        "font.family": "serif",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.grid": False 
    })

    # 4. Initialize Figure
    fig, ax = plt.subplots(figsize=(12, 9), dpi=300)

    # 5. Density-Aware Scatter Plot
    sns.scatterplot(
        data=df_sample, 
        x='tsne_1', 
        y='tsne_2', 
        hue='Tactical_Archetype', 
        palette='husl',
        s=12, 
        alpha=0.5, 
        edgecolor='none',
        legend=False,
        ax=ax
    )

    # 6. Calculate Centroids and Add Direct Typography Overlays
    for archetype in df_sample['Tactical_Archetype'].unique():
        subset = df_sample[df_sample['Tactical_Archetype'] == archetype]
        
        # Using median for robustness
        x_centroid = subset['tsne_1'].median()
        y_centroid = subset['tsne_2'].median()
        
        # Add a sleek text box at the center of the cluster
        props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='none')
        ax.text(x_centroid, y_centroid, archetype, 
                fontsize=10, fontweight='bold', color='#2c3e50',
                ha='center', va='center', bbox=props, zorder=10)

    # 7. Typography and Labels
    ax.set_title('Latent Tactical Space: t-SNE Topology of K-Means Archetypes', 
                 fontsize=16, fontweight='bold', pad=25)
    
    ax.set_xlabel('t-SNE Dimension 1 (Local Proximity)', fontsize=11, fontstyle='italic', color='#7f8c8d')
    ax.set_ylabel('t-SNE Dimension 2 (Local Proximity)', fontsize=11, fontstyle='italic', color='#7f8c8d')

    # Strip axes for t-SNE "map" look
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    
    # Save as high-res PNG
    os.makedirs('Desc/plots', exist_ok=True)
    save_path = 'Desc/plots/tsne_tactical_archetypes_publication.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Also overwrite the old one if that's what the user prefers, 
    # but I'll keep the publication name as per previous pattern.
    plt.savefig('Desc/plots/tsne_existing_clusters.png', bbox_inches='tight', dpi=300)
    
    print(f"✅ Publication-ready t-SNE saved to {save_path}")

if __name__ == "__main__":
    generate_publication_tsne()
