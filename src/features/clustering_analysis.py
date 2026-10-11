import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(data_path='data/processed/ml_ready/train.parquet'):
    df = pd.read_parquet(data_path)
    # Define features for clustering
    feature_cols = [
        'goals_p90_roll_3m', 'assists_p90_roll_3m', 'minutes_roll_3m', 
        'age_at_obs', 'club_elo'
    ]
    
    # Reconstruct position for ground truth ARI
    pos_cols = ['pos_Attack', 'pos_Defender', 'pos_Goalkeeper', 'pos_Midfield']
    def get_pos(row):
        for col in pos_cols:
            if row[col] == 1:
                return col.replace('pos_', '')
        return 'Unknown'
    
    df['position'] = df.apply(get_pos, axis=1)
    
    return df, feature_cols

def perform_analysis():
    df, feature_cols = load_data()
    X = df[feature_cols].fillna(0)
    
    # Load or fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Existing clusters
    y_existing = df['tactical_cluster']
    
    print("Calculating metrics for existing clustering (k=8)...")
    sil_existing = silhouette_score(X_scaled, y_existing, sample_size=10000, random_state=42)
    ari_existing = adjusted_rand_score(df['position'], y_existing)
    
    print(f"Existing Silhouette Score: {sil_existing:.4f}")
    print(f"Existing ARI (vs Position): {ari_existing:.4f}")
    
    # t-SNE (Subsampled for performance)
    print("Performing t-SNE on 10,000 samples...")
    df_sample = df.sample(n=min(10000, len(df)), random_state=42).copy()
    X_sample_scaled = StandardScaler().fit_transform(df_sample[feature_cols].fillna(0))
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_sample_scaled)
    
    df_sample['tsne_1'] = X_tsne[:, 0]
    df_sample['tsne_2'] = X_tsne[:, 1]
    
    # Visualization
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df_sample, x='tsne_1', y='tsne_2', hue='tactical_cluster', palette='viridis', alpha=0.6)
    plt.title('t-SNE Visualization of Existing tactical_cluster (Subsampled n=10,000)')
    plt.savefig('Desc/plots/tsne_existing_clusters.png')
    plt.close()
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df_sample, x='tsne_1', y='tsne_2', hue='position', alpha=0.6)
    plt.title('t-SNE Visualization colored by Player Position (Subsampled n=10,000)')
    plt.savefig('Desc/plots/tsne_positions.png')
    plt.close()
    
    # Search for better k
    print("Evaluating different values of k...")
    k_range = range(4, 13)
    sil_scores = []
    ari_scores = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels, sample_size=10000, random_state=42)
        ari = adjusted_rand_score(df['position'], labels)
        sil_scores.append(sil)
        ari_scores.append(ari)
        print(f"k={k}: Silhouette={sil:.4f}, ARI={ari:.4f}")
        
    # Plot k-evaluation
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, sil_scores, marker='o')
    plt.title('Silhouette Score vs k')
    plt.subplot(1, 2, 2)
    plt.plot(k_range, ari_scores, marker='o')
    plt.title('ARI vs k')
    plt.tight_layout()
    plt.savefig('Desc/plots/k_evaluation.png')
    plt.close()

    # Refined Clustering: Dropping age and club_elo
    print("Evaluating refined feature set (dropping age and club_elo)...")
    refined_features = ['goals_p90_roll_3m', 'assists_p90_roll_3m', 'minutes_roll_3m']
    X_refined = df[refined_features].fillna(0)
    X_refined_scaled = StandardScaler().fit_transform(X_refined)
    
    km_refined = KMeans(n_clusters=8, random_state=42, n_init=10)
    y_refined = km_refined.fit_predict(X_refined_scaled)
    sil_refined = silhouette_score(X_refined_scaled, y_refined, sample_size=10000, random_state=42)
    ari_refined = adjusted_rand_score(df['position'], y_refined)
    print(f"Refined Features (k=8) Silhouette: {sil_refined:.4f}, ARI: {ari_refined:.4f}")
    
    # GMM
    from sklearn.mixture import GaussianMixture
    print("Evaluating GMM (k=8) on original features...")
    gmm = GaussianMixture(n_components=8, random_state=42)
    y_gmm = gmm.fit_predict(X_scaled)
    sil_gmm = silhouette_score(X_scaled, y_gmm, sample_size=10000, random_state=42)
    ari_gmm = adjusted_rand_score(df['position'], y_gmm)
    print(f"GMM (k=8) Silhouette: {sil_gmm:.4f}, ARI: {ari_gmm:.4f}")
    
    # Save results to a report
    with open('Desc/clustering_report.md', 'w') as f:
        f.write("# Clustering Analysis Report\n\n")
        f.write(f"## Existing Clustering (k=8)\n")
        f.write(f"- Features: {feature_cols}\n")
        f.write(f"- Silhouette Score: {sil_existing:.4f}\n")
        f.write(f"- ARI (vs Position): {ari_existing:.4f}\n\n")
        
        f.write(f"## Refined Clustering (k=8)\n")
        f.write(f"- Features: {refined_features}\n")
        f.write(f"- Silhouette Score: {sil_refined:.4f}\n")
        f.write(f"- ARI (vs Position): {ari_refined:.4f}\n\n")
        
        f.write(f"## GMM Clustering (k=8)\n")
        f.write(f"- Features: {feature_cols}\n")
        f.write(f"- Silhouette Score: {sil_gmm:.4f}\n")
        f.write(f"- ARI (vs Position): {ari_gmm:.4f}\n\n")
        
        f.write(f"## k-Evaluation (Original Features)\n")
        f.write("| k | Silhouette Score | ARI (vs Position) |\n")
        f.write("|---|------------------|-------------------|\n")
        for i, k in enumerate(k_range):
            f.write(f"| {k} | {sil_scores[i]:.4f} | {ari_scores[i]:.4f} |\n")
        f.write(f"\n## t-SNE Visualization\n")
        f.write("Plots saved to `Desc/plots/`:\n")
        f.write("- `tsne_existing_clusters.png`: Existing clustering in t-SNE space.\n")
        f.write("- `tsne_positions.png`: Player positions in t-SNE space.\n")
        f.write("- `k_evaluation.png`: Silhouette and ARI across different k.\n")


if __name__ == "__main__":
    os.makedirs('Desc/plots', exist_ok=True)
    perform_analysis()
