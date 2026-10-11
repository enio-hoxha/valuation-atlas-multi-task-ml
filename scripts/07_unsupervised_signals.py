import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.kernel_approximation import Nystroem
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import joblib
import os

def run_unsupervised_pipeline(data_dir="data/processed/ml_ready"):
    print(f"Loading split data from {data_dir}...")
    train = pd.read_parquet(f"{data_dir}/train.parquet")
    val = pd.read_parquet(f"{data_dir}/val.parquet")
    test = pd.read_parquet(f"{data_dir}/test.parquet")
    
    # Define features for clustering (using columns present in ml_ready)
    feature_cols = [
        'goals_p90_roll_3m', 'assists_p90_roll_3m', 'minutes_roll_3m', 
        'age_at_obs', 'club_elo'
    ]
    
    # 1. Scaling (Fit ONLY on Train)
    print("Fitting Scaler for Unsupervised ML on Train...")
    scaler = StandardScaler()
    scaler.fit(train[feature_cols].fillna(0))
    
    def transform_df(df):
        X = df[feature_cols].fillna(0)
        X_scaled = scaler.transform(X)
        return X_scaled

    X_train_scaled = transform_df(train)
    X_val_scaled = transform_df(val)
    X_test_scaled = transform_df(test)
    
    # 2. Kernel K-Means (Archetypes) via Nystroem Approximation
    print("Fitting Kernel K-Means (Nystroem + KMeans, 8 Archetypes) on Train...")
    # Nystroem approximates the RBF kernel mapping for scalability
    nystroem = Nystroem(kernel='rbf', gamma=None, n_components=100, random_state=42)
    X_train_kernel = nystroem.fit_transform(X_train_scaled)
    X_val_kernel = nystroem.transform(X_val_scaled)
    X_test_kernel = nystroem.transform(X_test_scaled)

    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    kmeans.fit(X_train_kernel)
    
    train['tactical_cluster'] = kmeans.predict(X_train_kernel)
    val['tactical_cluster'] = kmeans.predict(X_val_kernel)
    test['tactical_cluster'] = kmeans.predict(X_test_kernel)
    
    # 3. Isolation Forest (Breakout Detection)
    print("Fitting Isolation Forest on Train...")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X_train_scaled)
    
    train['is_breakout'] = (iso_forest.predict(X_train_scaled) == -1).astype(int)
    val['is_breakout'] = (iso_forest.predict(X_val_scaled) == -1).astype(int)
    test['is_breakout'] = (iso_forest.predict(X_test_scaled) == -1).astype(int)
    
    # 4. Elite Distance
    print("Calculating Elite Distance...")
    # Define Elite as top 5% by performance in training
    perf_score = train['minutes_roll_3m'] * (1 + train['goals_p90_roll_3m'] + train['assists_p90_roll_3m'])
    elite_mask = perf_score >= perf_score.quantile(0.95)
    elite_centroid = X_train_scaled[elite_mask].mean(axis=0)
    
    for df, x_scaled in zip([train, val, test], [X_train_scaled, X_val_scaled, X_test_scaled]):
        df['dist_to_elite'] = np.linalg.norm(x_scaled - elite_centroid, axis=1)

    # 5. PCA for Visualization
    pca = PCA(n_components=2)
    pca.fit(X_train_scaled)
    
    for df, x_scaled in zip([train, val, test], [X_train_scaled, X_val_scaled, X_test_scaled]):
        X_pca = pca.transform(x_scaled)
        df['PC1'] = X_pca[:, 0]
        df['PC2'] = X_pca[:, 1]
    
    # Save updated parquets
    print(f"Saving updated features with archetypes to {data_dir}...")
    train.to_parquet(f"{data_dir}/train.parquet", index=False)
    val.to_parquet(f"{data_dir}/val.parquet", index=False)
    test.to_parquet(f"{data_dir}/test.parquet", index=False)
    
    # Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, 'models/scaler_unsupervised.joblib')
    joblib.dump(nystroem, 'models/nystroem.joblib')
    joblib.dump(kmeans, 'models/kmeans.joblib')
    joblib.dump(iso_forest, 'models/iso_forest.joblib')
    joblib.dump(pca, 'models/pca.joblib')
    print("✅ Unsupervised pipeline complete with Kernel K-Means.")

if __name__ == "__main__":
    run_unsupervised_pipeline()

