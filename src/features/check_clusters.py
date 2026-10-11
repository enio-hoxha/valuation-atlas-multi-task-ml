import pandas as pd
import numpy as np

def check_alignment():
    df = pd.read_parquet('data/processed/ml_ready/train.parquet')
    pos_cols = ['pos_Attack', 'pos_Defender', 'pos_Goalkeeper', 'pos_Midfield']
    def get_pos(row):
        for col in pos_cols:
            if row[col] == 1:
                return col.replace('pos_', '')
        return 'Unknown'
    
    df['position'] = df.apply(get_pos, axis=1)
    
    ct = pd.crosstab(df['tactical_cluster'], df['position'])
    print("Cross-tabulation of Existing Clusters (k=8) vs Position:")
    print(ct)
    
    # Calculate percentage of each position in each cluster
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    print("\nCluster Composition (%):")
    print(ct_pct.round(1))

if __name__ == "__main__":
    check_alignment()
