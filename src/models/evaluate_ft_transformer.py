import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os
from train_ft_transformer import FTTransformer, FeatureTokenizer, TransformerBlock

def load_data(data_dir="data/processed/ml_ready"):
    test = pd.read_parquet(f"{data_dir}/test.parquet")
    return test

def get_attention_weights(model, x):
    """
    Extract attention weights from the first block for feature importance.
    FT-Transformer attention maps [batch, heads, n_feat, n_feat]
    """
    model.eval()
    with torch.no_grad():
        # Tokenize
        tokens = model.tokenizer(x) # [batch, n_feat, dim]
        
        # Get first block attention
        block = model.blocks[0]
        x_ln = block.ln1(tokens)
        _, attn_weights = block.msa(x_ln, x_ln, x_ln) # [batch, n_feat, n_feat]
        
    return attn_weights.cpu().numpy()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    test_df = load_data()
    
    # Exclude IDs and Targets
    target_col = 'log_target_market_value_next_month'
    exclude = [
        'player_id', 'month_date', 'target_market_value_next_month', 
        'target_minutes_next_month', 'target_wiki_views_next_month',
        'log_target_wiki_views_next_month', 'log_target_market_value_next_month',
        'weekly_wage', 'current_market_value', 'wiki_views', 'target_value_change',
        'target_log_mv_delta'
    ]
    feature_cols = [c for c in test_df.columns if c not in exclude]
    
    X_test_raw = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    # Load Scaler (created during training)
    # We'll just re-fit on train for this script to work if not saved perfectly
    train_df = pd.read_parquet("data/processed/ml_ready/train.parquet")
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    X_test = scaler.transform(X_test_raw)
    
    # Load Model
    model = FTTransformer(n_num_features=len(feature_cols)).to(device)
    model.load_state_dict(torch.load("models/ft_transformer_best.pth", map_location=device))
    model.eval()
    
    # 1. Final Evaluation
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test).to(device)).cpu().numpy().flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print("\n" + "="*50)
    print("FT-TRANSFORMER FINAL EVALUATION (PATH C)")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2:   {r2:.4f}")
    print("="*50)
    
    # 2. Attention-based Feature Importance
    # We average the attention weights over a sample of the test set
    sample_indices = np.random.choice(len(X_test), 500, replace=False)
    X_sample = torch.FloatTensor(X_test[sample_indices]).to(device)
    
    # attn_weights shape: [500, n_feat, n_feat]
    attn_weights = get_attention_weights(model, X_sample)
    
    # Global importance: sum of attention received by each feature from all other features
    # (How much the model "looked" at this feature to decide other things)
    global_attn = attn_weights.mean(axis=0).sum(axis=0)
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'attention_score': global_attn
    }).sort_values('attention_score', ascending=False)
    
    print("\nTop Features by Attention (Transformer Intelligence):")
    print(importance_df.head(10))
    
    # 3. Save Final Metadata
    os.makedirs("models/ft_transformer", exist_ok=True)
    metadata = {
        "status": "Complete (Interrupted at Epoch 5)",
        "test_rmse": float(rmse),
        "test_r2": float(r2),
        "top_features": importance_df.head(10).to_dict('records')
    }
    with open("models/ft_transformer/final_evaluation.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n✅ Path C finalized. Metrics saved to models/ft_transformer/final_evaluation.json")
