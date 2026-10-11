import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import json
import time

# --- FT-Transformer Components ---

class FeatureTokenizer(nn.Module):
    """Converts individual numerical features into embeddings."""
    def __init__(self, n_num_features, embedding_dim):
        super().__init__()
        # Each numerical feature gets its own linear projection
        self.W_num = nn.Parameter(torch.empty(n_num_features, embedding_dim))
        self.bias = nn.Parameter(torch.empty(n_num_features, embedding_dim))
        nn.init.kaiming_uniform_(self.W_num)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x shape: [batch_size, n_num_features]
        # output shape: [batch_size, n_num_features, embedding_dim]
        return x.unsqueeze(-1) * self.W_num + self.bias

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.msa = nn.MultiheadAttention(embedding_dim, n_heads, dropout=dropout, batch_first=True)
        
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-Head Self-Attention
        x = x + self.dropout(self.msa(self.ln1(x), self.ln1(x), self.ln1(x))[0])
        # Feed-Forward Network
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

class FTTransformer(nn.Module):
    def __init__(self, n_num_features, embedding_dim=128, n_layers=3, n_heads=8, dropout=0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num_features, embedding_dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_final = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # 1. Tokenize
        x = self.tokenizer(x) # [batch, n_feat, dim]
        
        # 2. Transformer Encoder
        for block in self.blocks:
            x = block(x)
            
        # 3. Aggregation (Take the average or the 'CLS' token - here average of feature tokens)
        x = x.mean(dim=1) 
        
        # 4. Final Regression
        x = self.ln_final(x)
        return self.head(x)

# --- Data & Training ---

def load_data(data_dir="data/processed/ml_ready"):
    train = pd.read_parquet(f"{data_dir}/train.parquet")
    val = pd.read_parquet(f"{data_dir}/val.parquet")
    test = pd.read_parquet(f"{data_dir}/test.parquet")
    return train, val, test

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_df, val_df, test_df = load_data()
    
    # Exclude IDs and Targets
    target_col = 'log_target_market_value_next_month'
    exclude = [
        'player_id', 'month_date', 'target_market_value_next_month', 
        'target_minutes_next_month', 'target_wiki_views_next_month',
        'log_target_wiki_views_next_month', 'log_target_market_value_next_month',
        'weekly_wage', 'current_market_value', 'wiki_views', 'target_value_change',
        'target_log_mv_delta'
    ]
    feature_cols = [c for c in train_df.columns if c not in exclude]
    
    X_train_raw = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val_raw = val_df[feature_cols].values
    y_val = val_df[target_col].values
    X_test_raw = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    # Standard Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    
    # Create Dataloaders
    batch_size = 256
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).view(-1, 1))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # Init Model
    model = FTTransformer(n_num_features=len(feature_cols)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    print(f"Training FT-Transformer on {len(feature_cols)} features...")
    
    # Training Loop
    best_val_loss = float('inf')
    patience = 5
    trigger = 0
    
    for epoch in range(3):
        start_time = time.time()
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                total_val_loss += criterion(model(batch_X), batch_y).item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.6f} | Time: {epoch_time:.1f}s")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/ft_transformer_best.pth")
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                print("Early stopping triggered.")
                break
                
    # Final Evaluation
    model.load_state_dict(torch.load("models/ft_transformer_best.pth"))
    model.eval()
    with torch.no_grad():
        final_preds = model(torch.FloatTensor(X_test).to(device)).cpu().numpy().flatten()
        
    rmse = np.sqrt(mean_squared_error(y_test, final_preds))
    r2 = r2_score(y_test, final_preds)
    
    print("\n" + "="*30)
    print("FT-Transformer Final Results:")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2: {r2:.4f}")
    
    # Save Metadata
    os.makedirs("models/ft_transformer", exist_ok=True)
    joblib.dump(scaler, "models/ft_transformer/scaler_ft.joblib")
    
    metadata = {
        "features": feature_cols,
        "test_rmse": float(rmse),
        "test_r2": float(r2),
        "architecture": "FT-Transformer (Custom Implementation)"
    }
    with open("models/ft_transformer/metadata_ft.json", "w") as f:
        json.dump(metadata, f, indent=4)
