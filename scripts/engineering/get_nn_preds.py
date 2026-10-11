import sys, os, json, torch, joblib
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.getcwd(), 'src/models'))
from train_nn import MarketValueNN

def get_nn_preds():
    nn_dir = "models/nn"
    data_dir = "data/processed/ml_ready"
    
    with open(f"{nn_dir}/metadata_nn.json", "r") as f:
        nn_meta = json.load(f)
    
    model = MarketValueNN(len(nn_meta['features']), nn_meta['best_params']['hidden_dims'], nn_meta['best_params']['dropout'])
    model.load_state_dict(torch.load(f"{nn_dir}/market_value_nn.pth", map_location='cpu'))
    model.eval()
    scaler = joblib.load(f"{nn_dir}/scaler_nn.joblib")
    
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    top_players_df = test_df.sort_values('current_market_value', ascending=False).drop_duplicates('player_id').head(12)
    
    X_nn = scaler.transform(top_players_df[nn_meta['features']].values)
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_nn)).cpu().numpy().flatten().tolist()
    
    result = dict(zip(top_players_df['player_id'].tolist(), preds))
    with open("nn_preds_temp.json", "w") as f:
        json.dump(result, f)
    print("✅ NN Predictions saved to nn_preds_temp.json")

if __name__ == "__main__":
    get_nn_preds()
