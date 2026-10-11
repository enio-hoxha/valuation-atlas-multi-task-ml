import os
import subprocess
import sys

def run_step(description, script_path):
    print(f"\n>>> {description}...")
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Step failed: {description}")
        sys.exit(1)

def main():
    # 1. Data Engineering
    run_step("Building Fuzzy Crosswalks", "src/data_engineering/fuzzy_matcher.py")
    run_step("Merging Enriched Dataset", "src/data_engineering/merge_enriched.py")
    
    # 2. Features
    run_step("Engineering Base Features (Lags)", "src/features/engineer.py")
    run_step("Leakage-Free Unsupervised ML", "src/features/unsupervised.py")
    
    # 3. Models
    run_step("Training 3x4 MTL/MV Architecture", "src/models/train.py")
    
    print("\n✅ Pipeline Complete! Run the dashboard with: streamlit run src/visualization/app.py")

if __name__ == "__main__":
    main()
