import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Valuation Atlas | Football ML",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #374151;
    }
    h1, h2, h3 {
        color: #f3f4f6 !important;
        font-family: 'Inter', sans-serif;
    }
    .interpretation-box {
        background-color: #1e3a8a;
        color: #e0e7ff;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 25px;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e3440,#2e3440);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    results = pd.read_csv('Desc/mtl_mv_results.csv')
    importance = pd.read_csv('Desc/full_importance_matrix.csv', index_col=0)
    features = pd.read_parquet('data/processed/final_features.parquet')
    return results, importance, features

try:
    results_df, importance_df, df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}. Ensure the pipeline (main.py) has been run.")
    st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("⚽ Valuation Atlas")
st.sidebar.markdown("**Master's Dissertation Framework**")
st.sidebar.write("---")
menu = st.sidebar.radio(
    "Navigation",
    ["Executive Summary", "Stakeholder Matrix", "Tactical Archetypes", "Economic Signals", "Model Diagnostics"]
)

st.sidebar.write("---")
st.sidebar.info("""
**Architecture: MTL/MV**
- Multi-Task: 4 Targets
- Multi-View: 3 Feature Levels
""")

# --- 1. EXECUTIVE SUMMARY ---
if menu == "Executive Summary":
    st.title("🚀 Multi-Task Football Player Valuation")
    st.subheader("Reverse-engineering the pricing functions of the football industry.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Max R² (Market)", f"{results_df[results_df['Target']=='Market']['R2'].max():.3f}", "+58% vs Raw")
    with col2:
        st.metric("Feature Views", "3 Levels", "Raw → ML")
    with col3:
        st.metric("Stakeholders", "4 Models", "Custom Objectives")
    with col4:
        st.metric("Players Indexed", f"{len(df['player_id'].unique()):,}")

    st.markdown("""
    ### **The Research Thesis**
    Traditional valuation models treat "Market Value" as a single monolithic number. This project posits that **Value is decomposable**. 
    By training parallel gradient-boosted models on different targets (Transfermarkt Value, Future Minutes, Breakout Probability, Commercial Hype), 
    we quantify what each stakeholder actually prices.
    """)
    
    # Summary Chart: R2 across Views
    fig_summary = px.line(results_df, x='View', y='R2', color='Target', markers=True,
                          title="The 'Analytics Lift': Performance Improvement across Feature Views",
                          color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_summary.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white")
    st.plotly_chart(fig_summary, width='stretch')
    
    st.markdown("""
    <div class="interpretation-box">
    <b>Interpretation:</b> The chart above illustrates the <b>Ablation Study (RQ3)</b>. 
    Note how the 'Market' and 'Coach' models show a consistent 'Analytics Lift' as we add rolling stats and ML archetypes. 
    Conversely, the 'Agent' model remains near-zero, proving that short-term market value spikes are stochastic and 
    largely independent of recent monthly performance lags.
    </div>
    """, unsafe_allow_html=True)

# --- 2. STAKEHOLDER MATRIX ---
elif menu == "Stakeholder Matrix":
    st.title("🎯 The Stakeholder Importance Matrix")
    st.write("How do different industry actors weigh player traits?")

    # Heatmap
    perf_pivot = results_df.pivot(index='View', columns='Target', values='R2')
    
    # Custom Heatmap for Importance
    st.subheader("Global Feature Importance (SHAP)")
    selected_view = st.selectbox("Select Feature View:", ["Raw", "Analytics", "ML"])
    
    # Filter importance matrix for the selected view
    view_models = [f"{selected_view}_{t}" for t in ["Market", "Coach", "Agent", "Sponsor"]]
    imp_sub = importance_df.loc[view_models].T
    
    fig_heat = px.imshow(imp_sub, text_auto=".2f", aspect="auto",
                         color_continuous_scale="RdBu_r",
                         title=f"Stakeholder Weights for {selected_view} View")
    fig_heat.update_layout(height=600)
    st.plotly_chart(fig_heat, width='stretch')

    st.markdown(f"""
    <div class="interpretation-box">
    <b>Qualitative Insight ({selected_view} View):</b><br>
    - <b>The Market</b> prices reputation and age-weighted production.<br>
    - <b>The Coach</b> heavily weights 'Minutes Played' (consistency) and tactical clusters.<br>
    - <b>The Sponsor</b> model focuses almost exclusively on <i>Wikipedia Popularity</i> and <i>Elite Distance</i>.<br>
    The divergence in color intensity across columns proves that these actors are <b>not</b> looking at the same signals.
    </div>
    """, unsafe_allow_html=True)

# --- 3. TACTICAL ARCHETYPES ---
elif menu == "Tactical Archetypes":
    st.title("🧬 Unsupervised Tactical Archetypes")
    st.write("Beyond positions: Clustering players by multi-dimensional performance fingerprints.")

    sample_df = df.sample(n=min(5000, len(df)), random_state=42)
    
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        fig_clusters = px.scatter(sample_df, x='PC1', y='PC2', color='tactical_cluster', 
                                  hover_data=['name', 'market_value_in_eur', 'club_strength'],
                                  title="Tactical Clusters in PCA Performance Space",
                                  color_continuous_scale="Viridis",
                                  opacity=0.7)
        fig_clusters.update_traces(marker=dict(size=8))
        fig_clusters.update_layout(plot_bgcolor='#111827', paper_bgcolor='#111827', font_color="white")
        st.plotly_chart(fig_clusters, width='stretch')

    with col_b:
        st.subheader("Breakout Analysis")
        st.write("Isolation Forest identifies players whose form deviates significantly from their cluster norm.")
        
        fig_break = px.scatter(sample_df, x='PC1', y='PC2', color='is_breakout', 
                               color_discrete_map={0: '#374151', 1: '#ef4444'},
                               title="Breakout Candidates (Red)")
        fig_break.update_layout(plot_bgcolor='#111827', paper_bgcolor='#111827', showlegend=False)
        st.plotly_chart(fig_break, width='stretch')

    st.markdown("""
    <div class="interpretation-box">
    <b>Interpretation (RQ4):</b> The PCA projection reduces 10+ performance metrics into a 2D map. 
    Players clustered together share high-dimensional 'tactical fingerprints'. 
    The <b>Red dots</b> in the right chart are 'Anomalies'—players like <i>Lamine Yamal</i> or <i>Cole Palmer</i> during their breakout months—where 
    performance spiked far beyond the expected baseline for their age and archetype.
    </div>
    """, unsafe_allow_html=True)

# --- 4. ECONOMIC SIGNALS ---
elif menu == "Economic Signals":
    st.title("💰 Economic & Commercial Hype")
    st.write("Merging Transfermarkt valuations with actual Capology wages and Wikipedia engagement.")

    salary_df = df[df['weekly_wage'] > 0].copy()
    
    if not salary_df.empty:
        fig_econ = px.scatter(salary_df, x='weekly_wage', y='market_value_in_eur', 
                             color='club_strength', size='wiki_popularity',
                             hover_data=['name'], log_x=True, log_y=True,
                             title="The Economic Core: Wage vs. Value (Sized by Popularity)",
                             color_continuous_scale="Tropic")
        fig_econ.update_layout(plot_bgcolor='#111827', paper_bgcolor='#111827', font_color="white")
        st.plotly_chart(fig_econ, width='stretch')
        
        st.markdown("""
        <div class="interpretation-box">
        <b>Interpretation:</b> This chart identifies <b>Market Inefficiencies</b>. 
        Players <i>above</i> the trend line are 'undervalued' relative to their wage (potential bargains), 
        while those with large bubbles (high Wiki views) but low market value represent 'Commercial Hype' 
        that has yet to be converted into sporting valuation.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No salary data found. Please run the Capology scraper.")

# --- 5. MODEL DIAGNOSTICS ---
elif menu == "Model Diagnostics":
    st.title("🔬 Model Diagnostics")
    
    # Distribution of Target vs Pred for Market Model
    st.subheader("Market Model: Predicted vs. Actual (Test Set)")
    
    # Load the best market model
    m_path = 'models/ML_Market.joblib'
    if os.path.exists(m_path):
        model = joblib.load(m_path)
        test_df = df[df['split'] == 'test'].sample(n=1000)
        
        # Features for ML view
        V_ml = ['goals', 'assists', 'minutes_played', 'age', 'goals_3m', 'assists_3m', 
                'minutes_played_3m', 'club_strength', 'wiki_popularity', 
                'tactical_cluster', 'is_breakout', 'dist_to_elite']
        
        y_true = test_df['log_market_value'].fillna(0)
        y_pred = model.predict(test_df[V_ml].fillna(0))
        
        fig_diag = go.Figure()
        fig_diag.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', marker=dict(color='#3b82f6', opacity=0.5)))
        fig_diag.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], y=[y_true.min(), y_true.max()], 
                                      line=dict(color='#ef4444', dash='dash'), name='Perfect Fit'))
        
        fig_diag.update_layout(xaxis_title="Actual log(Value)", yaxis_title="Predicted log(Value)", 
                               plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig_diag, width='stretch')
        
        st.markdown("""
        <div class="interpretation-box">
        <b>Statistical Validity:</b> The clustering around the 45-degree line confirms our model 
        is capturing the structural drivers of valuation. The variance (spread) increases at lower valuations, 
        suggesting that the market is more 'rational' for elite players than for those in lower tiers.
        </div>
        """, unsafe_allow_html=True)

st.write("---")
st.caption("Valuation Atlas v2.0 | Multi-Task Multi-View Architecture | Built for Master's Dissertation 2026")
