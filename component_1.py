import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="PQC Aviation Diagnostic Suite", layout="wide")

# --- CUSTOM THEMING ---
st.markdown("""
    <style>
    .main { background-color: #0d1117; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

def run_diagnostic_monitor():
    st.title("📊 Component 1: Robust PQC Diagnostic & Oversight Suite")
    st.info("Unified Monitoring: Integrating Statistical Reliability, Forensic Entropy, and Aviation Governance.")

    # 1. LOAD AND ENHANCE DATA
    @st.cache_data
    def load_full_dataset():
        try:
            df = pd.read_csv("ml_features_and_labels.csv")
            
            # Scenario Mapping
            try:
                manifest = pd.read_csv("scenario_manifest.csv")
                df['scenario_type'] = np.where(df['e5_entropy_c'] > 7.2, 'Adversarial Fuzzing', 'Standard Operation')
            except:
                df['scenario_type'] = 'Standard Operation'

            # Standardizing terminology for aviation research
            df['LATENCY_MS'] = df['e6b_flow_duration_ms']
            df['PAYLOAD_SIZE'] = df['e2_client_record_len']
            df['ENTROPY'] = df['e5_entropy_c']
            
            # Robust Jitter Calculation (Rolling Mean Difference)
            df['JITTER'] = df.groupby(np.arange(len(df))//10)['LATENCY_MS'].transform(lambda x: x.diff().abs().mean()).fillna(0)
            
            # Statistical Outlier Detection (Z-Score)
            df['Z_SCORE'] = np.abs(stats.zscore(df['LATENCY_MS']))
            df['IS_OUTLIER'] = df['Z_SCORE'] > 3
            
            # Identify PQC Algorithm Suites
            alg_cols = [c for c in df.columns if 'e1_alg_suite' in c]
            def get_alg(row):
                for col in alg_cols:
                    if row[col] == 1: return col.split('_')[-1].upper()
                return "HYBRID-RSA"
            df['ALGORITHM'] = df.apply(get_alg, axis=1)
            
            return df
        except Exception as e:
            st.error(f"Critical error loading dataset: {e}")
            return None

    df = load_full_dataset()
    if df is None: return

    # 2. SIDEBAR GOVERNANCE PANEL
    with st.sidebar:
        st.header("🛡️ Governance Controls")
        safety_limit = st.slider("Max Permissible Latency (ms)", 50, 1000, 200)
        st.markdown("---")
        selected_algs = st.multiselect("Algorithm Focus", options=df['ALGORITHM'].unique(), default=df['ALGORITHM'].unique())
        selected_scenarios = st.multiselect("Traffic Context", options=df['scenario_type'].unique(), default=df['scenario_type'].unique())

    # Apply Filters
    mask = (df['ALGORITHM'].isin(selected_algs)) & (df['scenario_type'].isin(selected_scenarios))
    f_df = df[mask].copy()

    # 3. HIGH-LEVEL KPI LAYER (Aviation Safety Board)
    f_df['is_violation'] = f_df['LATENCY_MS'] > safety_limit
    v_rate = f_df['is_violation'].mean() * 100
    p95 = np.percentile(f_df['LATENCY_MS'], 95)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Observed Sessions", f"{len(f_df):,}")
    col2.metric("Mean Latency", f"{f_df['LATENCY_MS'].mean():.2f} ms")
    col3.metric("Violation Rate", f"{v_rate:.2f}%", delta=f"{v_rate-5.0:.1f}% vs Goal", delta_color="inverse")
    col4.metric("95th Percentile (P95)", f"{p95:.1f} ms")

    st.markdown("---")

    # 4. ADVANCED VISUAL ANALYTICS
    tab_dist, tab_cdf, tab_alg, tab_forensics = st.tabs([
        "📉 Variance Analysis", 
        "📈 CDF Reliability", 
        "🧬 Algorithm Benchmarking", 
        "📊 Forensic Entropy"
    ])

    with tab_dist:
        st.subheader("Statistical Variance Analysis")
        fig_hist = px.histogram(f_df, x='LATENCY_MS', color='is_violation', nbins=100, marginal="box",
                                color_discrete_map={True: '#ef4444', False: '#10b981'},
                                title="Latency Distribution with Safety Boundary")
        fig_hist.add_vline(x=safety_limit, line_dash="dash", line_color="yellow", annotation_text="Safety Limit")
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab_cdf:
        st.subheader("Cumulative Distribution Function (Reliability Analysis)")
        sorted_latency = np.sort(f_df['LATENCY_MS'])
        y_vals = np.arange(len(sorted_latency)) / float(len(sorted_latency) - 1)
        fig_cdf = go.Figure()
        fig_cdf.add_trace(go.Scatter(x=sorted_latency, y=y_vals, name='Reliability Curve', line=dict(color='#636EFA', width=3)))
        fig_cdf.add_vline(x=safety_limit, line_dash="dot", line_color="red", annotation_text="Safety Deadline")
        fig_cdf.update_layout(xaxis_title="Latency (ms)", yaxis_title="Probability of Success", template="plotly_dark")
        st.plotly_chart(fig_cdf, use_container_width=True)

    with tab_alg:
        st.subheader("Cross-Algorithm Latency Jitter")
        fig_box = px.box(f_df, x='ALGORITHM', y='LATENCY_MS', color='ALGORITHM', points="outliers", title="Latency Jitter per NIST Suite")
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Summary Table
        stats_df = f_df.groupby('ALGORITHM').agg({'LATENCY_MS': ['mean', 'std'], 'is_violation': 'mean'})
        stats_df.columns = ['Avg Latency', 'Jitter (StdDev)', 'Violation Rate (%)']
        st.dataframe(stats_df.style.format("{:.2f}").background_gradient(cmap='Reds', subset=['Violation Rate (%)']))

    with tab_forensics:
        st.subheader("Ciphertext Entropy vs. Performance Correlation")
        fig_scatter = px.scatter(f_df, x='ENTROPY', y='LATENCY_MS', color='is_violation', size='PAYLOAD_SIZE',
                                hover_data=['ALGORITHM', 'scenario_type'], opacity=0.6,
                                color_discrete_map={True: '#ef4444', False: '#10b981'})
        st.plotly_chart(fig_scatter, use_container_width=True)

    # 5. GOVERNANCE VERDICT ENGINE
    st.markdown("---")
    st.subheader("📜 Comprehensive Governance Verdict")
    if v_rate > 10:
        st.error(f"VERDICT: CRITICAL FAILURE. Violation rate ({v_rate:.1f}%) exceeds the 10% safety tolerance. Infrastructure upgrade required.")
    elif p95 > safety_limit:
        st.warning(f"VERDICT: MARGINAL. Mean latency is acceptable, but P95 ({p95:.1f}ms) indicates unsafe peak jitter for flight-critical systems.")
    else:
        st.success("VERDICT: OPERATIONAL. PQC stability markers are within nominal aviation safety standards.")

    st.download_button("Export Evidence (CSV)", f_df.to_csv(index=False), "pqc_diagnostic_data.csv", "text/csv")

if __name__ == "__main__":
    run_diagnostic_monitor()