import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ------------------------------
# PAGE CONFIGURATION
# ------------------------------
st.set_page_config(
    page_title="QR-CSGICF Maturity Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# CUSTOM CSS FOR BETTER VISUALS
# ------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .domain-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1E3A8A;
    }
    .level-badge {
        display: inline-block;
        background-color: #1E3A8A;
        color: white;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .footer {
        text-align: center;
        font-size: 0.8rem;
        color: #6B7280;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# TITLE
# ------------------------------
st.markdown('<div class="main-header">QR‑CSGICF Maturity & Governance Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Quantum‑Resilient Cyber‑Safety Governance & Consolidated Internal Control Framework</div>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------------
# SIDEBAR – INPUT PARAMETERS (Technical Core Results)
# ------------------------------
st.sidebar.header("📊 Input Parameters (Component 1–3 Results)")

default_violation = 2.13      # % violation rate (Component 1)
default_recall = 95.7         # Anomaly recall % (Component 2)
default_stability = 0.88      # Hybrid stability score (Component 3)

violation_rate = st.sidebar.number_input(
    "Latency Violation Rate (%) – Component 1",
    min_value=0.0, max_value=100.0, value=default_violation, step=0.1,
    help="Percentage of handshakes exceeding 200ms. Lower is better."
)
anomaly_recall = st.sidebar.number_input(
    "Anomaly Detection Recall (%) – Component 2",
    min_value=0.0, max_value=100.0, value=default_recall, step=0.1,
    help="Recall (sensitivity) of the Random Forest classifier. Higher is better."
)
stability_score = st.sidebar.number_input(
    "Hybrid PQC Stability Score – Component 3",
    min_value=0.0, max_value=1.0, value=default_stability, step=0.01,
    help="Stability score from emulation (0=low, 1=high)."
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Literature‑driven maturity criteria**\n\n"
    "Maturity thresholds derived from systematic review (69 studies, 2021–2026):\n"
    "- **AI Governance**: Adabara et al. (2025), Alikhani (2025), Henderson et al. (2022)\n"
    "- **Human Factors**: Pollini et al. (2022), Khadka & Ullah (2025), Nobles et al. (2025)\n"
    "- **Autonomous Assurance**: Fisher et al. (2021), Tallam (2025), Thakur et al. (2026)"
)

# Additional sliders for non‑quantitative domains (to be adjusted based on document analysis)
st.sidebar.markdown("### 📄 Domain Implementation Scores")
st.sidebar.caption("Based on document analysis (case studies / public reports)")

ai_score = st.sidebar.slider(
    "AI Governance Implementation (0–100)",
    min_value=0, max_value=100, value=65, step=5,
    help="Higher = better integration of XAI, ethical controls, regulatory alignment."
)
hf_score = st.sidebar.slider(
    "Human Factors Integration (0–100)",
    min_value=0, max_value=100, value=50, step=5,
    help="Measures embedding of fatigue, cognitive load, behavioural anomalies."
)
assurance_score = st.sidebar.slider(
    "Autonomous Assurance Capability (0–100)",
    min_value=0, max_value=100, value=40, step=5,
    help="Dynamic risk scoring, real‑time validation, predictive auditing."
)

# ------------------------------
# MATURITY SCORING FUNCTIONS (defined before use)
# ------------------------------
def get_domain_description(domain, level):
    """Return a text description for each domain and maturity level."""
    descriptions = {
        "Cybersecurity": {
            1: "No real‑time monitoring; high latency violation; poor anomaly detection.",
            2: "Basic monitoring but lacks predictive intelligence.",
            3: "Integrated rule‑based and ML detection; violation rate <10%.",
            4: "Advanced anomaly detection (recall >95%); adaptive responses.",
            5: "Fully autonomous cyber‑resilience with self‑healing capabilities."
        },
        "Safety (SMS)": {
            1: "Reactive safety management; no digital integration.",
            2: "Digital dashboards but manual analysis.",
            3: "Predictive safety intelligence using stability scores.",
            4: "Real‑time safety telemetry with automated alerts.",
            5: "Autonomous safety‑critical decision support."
        },
        "AI Governance": {
            1: "No AI governance framework; black‑box models.",
            2: "Basic explainability (LIME/SHAP) in limited use.",
            3: "Formal XAI requirements; ethical guidelines adopted.",
            4: "Continuous AI trust scoring; human‑in‑the‑loop.",
            5: "Full lifecycle AI governance with quantum‑resilient audit."
        },
        "Human Factors Intelligence": {
            1: "Human factors ignored in cybersecurity.",
            2: "Awareness training but no dynamic variables.",
            3: "Fatigue/cognitive load monitored via surveys.",
            4: "Real‑time sensor‑based human state tracking.",
            5: "Autonomous adaptation to human cognitive state."
        },
        "Autonomous Assurance": {
            1: "Periodic audits only.",
            2: "Basic runtime monitoring for known failures.",
            3: "Predictive risk scoring with dashboards.",
            4: "Continuous assurance with automated evidence collection.",
            5: "Self‑certifying autonomous assurance certified by regulators."
        },
        "Operational Resilience": {
            1: "Brittle; no contingency plans.",
            2: "Static resilience plans.",
            3: "Adaptive continuity based on stability scores.",
            4: "Real‑time dependency intelligence.",
            5: "Black‑swan ready; self‑learning resilience."
        }
    }
    return descriptions.get(domain, {}).get(level, "No description available.")

def get_cyber_maturity(violation, recall):
    if violation <= 2.5 and recall >= 98:
        return 5
    elif violation <= 5.0 and recall >= 95:
        return 4
    elif violation <= 10.0 and recall >= 90:
        return 3
    elif violation <= 15.0 and recall >= 80:
        return 2
    else:
        return 1

def get_safety_maturity(stability):
    if stability >= 0.95:
        return 5
    elif stability >= 0.88:
        return 4
    elif stability >= 0.80:
        return 3
    elif stability >= 0.70:
        return 2
    else:
        return 1

def get_ai_governance_maturity(score):
    if score >= 90:
        return 5
    elif score >= 75:
        return 4
    elif score >= 60:
        return 3
    elif score >= 40:
        return 2
    else:
        return 1

def get_human_factors_maturity(score):
    if score >= 85:
        return 5
    elif score >= 70:
        return 4
    elif score >= 55:
        return 3
    elif score >= 35:
        return 2
    else:
        return 1

def get_assurance_maturity(score):
    if score >= 90:
        return 5
    elif score >= 75:
        return 4
    elif score >= 60:
        return 3
    elif score >= 40:
        return 2
    else:
        return 1

def get_resilience_maturity(stability):
    if stability >= 0.92:
        return 5
    elif stability >= 0.85:
        return 4
    elif stability >= 0.78:
        return 3
    elif stability >= 0.68:
        return 2
    else:
        return 1

# ------------------------------
# COMPUTE ALL MATURITY LEVELS
# ------------------------------
cyber_level = get_cyber_maturity(violation_rate, anomaly_recall)
safety_level = get_safety_maturity(stability_score)
ai_level = get_ai_governance_maturity(ai_score)
hf_level = get_human_factors_maturity(hf_score)
assurance_level = get_assurance_maturity(assurance_score)
resilience_level = get_resilience_maturity(stability_score)

domains = {
    "Cybersecurity": cyber_level,
    "Safety (SMS)": safety_level,
    "AI Governance": ai_level,
    "Human Factors Intelligence": hf_level,
    "Autonomous Assurance": assurance_level,
    "Operational Resilience": resilience_level
}

# ------------------------------
# MAIN COLUMN – MATURITY CARDS
# ------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Governance Domain Maturity Levels")
    st.markdown("Maturity scale: **1 (Fragmented)** → **5 (Autonomous)**")

    for domain, level in domains.items():
        level_desc = ["Fragmented", "Coordinated", "Integrated", "Adaptive", "Autonomous"][level-1]
        color = ["#DC2626", "#F59E0B", "#3B82F6", "#10B981", "#8B5CF6"][level-1]
        description = get_domain_description(domain, level)
        st.markdown(f"""
        <div class="domain-card" style="border-left-color: {color};">
            <strong>{domain}</strong>
            <span class="level-badge" style="background-color: {color};">Level {level} – {level_desc}</span>
            <p style="margin-top: 0.5rem; font-size: 0.9rem;">{description}</p>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------
# RADAR CHART
# ------------------------------
with col2:
    st.subheader("🕸️ Governance Radar")
    categories = list(domains.keys())
    values = list(domains.values())

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        marker=dict(color='#1E3A8A', size=8),
        line=dict(color='#1E3A8A', width=2),
        name="Current Maturity"
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[1, 5],
                tickvals=[1,2,3,4,5],
                ticktext=["1", "2", "3", "4", "5"]
            )
        ),
        showlegend=False,
        height=450,
        margin=dict(l=50, r=50, t=30, b=30)
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# OVERALL MATURITY SCORE
# ------------------------------
overall = sum(values) / len(values)
overall_level = round(overall)
overall_desc = ["Fragmented", "Coordinated", "Integrated", "Adaptive", "Autonomous"][overall_level-1]

st.markdown("---")
col_a, col_b, col_c = st.columns([1,2,1])
with col_b:
    st.markdown(f"""
    <div style="text-align: center; background-color: #EFF6FF; border-radius: 20px; padding: 1.5rem;">
        <h2 style="margin:0;">Overall QR‑CSGICF Maturity</h2>
        <h1 style="font-size: 4rem; margin:0; color:#1E3A8A;">Level {overall_level}</h1>
        <h3 style="margin:0;">{overall_desc}</h3>
        <p style="margin-top: 0.5rem;">Based on {len(categories)} governance domains</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------
# FOOTER WITH LITERATURE REFERENCES
# ------------------------------
st.markdown("---")
st.markdown("""
<div class="footer">
    <strong>Evidence base:</strong> Systematic literature review (69 studies, 2021–2026) |
    <strong>Technical validation:</strong> Component 1 (latency 46ms, violation 2.13%),
    Component 2 (anomaly recall 95.7%), Component 3 (hybrid stability 0.88)<br>
    Maturity criteria derived from Adabara (2025), Alikhani (2025), Pollini (2022), Fisher (2021), Tallam (2025), and 60+ other peer‑reviewed sources.
</div>
""", unsafe_allow_html=True)