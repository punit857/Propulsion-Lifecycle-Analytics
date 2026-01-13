import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import time
import plotly.graph_objects as go

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title=" Propulsion Health Management System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #41444C;
    }
    h1, h2, h3 { color: #FAFAFA; font-family: 'Helvetica Neue', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model('rul_model.keras')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# --- 4. SIDEBAR ---
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🔧 Sensor Simulation")

# Sliders with specific ranges for FD001 dataset
s2 = st.sidebar.slider("Sensor 2 (LPC Temp)", 641.0, 645.0, 642.0, help="Healthy: ~642 | Broken: ~644")
s4 = st.sidebar.slider("Sensor 4 (LPT Temp)", 1390.0, 1440.0, 1400.0, help="Healthy: ~1400 | Broken: ~1430")
s11 = st.sidebar.slider("Sensor 11 (HPC Pressure)", 47.0, 48.5, 47.4, help="Healthy: ~47.4 | Broken: ~48.0")
s15 = st.sidebar.slider("Sensor 15 (Bypass Ratio)", 8.3, 8.6, 8.4, help="Healthy: ~8.41 | Broken: ~8.50")



# --- 5. MAIN APP ---
st.title("Turbine Lifecycle Analytics")

st.markdown("---")

if st.button("Predict"):
    
    # --- A. GENERATE BASELINE DATA ---
    features = ['setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]
    
    # Healthy Baseline values
    base_values = {
        'setting_1': 0.0, 'setting_2': 0.0, 'setting_3': 100.0,
        's_1': 518.67, 's_2': 641.82, 's_3': 1589.70, 's_4': 1400.60, 's_5': 14.62,
        's_6': 21.61, 's_7': 554.36, 's_8': 2388.0, 's_9': 9046.0, 's_10': 1.30,
        's_11': 47.47, 's_12': 521.66, 's_13': 2388.0, 's_14': 8133.0, 's_15': 8.4195,
        's_16': 0.03, 's_17': 392.0, 's_18': 2388.0, 's_19': 100.0, 's_20': 38.0, 's_21': 23.0
    }
    
    # Create DataFrame (30 cycles)
    input_df = pd.DataFrame([base_values] * 30)

    # --- B. APPLY SLIDERS + NOISE (The Fix) ---
    # 1. Create the Trend (Straight Line)
    s2_trend = np.linspace(s2, s2 + 0.5, 30)
    s4_trend = np.linspace(s4, s4 + 10, 30) 
    s11_trend = np.linspace(s11, s11 + 0.1, 30) 
    s15_trend = np.linspace(s15, s15 + 0.02, 30)

    # 2. Add Noise ON TOP of the trend (This makes it wiggle)
    # We add significant noise to s_4 because temperature sensors are naturally noisy
    input_df['s_2'] = s2_trend + np.random.normal(0, 0.2, 30)
    input_df['s_4'] = s4_trend + np.random.normal(0, 1.5, 30)  # High noise for LPT Temp
    input_df['s_11'] = s11_trend + np.random.normal(0, 0.05, 30)
    input_df['s_15'] = s15_trend + np.random.normal(0, 0.01, 30)
    
    # 3. Add noise to all other columns so they aren't completely flat
    for col in input_df.columns:
        if col not in ['s_2', 's_4', 's_11', 's_15', 'setting_1', 'setting_2', 'setting_3']:
            input_df[col] = input_df[col] + np.random.normal(0, 0.1, 30)

    # --- C. PREDICTION ---
    if hasattr(scaler, 'feature_names_in_'):
        input_df_sorted = input_df[scaler.feature_names_in_]
    else:
        input_df_sorted = input_df
    
    scaled_data = scaler.transform(input_df_sorted)
    model_input = scaled_data.reshape(1, 30, 24)
    prediction = model.predict(model_input)
    rul = int(prediction[0][0])
    
    # --- D. DISPLAY RESULTS ---
    kpi1, kpi2 = st.columns(2)
    with kpi1:
        st.metric("Predicted RUL", f"{rul} Cycles")
    with kpi2:
        if rul > 120:
            st.success(f"Status: HEALTHY ✅")
        elif rul > 50:
            st.warning(f"Status: WARNING ⚠️")
        else:
            st.error(f"Status: CRITICAL 🚨")

    # --- E. PLOTS (Multi-Sensor) ---
    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Multi-Sensor Real-Time Trends")
        fig = go.Figure()
        
        # Plot 1: LPT Temperature (Red)
        fig.add_trace(go.Scatter(y=input_df['s_4'], mode='lines', name='LPT Temp (s_4)', line=dict(color='#FF4B4B', width=3)))
        
        # Plot 2: HPC Pressure (Cyan) - Scaled for visibility
        fig.add_trace(go.Scatter(y=input_df['s_11'] * 30, mode='lines', name='HPC Pressure (x30)', line=dict(color='#00CC96', width=2, dash='dot')))
        
        # Plot 3: Fan Speed (Orange) - Offset for visibility
        fig.add_trace(go.Scatter(y=input_df['s_12'] + 900, mode='lines', name='Fan Speed (+900)', line=dict(color='#FFA500', width=2)))

        fig.update_layout(height=350, margin=dict(t=20, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Critical Pressure")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = s11,
            title = {'text': "HPC Pressure (psi)"},
            gauge = {
                'axis': {'range': [46.8, 48.2]}, 
                'bar': {'color': "#636EFA"},
                'steps': [{'range': [46.8, 47.6], 'color': "rgba(0, 255, 0, 0.3)"}, {'range': [47.6, 48.2], 'color': "rgba(255, 0, 0, 0.3)"}]
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig_gauge, use_container_width=True)

else:
    st.info(" Set Sensor 4 to ~1400 for Healthy, or ~1430 for Broken.")