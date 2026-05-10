import streamlit as st
import joblib
import pandas as pd
import time
import random
import plotly.graph_objects as go

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Heat Exchanger Predictive Maintenance",
    layout="wide"
)

# =========================
# LOAD MODEL + SCALER
# =========================
@st.cache_resource
def load_assets():
    model = joblib.load("xgboost_predictive_maintenance_model.pkl")
    scaler = joblib.load("standard_scaler.pkl")
    return model, scaler

model, scaler = load_assets()

# =========================
# INITIALIZE HISTORY
# =========================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'sim_step' not in st.session_state:
    st.session_state.sim_step = 0

# =========================
# TITLE
# =========================
st.title("🛠️ AI Predictive Maintenance System")
st.subheader("Heat Exchanger Health Monitoring")

# =========================
# SYSTEM MODE SELECTION
# =========================
st.sidebar.title("🏭 Plant Control Room")

mode = st.sidebar.radio(
    "Operation Mode",
    ["Manual Testing", "Live Simulation"]
)

# =====================================
# SIDEBAR LOGIC
# =====================================
if mode == "Manual Testing":
    st.sidebar.header("Input Sensor Parameters")
    
    # Interactive Sliders for Manual Testing
    temp_in_hot = st.sidebar.slider("Hot Inlet Temperature (°C)", 50, 300, 150)
    temp_out_cold = st.sidebar.slider("Cold Outlet Temperature (°C)", 20, 250, 80)
    flow_rate = st.sidebar.slider("Flow Rate", 1.0, 50.0, 10.0)
    pressure_drop = st.sidebar.slider("Pressure Drop", 0.1, 20.0, 3.0)
    operating_hours = st.sidebar.number_input("Operating Hours", 0, 50000, 1000)
    step = 0

else:
    st.sidebar.info("Simulation Mode Active: System is aging in real-time.")
    
    # Increase simulation step
    st.session_state.sim_step += 1
    step = st.session_state.sim_step
    st.sidebar.write(f"**Current Simulation Step:** {step}")

    # Simulation logic (Degradation physics)
    noise = random.uniform(-0.2, 0.2)
    temp_in_hot = 180.0 + noise
    temp_out_cold = min(175.0, 90.0 + (step * 2.0) + noise)
    flow_rate = max(1.0, 20.0 - (step * 0.5) + noise)
    pressure_drop = min(20.0, 1.0 + (step * 0.4) + noise)
    operating_hours = 1000 + (step * 100)

    # --- NEW: Live Telemetry Display (Read-Only Metrics) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Live Telemetry")
    st.sidebar.metric("Hot Inlet Temp", f"{temp_in_hot:.1f} °C")
    st.sidebar.metric("Flow Rate", f"{flow_rate:.2f}")
    st.sidebar.metric("Pressure Drop", f"{pressure_drop:.2f}")
    st.sidebar.metric("Operating Hours", f"{operating_hours} hrs")

    # Maintenance Button
    if st.sidebar.button("🛠️ Perform Maintenance (Reset)"):
        st.session_state.sim_step = 0
        st.session_state.history = []
        st.rerun()

# =========================
# PHYSICS VALIDATION
# =========================
if temp_out_cold >= temp_in_hot:
    st.error("Physics Error: Cold outlet cannot be hotter than hot inlet.")
    st.stop()

# =========================
# FEATURE ENGINEERING
# =========================
delta_T = temp_in_hot - temp_out_cold
flow_resistance = pressure_drop / flow_rate
cooling_efficiency = delta_T / temp_in_hot

input_data = pd.DataFrame([{
    'temp_in_hot': temp_in_hot,
    'flow_rate': flow_rate,
    'pressure_drop': pressure_drop,
    'operating_hours': operating_hours,
    'delta_T': delta_T,
    'flow_resistance': flow_resistance,
    'cooling_efficiency': cooling_efficiency
}])

# =========================
# PREDICT & STORE HISTORY
# =========================
scaled_input = scaler.transform(input_data)
prediction = model.predict(scaled_input)[0]

# Only save history during simulation to keep charts clean
if mode == "Live Simulation":
    st.session_state.history.append({
        "step": step,
        "cooling_efficiency": cooling_efficiency * 100,
        "flow_resistance": flow_resistance
    })
    st.session_state.history = st.session_state.history[-30:]

# =========================
# UI DASHBOARD
# =========================
status_map = {0: "Normal ✅", 1: "Warning ⚠️", 2: "Failure 🚨"}
col_left, col_right = st.columns([1, 1])

# --- LEFT PANEL: Metrics ---
with col_left:
    st.subheader("Equipment Condition")
    if prediction == 0: st.success(status_map[prediction])
    elif prediction == 1: st.warning(status_map[prediction])
    else: st.error(status_map[prediction])

    st.markdown("---")
    st.subheader("Engineering Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Delta T", f"{delta_T:.2f}")
    m2.metric("Resistance", f"{flow_resistance:.4f}")
    m3.metric("Efficiency", f"{cooling_efficiency:.2%}")

# --- RIGHT PANEL: Two Trend Charts ---
with col_right:
    # Logic to only show charts during Live Simulation
    if mode == "Live Simulation":
        if len(st.session_state.history) > 1:
            history_df = pd.DataFrame(st.session_state.history)

            # Chart 1: Efficiency (Teal)
            st.subheader("Efficiency Trend (%)")
            fig_eff = go.Figure()
            fig_eff.add_trace(go.Scatter(
                x=history_df["step"], y=history_df["cooling_efficiency"],
                mode='lines+markers', line=dict(color='#00d1b2', width=3)
            ))
            fig_eff.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20),
                                  xaxis_title="Step", yaxis_title="%")
            st.plotly_chart(fig_eff, width='stretch')

            # Chart 2: Resistance (Red)
            st.subheader("Flow Resistance Trend")
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                x=history_df["step"], y=history_df["flow_resistance"],
                mode='lines+markers', line=dict(color='#ff4b4b', width=3)
            ))
            fig_res.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20),
                                  xaxis_title="Step", yaxis_title="Resistance")
            st.plotly_chart(fig_res, width='stretch')
        else:
            st.info("Simulation initiated. Waiting for data steps...")
    
    else:
        # Visual feedback for Manual Mode
        st.subheader("Manual Analysis")
        st.info("Trend charts are disabled in Manual Mode. Use sliders to simulate specific sensor values and observe the model's prediction on the left.")

st.markdown("---")
st.subheader("Processed Model Input (Current State)")
st.dataframe(input_data)

# =========================
# AUTO-REFRESH
# =========================
if mode == "Live Simulation":
    time.sleep(2)
    st.rerun()