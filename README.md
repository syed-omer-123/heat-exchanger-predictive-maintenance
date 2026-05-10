# 🛠️ AI Predictive Maintenance System (Heat Exchanger)

A real-time AI-powered predictive maintenance dashboard for monitoring Heat Exchanger health using Machine Learning, physics-based simulation, and Streamlit.

---

🚀 Live Demo  
https://heat-exchanger-ai-dashboard.streamlit.app/

## 📌 Project Overview
This project simulates an industrial Heat Exchanger system and uses a trained **XGBoost machine learning model** to predict equipment health in real-time.

It acts as a **Digital Twin prototype**, combining:
- **Physics-based degradation simulation**: Realistically models equipment aging.
- **Advanced Feature Engineering**: Calculates Delta T, Flow Resistance, and Efficiency.
- **Machine Learning Inference**: Real-time classification of system health.
- **Interactive Dashboard**: Built for plant operators to monitor trends.

---

## ✨ Features
- 🔴 **Real-time Simulation**: Watch the system degrade step-by-step.
- 🤖 **XGBoost Classifier**: Predicts Normal, Warning, and Failure states.
- 📊 **Trend Visualization**: Live Plotly charts for Efficiency and Resistance.
- 🔄 **Dual Mode**: Switch between "Manual Testing" and "Live Simulation."
- 🛠️ **Maintenance Reset**: Reset the "Digital Twin" to a healthy state.

---

🧠 Machine Learning Pipeline (Feature Engineering + Classification)

### Input Features:
- Hot Inlet Temperature (°C)
- Cold Outlet Temperature (°C)
- Flow Rate
- Pressure Drop
- Operating Hours

### Engineered Features (The Math):
- **Delta T**: Heat transfer differential.
- **Flow Resistance**: Pressure drop relative to flow.
- **Cooling Efficiency**: Normalized heat exchange performance.

---

## 🏗️ Tech Stack
- **Languages**: Python
- **Libraries**: Streamlit, XGBoost, Pandas, Plotly, Scikit-learn, Joblib
- **Deployment**: Streamlit Community Cloud

---

## 📁 Project Structure
```bash
├── app.py
├── requirements.txt
├── README.md
├── xgboost_predictive_maintenance_model.pkl
└── standard_scaler.pkl
```

