# AI Predictive Maintenance System 🛠️

A real-time AI-powered predictive maintenance dashboard for monitoring Heat Exchanger health using Machine Learning and Streamlit.

## Features

- Real-time equipment degradation simulation
- XGBoost predictive maintenance model
- Live Efficiency Trend visualization
- Flow Resistance monitoring
- Failure prediction (Normal / Warning / Failure)
- Maintenance reset simulation
- Interactive Streamlit dashboard

## Technologies Used

- Python
- Streamlit
- XGBoost
- Plotly
- Pandas
- Scikit-learn

## Project Structure

```bash
├── app.py
├── pipeline.py
├── requirements.txt
├── README.md
├── xgboost_predictive_maintenance_model.pkl
├── standard_scaler.pkl
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## System Behavior

The simulation models Heat Exchanger degradation over time:

- Cooling efficiency decreases
- Flow resistance increases
- AI predicts equipment condition
- Maintenance reset restores healthy operation

## Author

Syed Omer