"""Streamlit UI. Run (API must be running):  streamlit run app.py"""
import json, os, joblib, requests, pandas as pd, streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
metrics = json.load(open("metrics.json"))                        # real numbers saved by q3_automl.py
csv = "adult.csv" if os.path.exists("adult.csv") else "../ASSIGNMENT03/adult.csv"
df = pd.read_csv(csv).replace("?", pd.NA)                        # used only to build the input widgets

st.title("Adult Income Prediction")
st.sidebar.metric(f"Test accuracy ({metrics['model']})", f"{metrics['accuracy']:.2%}")
st.sidebar.subheader("Top 5 features")
for k, v in metrics["top5"].items():
    st.sidebar.write(f"**{k}**: {v}")

inputs = {}
for col in df.columns.drop("income"):
    if df[col].dtype.kind in "if":
        inputs[col] = int(st.number_input(col, int(df[col].min()), int(df[col].max()), int(df[col].median())))
    else:
        inputs[col] = st.selectbox(col, sorted(df[col].dropna().unique()))

if st.button("Predict"):
    try:
        st.success(f"Predicted income: {requests.post(API_URL, json=inputs, timeout=10).json()['prediction']}")
    except Exception:                                            # fallback so a hosted demo still works
        m = joblib.load("best_adult_income_model.joblib")
        x = pd.DataFrame([inputs]); x["net.capital"] = x["capital.gain"] - x["capital.loss"]
        st.warning("API unreachable - used local model")
        st.success(f"Predicted income: {'>50K' if m.predict(x)[0] == 1 else '<=50K'}")
