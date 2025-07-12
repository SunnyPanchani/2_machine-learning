import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and threshold
model = joblib.load("models/best_bank_marketing_model.pkl")
threshold = joblib.load("models/best_threshold.pkl")  # value like 0.65

# App configuration
st.set_page_config(page_title="💼 Bank Term Deposit Predictor", layout="centered")
st.title("💼 Bank Marketing Subscription Predictor")
st.markdown("🔍 Predict whether clients will subscribe to a term deposit based on their profile.")

# --- Sample CSV Download ---
sample_data = pd.DataFrame([{
    'age': 35,
    'campaign': 2,
    'pdays': 999,
    'previous': 0,
    'cons.price.idx': 93.994,
    'cons.conf.idx': -36.4,
    'euribor3m': 4.857,
    'was_contacted': 1,
    'job': 'services',
    'marital': 'married',
    'education': 'high.school',
    'housing': 'no',
    'contact': 'telephone',
    'month': 'may',
    'day_of_week': 'mon',
    'poutcome': 'nonexistent'
}])
csv_bytes = sample_data.to_csv(index=False).encode()

with st.expander("📥 CSV Upload (Batch Prediction)"):
    st.download_button("📄 Download Sample CSV", data=csv_bytes, file_name="sample_input.csv", mime="text/csv")

    uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"])
    if uploaded_file:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("✅ Uploaded Data Preview", df_uploaded.head())

        try:
            proba = model.predict_proba(df_uploaded)[:, 1]
            preds = (proba >= threshold).astype(int)

            df_uploaded["Subscription Probability"] = np.round(proba * 100, 2)
            df_uploaded["Prediction"] = np.where(preds == 1, "✅ Subscribed", "❌ Not Subscribed")

            st.success("🎯 Predictions Completed!")
            st.dataframe(df_uploaded)

            result_csv = df_uploaded.to_csv(index=False).encode()
            st.download_button("⬇️ Download Predictions CSV", data=result_csv, file_name="predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# --- Single Input UI ---
st.header("🧍 Predict for One Client")
with st.form("single_form"):
    age = st.number_input("Age", 18, 100, 35)
    campaign = st.number_input("Number of Contacts During Campaign", 1, 100, 2)
    pdays = st.number_input("Days Since Last Contact (999 = never)", 0, 10000, 999)
    previous = st.number_input("Previous Contacts", 0, 50, 0)
    cons_price_idx = st.number_input("Consumer Price Index", value=93.994)
    cons_conf_idx = st.number_input("Consumer Confidence Index", value=-36.4)
    euribor3m = st.number_input("Euribor 3M", value=4.857)
    was_contacted = st.selectbox("Was Contacted Before?", ["yes", "no"])

    job = st.selectbox("Job", ['admin.', 'blue-collar', 'technician', 'services', 'management',
                                'retired', 'entrepreneur', 'self-employed', 'housemaid',
                                'unemployed', 'student', 'unknown'])
    marital = st.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'])
    education = st.selectbox("Education", ['university.degree', 'high.school', 'basic.9y',
                                           'professional.course', 'basic.4y', 'basic.6y',
                                           'unknown', 'illiterate'])
    housing = st.selectbox("Has Housing Loan?", ['yes', 'no', 'unknown'])
    contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
    month = st.selectbox("Last Contact Month", ['may', 'jul', 'aug', 'jun', 'nov', 'apr', 'oct', 'sep', 'mar', 'dec'])
    day_of_week = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
    poutcome = st.selectbox("Previous Outcome", ['nonexistent', 'failure', 'success'])

    submit = st.form_submit_button("🔮 Predict")

if submit:
    input_dict = {
        'age': age,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'was_contacted': 1 if was_contacted == 'yes' else 0,
        'job': job,
        'marital': marital,
        'education': education,
        'housing': housing,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'poutcome': poutcome
    }

    input_df = pd.DataFrame([input_dict])
    prob = model.predict_proba(input_df)[0][1]
    pred = int(prob >= threshold)
    label = "✅ Subscribed" if pred == 1 else "❌ Not Subscribed"

    st.success(f"🎯 Prediction: **{label}**")
    st.info(f"📊 Probability of Subscribing: **{prob * 100:.2f}%** (Threshold = {threshold:.2f})")

    result_df = input_df.copy()
    result_df["Subscription Probability"] = f"{prob * 100:.2f}%"
    result_df["Prediction"] = label

    single_csv = result_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Single Prediction CSV", single_csv, "single_prediction.csv", "text/csv")
