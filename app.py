import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Customer Churn Predictor", page_icon="💼", layout="centered")

# ----- CUSTOM STYLES -----
CUSTOM_CSS = """
<style>
/* Page and container */
.block-container{padding:1.25rem 1.5rem 2rem 1.5rem}
[data-testid='stHeader']{display:none}

/* Card */
.card{background:linear-gradient(180deg,#ffffff,#fbfdff);border-radius:12px;padding:18px;box-shadow:0 6px 18px rgba(15,23,42,0.08);}
.title{font-family: 'Segoe UI', Roboto, Arial, sans-serif; font-weight:700; color:#073763}
.subtitle{color:#334155}

/* Buttons */
.streamlit-button{background:#0f62fe !important;color:#fff !important;border-radius:10px;padding:8px 12px}

/* Metric */
.metric-container {display:flex;gap:1rem;align-items:center}
.prob-bubble{font-size:1.8rem; font-weight:700; color:#0f62fe}

/* Footer small */
.small {font-size:12px;color:#6b7280}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----- Helpers to load resources (cached) -----
@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        return tf.keras.models.load_model('model.h5')
    except Exception as e:
        return e

@st.cache_data
def load_pickles():
    with open('ohe_geo.pkl', 'rb') as file:
        ohe_geo = pickle.load(file)
    with open('labe_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return ohe_geo, label_encoder_gender, scaler

# ----- Load once with friendly UI -----
with st.spinner('Loading model and preprocessing objects...'):
    model_or_error = load_model()
    pickles = load_pickles()

if isinstance(model_or_error, Exception):
    st.error("Error loading model. Make sure TensorFlow is installed in the environment that runs Streamlit.\n" + repr(model_or_error))
    st.stop()

ohe_geo, label_encoder_gender, scaler = pickles
model = model_or_error

# ----- Layout -----
st.markdown("<div class='card'> <div class='title'>💼 Customer Churn Predictor</div> <div class='subtitle'>Enter customer data and get a churn probability</div></div>", unsafe_allow_html=True)
st.write("")

col1, col2 = st.columns([2, 1])

with col1:
    with st.form(key='predict_form'):
        st.subheader('Customer profile')
        geography = st.selectbox('Geography', ohe_geo.categories_[0])
        gender = st.selectbox('Gender', label_encoder_gender.classes_)
        age = st.slider('Age', 18, 92, 35)
        credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
        balance = st.number_input('Balance', min_value=0.0, format="%.2f")
        estimated_salary = st.number_input('Estimated Salary', min_value=0.0, format="%.2f")
        tenure = st.slider('Tenure (years)', 0, 10, 3)
        num_of_products = st.slider('Number of Products', 1, 4, 1)
        has_cr_card = st.selectbox('Has Credit Cards', ['No', 'Yes'])
        isActiveMember = st.selectbox('Is Active Member', ['No', 'Yes'])

        submitted = st.form_submit_button('Predict')

with col2:
    st.subheader('About this model')
    st.markdown("""
    - Trained to predict customer churn probability.
    - **Inputs:** Credit score, Age, Balance, etc.
    - **Outputs:** Probability + status (likely / not likely to churn).
    """)
    st.write('')
    st.markdown("<div class='card small'><b>Tip:</b> Use realistic values for better results.</div>", unsafe_allow_html=True)

# ----- Prediction logic -----
if submitted:
    # prepare input
    in_df = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
        'IsActiveMember': [1 if isActiveMember == 'Yes' else 0],
        'EstimatedSalary': [estimated_salary],
        'Geography': [geography]
    })

    geo_encoded = ohe_geo.transform(in_df[['Geography']]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))
    in_df = pd.concat([in_df.drop(columns=['Geography']), geo_encoded_df], axis=1)

    scaled = scaler.transform(in_df)
    pred = model.predict(scaled)
    prob = float(pred[0][0])
    percent = int(round(prob * 100))

    # result card
    if prob > 0.5:
        status = "Likely to churn"
        st.markdown(f"<div class='card' style='border-left:4px solid #ff6b6b'> <div style='display:flex;justify-content:space-between;align-items:center'> <div><h3 style='margin:0'>Prediction</h3><p style='margin:0'>{status}</p></div><div class='prob-bubble'>{percent}%</div></div></div>", unsafe_allow_html=True)
        st.error(f"Churn probability: {percent}%")
    else:
        status = "Not likely to churn"
        st.markdown(f"<div class='card' style='border-left:4px solid #10b981'> <div style='display:flex;justify-content:space-between;align-items:center'> <div><h3 style='margin:0'>Prediction</h3><p style='margin:0'>{status}</p></div><div class='prob-bubble'>{percent}%</div></div></div>", unsafe_allow_html=True)
        st.success(f"Churn probability: {percent}%")

    st.progress(percent)
    st.write('')
    st.markdown("<div class='small'>Model output is a probability between 0 and 100%. Use it as guidance, not an absolute decision.</div>", unsafe_allow_html=True)
else:
    st.info('Fill the form and click **Predict** to see results.')

# Footer
st.markdown("---")
st.markdown("<div class='small'>Built with ❤️ using Streamlit</div>", unsafe_allow_html=True)


