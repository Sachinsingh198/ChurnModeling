import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd

# load model and pickles
model = tf.keras.models.load_model('model.h5')

with open('ohe_geo.pkl', 'rb') as file:
    ohe_geo = pickle.load(file)

with open('labe_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("Customer Churn Prediction")

# inputs (fixed labels/choices)
geography = st.selectbox('Geography', ohe_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)  # LabelEncoder has .classes_
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')  # label fixed
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Cards', [0, 1])  # choices fixed
isActiveMember = st.selectbox('Is Active Member', [0, 1])

# build input dataframe (include Geography)
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],  # fixed typo
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [isActiveMember],
    'EstimatedSalary': [estimated_salary],
    'Geography': [geography]   # include Geography so OHE can use it
})

# OHE geography (pass raw value or DataFrame column)
geo_encoded = ohe_geo.transform(input_data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.drop(columns=['Geography']), geo_encoded_df], axis=1)

# scaling and predict
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)
prediction_probab = float(prediction[0][0])

st.write(f"Churn Probablity : {prediction_probab}")

if prediction_probab > 0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")


