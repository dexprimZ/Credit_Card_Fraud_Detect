import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a function to predict the probability of fraud
def predict_fraud(features):
    prediction = model.predict(features)
    return prediction

# Create a Streamlit app
st.title('Credit Card Fraud Detection')

# Input features
st.header('Input Features')

# Create a form to collect the input features
with st.form('fraud_detection_form'):
    amount = st.number_input('Amount', value=0.0)
    time = st.number_input('Time', value=0.0)
    type = st.selectbox('Type', ['Purchase', 'Withdrawal', 'Transfer'])
    location = st.selectbox('Location', ['Home', 'Work', 'Other'])

# Submit the form
    submitted = st.form_submit_button("Submit")
    if submitted:
    # Collect the input features
      features = np.array([amount, time, type, location])

    # Predict the probability of fraud
      probability = predict_fraud(features)

    # Display the results
      st.write('The probability of fraud is {}%'.format(probability * 100))

    # If the probability is high, then the transaction is likely to be fraudulent.
      if probability > 0.5:
          st.write('The transaction is likely to be fraudulent.')
      else:
          st.write('The transaction is not likely to be fraudulent.')