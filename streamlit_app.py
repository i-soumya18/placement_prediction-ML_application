import streamlit as st
import joblib
import pandas as pd

# Load the trained model and encoders
loaded_model = joblib.load('ensemble_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder_gender = joblib.load('label_encoder_gender.pkl')
label_encoder_stream = joblib.load('label_encoder_stream.pkl')

# Function to preprocess and predict custom data
def predict_custom_data(custom_data):
    # Encode categorical variables
    custom_data['Gender'] = label_encoder_gender.transform(custom_data['Gender'])
    custom_data['Stream'] = label_encoder_stream.transform(custom_data['Stream'])

    # Scale features
    custom_data_scaled = scaler.transform(custom_data)

    # Make prediction
    prediction = loaded_model.predict(custom_data_scaled)
    return 'Placed' if prediction[0] == 1 else 'Not Placed'

# Set page configuration
st.set_page_config(page_title="Placement Prediction App", layout="centered")

# Add a title and subtitle
st.title("🎓 Placement Prediction App")
st.subheader("Fill in the details below to check your placement chances")

# Add CSS for custom styling
st.markdown("""
    <style>
    body {
        background-image: url('placement-prediction-using-machine-learning.jpg');
        background-size: cover;
        color: #fff;
    }
    .stForm {
        background-color: rgba(0, 0, 0, 0.7);
        padding: 20px;
        border-radius: 10px;
    }
    .stButton {
        background-color: #0A74DA;
        color: white;
        font-weight: bold;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# User input form with a card-like appearance
with st.form(key='prediction_form', clear_on_submit=True):
    age = st.number_input("Age", min_value=18, max_value=100, value=21, step=1)
    gender = st.selectbox("Gender", options=['Male', 'Female'])
    stream = st.selectbox("Stream", options=['Mechanical', 'Computer Science', 'Electronics And Communication', 'Information Technology', 'Electrical'])
    internships = st.number_input("Number of Internships", min_value=0, value=1, step=1)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.0, format="%.2f")
    hostel = st.selectbox("Hostel Accommodation", options=[0, 1], format_func=lambda x: 'Yes' if x else 'No')  # 0 = No, 1 = Yes
    backlogs = st.number_input("History of Backlogs", min_value=0, value=0, step=1)

    # Submit button
    submit_button = st.form_submit_button("Predict")

# Prediction logic
if submit_button:
    # Create a DataFrame for the custom data
    custom_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Stream': [stream],
        'Internships': [internships],
        'CGPA': [cgpa],
        'Hostel': [hostel],
        'HistoryOfBacklogs': [backlogs]
    })

    # Make prediction
    prediction_result = predict_custom_data(custom_data)

    # Display prediction result
    st.success(f"Prediction: **{prediction_result}**")
