import joblib
import pandas as pd

# Load the trained model from the file
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
    return prediction


# Example custom data
custom_data = pd.DataFrame({
    'Age': [21],
    'Gender': ['Male'],
    'Stream': ['Mechanical'],
    'Internships': [1],
    'CGPA': [8],
    'Hostel': [1],
    'HistoryOfBacklogs': [0]
})

# Predict on custom data
prediction = predict_custom_data(custom_data)
print(f'Custom Data Prediction: {prediction[0]}')
