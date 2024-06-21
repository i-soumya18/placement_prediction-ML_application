from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and encoders
model = joblib.load('ensemble_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder_gender = joblib.load('label_encoder_gender.pkl')
label_encoder_stream = joblib.load('label_encoder_stream.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = request.form['age']
    gender = request.form['gender']
    stream = request.form['stream']
    internships = request.form['internships']
    cgpa = request.form['cgpa']
    hostel = request.form['hostel']
    backlogs = request.form['backlogs']

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

    # Encode categorical variables
    custom_data['Gender'] = label_encoder_gender.transform(custom_data['Gender'])
    custom_data['Stream'] = label_encoder_stream.transform(custom_data['Stream'])

    # Scale features
    custom_data_scaled = scaler.transform(custom_data)

    # Make prediction
    prediction = model.predict(custom_data_scaled)
    prediction_result = 'Placed' if prediction[0] == 1 else 'Not Placed'

    return render_template('result.html', prediction=prediction_result)

if __name__ == "__main__":
    app.run(debug=True)
