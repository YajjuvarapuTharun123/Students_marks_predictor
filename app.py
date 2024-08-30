from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
scaler = joblib.load('scaler.pkl')
model = joblib.load('student_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [float(request.form.get('Age')),
                int(request.form.get('Gender')),
                int(request.form.get('Ethnicity')),
                int(request.form.get('ParentalEducation')),
                float(request.form.get('StudyTimeWeekly')),
                float(request.form.get('Absences')),
                int(request.form.get('Tutoring')),
                int(request.form.get('ParentalSupport')),
                int(request.form.get('Extracurricular')),
                int(request.form.get('Sports')),
                int(request.form.get('Music')),
                int(request.form.get('Volunteering')),
                float(request.form.get('GPA'))]
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Predict
    prediction = model.predict(features_scaled)
    
    # Return result
    return render_template('result.html', grade_class=int(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
