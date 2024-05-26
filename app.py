import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model from the pickle file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    """Home route."""
    return "Welcome to the Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    """Predict route."""
    # Validate input data
    input_data = request.get_json()
    if not input_data or not all(key in input_data for key in ['HEART RATE', 'CALORIES', 'TRISEMESTER', 'SLEEP TIME']):
        return jsonify({'error': 'Invalid input data'}), 400
    
    # Extract features and convert to 2D array
    features = [input_data['HEART RATE'], input_data['CALORIES'], input_data['TRISEMESTER'], input_data['SLEEP TIME']]
    features_array = np.array(features).reshape(1, -1)  # Shape into (1, n_features)
    
    # Make prediction
    prediction = model.predict(features_array)
    
    # Add a random number between 3 and 9 to the prediction
    random_number = random.randint(3, 9)
    prediction_with_random = prediction[0] 
    
    # Convert prediction to standard Python int if necessary
    prediction_value = int(prediction_with_random)  # Ensure prediction_value is always an int
    
    return jsonify({'prediction': prediction_value/30})

if __name__ == '__main__':
    app.run(debug=True)
