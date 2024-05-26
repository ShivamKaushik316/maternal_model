# # import numpy as np
# # import pickle
# # from flask import Flask, request, jsonify

# # app = Flask(__name__)

# # # Load the model from the pickle file
# # with open('model.pkl', 'rb') as model_file:
# #     model = pickle.load(model_file)

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     input_data = request.get_json()
    
# #     # Extract features and convert to 2D array
# #     features = [input_data['HEART RATE'], input_data['CALORIES'], input_data['TRISEMESTER'], input_data['SLEEP TIME']]
# #     features_array = np.array(features).reshape(1, -1)  # Shape into (1, n_features)
    
# #     # Make prediction
# #     prediction = model.predict(features_array)
    
# #     # Convert prediction to standard Python int if necessary
# #     prediction_value = prediction[0]
# #     if isinstance(prediction_value, np.integer):
# #         prediction_value = int(prediction_value)
    
# #     return jsonify({'prediction': prediction_value})

# # if __name__ == '__main__':
# #     app.run(debug=True)




# import numpy as np
# import pickle
# from flask import Flask, request, jsonify, render_template

# app = Flask(__name__)

# # Load the model from the pickle file
# with open('model.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)



# @app.route('/predict', methods=['POST'])
# def predict():
#     input_data = request.get_json()
    
#     # Extract features and convert to 2D array
#     features = [input_data['HEART RATE'], input_data['CALORIES'], input_data['TRISEMESTER'], input_data['SLEEP TIME']]
#     features_array = np.array(features).reshape(1, -1)  # Shape into (1, n_features)
    
#     # Make prediction
#     prediction = model.predict(features_array)
    
#     # Convert prediction to standard Python int if necessary
#     prediction_value = prediction[0]
#     if isinstance(prediction_value, np.integer):
#         prediction_value = int(prediction_value)









import numpy as np
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model from the pickle file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return "Welcome to the Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    
    # Extract features and convert to 2D array
    features = [input_data['HEART RATE'], input_data['CALORIES'], input_data['TRISEMESTER'], input_data['SLEEP TIME']]
    features_array = np.array(features).reshape(1, -1)  # Shape into (1, n_features)
    
    # Make prediction
    prediction = model.predict(features_array)
    
    # Convert prediction to standard Python int if necessary
    prediction_value = prediction[0]
    if isinstance(prediction_value, np.integer):
        prediction_value = int(prediction_value)
    
    return jsonify({'prediction': prediction_value})

if __name__ == '__main__':
    app.run(debug=True)

    
#     return jsonify({'prediction': prediction_value})

# if __name__ == '__main__':
#     app.run(debug=True)
