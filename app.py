import warnings
# Silence the scikit-learn version mismatch warning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import flask
from flask import Flask, request, jsonify
import pickle
import numpy as np

# ... rest of your app.py code stays exactly the same

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Model is running. Send a POST request to /predict."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json()
        
        # Extract features in the correct order: 
        # beds, baths, size, lot_size, zip_code
        features = [
            data['beds'], 
            data['baths'], 
            data['size'], 
            data['lot_size'], 
            data['zip_code']
        ]
        
        # Convert to numpy array and reshape for prediction
        final_features = np.array([features])
        
        # Make prediction
        prediction = model.predict(final_features)
        
        return jsonify({
            'status': 'success',
            'prediction': float(prediction[0])
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
