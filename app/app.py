from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model from the pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Welcome to the Penguin Classifier API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input features from the request body (assumed to be JSON)
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features).tolist()

    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)
