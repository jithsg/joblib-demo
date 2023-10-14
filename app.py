from flask import Flask, request, jsonify
from sklearn import datasets
import joblib
app = Flask(__name__)


# Load the saved model
model = joblib.load('iris_model.joblib')
iris = datasets.load_iris()

@app.route('/')
def home():
    return 'Scikit-Learn Model with Flask API'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data as JSON
        data = request.get_json()

        # Ensure the input data has the expected features
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" in the input data'}), 400

        # Make predictions using the loaded model
        features = data['features']
        prediction = model.predict([features])[0]

        # Map the prediction to the corresponding class label
        target_names = iris.target_names
        predicted_class = target_names[prediction]

        return jsonify({'predicted_class': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
