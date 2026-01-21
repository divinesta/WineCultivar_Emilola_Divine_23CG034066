from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and scaler
MODEL_PATH = os.path.join('model', 'wine_cultivar_model.pkl')
SCALER_PATH = os.path.join('model', 'feature_scaler.pkl')
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Feature names (the 6 features you selected during training)
# Update these based on the features you actually use in your model
FEATURE_NAMES = [
    'alcohol',
    'malic_acid',
    'ash',
    'total_phenols',
    'flavanoids',
    'color_intensity'
]

# Cultivar labels
CULTIVAR_LABELS = {
    0: 'Cultivar 1',
    1: 'Cultivar 2',
    2: 'Cultivar 3'
}


@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', features=FEATURE_NAMES)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get input data from form
        features = []
        for feature_name in FEATURE_NAMES:
            value = float(request.form.get(feature_name, 0))
            features.append(value)

        # Convert to numpy array and reshape for prediction
        features_array = np.array(features).reshape(1, -1)

        # Scale the features (IMPORTANT: Must scale before prediction!)
        features_scaled = scaler.transform(features_array)

        # Make prediction
        prediction = model.predict(features_scaled)
        cultivar = CULTIVAR_LABELS.get(prediction[0], 'Unknown')

        # Get prediction probabilities (if available)
        try:
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = {
                'Cultivar 1': f'{probabilities[0]*100:.2f}%',
                'Cultivar 2': f'{probabilities[1]*100:.2f}%',
                'Cultivar 3': f'{probabilities[2]*100:.2f}%'
            }
        except:
            confidence = None

        return render_template('index.html',
                               features=FEATURE_NAMES,
                               prediction=cultivar,
                               confidence=confidence,
                               input_values=dict(zip(FEATURE_NAMES, features)))

    except Exception as e:
        error_message = f"Error making prediction: {str(e)}"
        return render_template('index.html',
                               features=FEATURE_NAMES,
                               error=error_message)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    try:
        data = request.get_json()
        features = [data.get(feature, 0) for feature in FEATURE_NAMES]
        features_array = np.array(features).reshape(1, -1)

        prediction = model.predict(features_array)
        cultivar = CULTIVAR_LABELS.get(prediction[0], 'Unknown')

        try:
            probabilities = model.predict_proba(features_array)[0]
            confidence = {
                'cultivar_1': float(probabilities[0]),
                'cultivar_2': float(probabilities[1]),
                'cultivar_3': float(probabilities[2])
            }
        except:
            confidence = None

        return jsonify({
            'success': True,
            'prediction': cultivar,
            'prediction_class': int(prediction[0]),
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
