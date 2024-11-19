from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load pre-trained model and preprocessing objects
model = joblib.load('calibrated_xgboost_model.pkl')  # Path to your model
scaler = joblib.load('scaler.pkl')  # Path to your scaler
pca = joblib.load('pca.pkl')  # Path to your PCA

# Class index to name mapping
class_mapping = {
    0: 'ELASTICNETCV',
    1: 'HUBERREGRESSOR',
    2: 'LASSO',
    3: 'LinearSVR',
    4: 'QUANTILEREGRESSOR',
    5: 'XGBRegressor'
}

@app.route('/predict', methods=['POST'])
def predict():
    input_json = request.get_json()
    input_df = pd.DataFrame.from_dict(input_json, orient='index')

    # Preprocessing
    scaled_data = scaler.transform(input_df)
    pca_data = pca.transform(scaled_data)

    # Predict probabilities
    prob_predictions = model.predict_proba(pca_data)

    # Map predictions to class names
    mapped_predictions = {
        idx: {class_mapping[i]: prob for i, prob in enumerate(probs)}
        for idx, probs in enumerate(prob_predictions)
    }

    # Return predictions
    return jsonify(mapped_predictions)

if __name__ == '__main__':
    app.run(debug=True)
