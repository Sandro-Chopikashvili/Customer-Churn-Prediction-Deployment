import os
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_xgb_churn_model_pipeline.joblib')

app = Flask(__name__)

model_pipeline = None
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return jsonify({'error': 'Model not loaded. Please check server logs for details.'}), 500

    try:
        data = request.get_json(force=True)

        expected_columns = [
            'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges', 'Charge_per_Tenure'
        ]

        input_data_dict_full = {col: data.get(col, None) for col in expected_columns}

        input_df = pd.DataFrame([input_data_dict_full], columns=expected_columns)

        input_df['SeniorCitizen'] = pd.to_numeric(input_df['SeniorCitizen'], errors='coerce').fillna(0).astype(int)
        input_df['tenure'] = pd.to_numeric(input_df['tenure'], errors='coerce').fillna(0).astype(int)
        input_df['MonthlyCharges'] = pd.to_numeric(input_df['MonthlyCharges'], errors='coerce').fillna(0.0).astype(float)
        input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce').fillna(0.0).astype(float)

        input_df['Charge_per_Tenure'] = np.where(input_df['tenure'] == 0,
                                                 0.0,
                                                 input_df['TotalCharges'] / input_df['tenure'])

        input_df['customerID'] = input_df['customerID'].astype(str)
        input_df['gender'] = input_df.get('gender', 'Male')
        input_df['Partner'] = input_df.get('Partner', 'No')
        input_df['Dependents'] = input_df.get('Dependents', 'No')
        input_df['PhoneService'] = input_df.get('PhoneService', 'Yes')
        input_df['MultipleLines'] = input_df.get('MultipleLines', 'No')
        input_df['InternetService'] = input_df.get('InternetService', 'Fiber optic')
        input_df['OnlineSecurity'] = input_df.get('OnlineSecurity', 'No')
        input_df['OnlineBackup'] = input_df.get('OnlineBackup', 'No')
        input_df['DeviceProtection'] = input_df.get('DeviceProtection', 'No')
        input_df['TechSupport'] = input_df.get('TechSupport', 'No')
        input_df['StreamingTV'] = input_df.get('StreamingTV', 'No')
        input_df['StreamingMovies'] = input_df.get('StreamingMovies', 'No')
        input_df['Contract'] = input_df.get('Contract', 'Month-to-month')
        input_df['PaperlessBilling'] = input_df.get('PaperlessBilling', 'Yes')
        input_df['PaymentMethod'] = input_df.get('PaymentMethod', 'Electronic check')

        prediction_proba = model_pipeline.predict_proba(input_df)[0][1]
        churn_prediction = 'Yes' if prediction_proba >= 0.5 else 'No'

        return jsonify({
            'churn_probability': float(prediction_proba),
            'churn_prediction': churn_prediction
        })

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)