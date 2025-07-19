from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import os
import pickle
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
MODEL_FOLDER = 'models'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load all models
with open(f'{MODEL_FOLDER}/xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open(f'{MODEL_FOLDER}/num_imputer.pkl', 'rb') as f:
    num_imputer = pickle.load(f)
with open(f'{MODEL_FOLDER}/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open(f'{MODEL_FOLDER}/cat_imputer.pkl', 'rb') as f:
    cat_imputer = pickle.load(f)
with open(f'{MODEL_FOLDER}/ohe.pkl', 'rb') as f:
    ohe = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        df = pd.read_csv(filepath)

        # Split numeric and categorical
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()

        # Preprocess
        X_num = num_imputer.transform(df[num_cols])
        X_num_scaled = scaler.transform(X_num)

        X_cat = cat_imputer.transform(df[cat_cols])
        X_cat_encoded = ohe.transform(X_cat)

        # Combine
        X_final = np.hstack([X_num_scaled, X_cat_encoded])

        # Predict
        y_pred_proba = model.predict_proba(X_final)[:, 1]
        y_pred = (y_pred_proba >= 0.69).astype(int)

        # Add to DataFrame
        df['PREDICTED_PROBABILITY'] = y_pred_proba
        df['PREDICTED_TARGET'] = y_pred

        # Save result
        result_path = os.path.join(RESULT_FOLDER, 'application_test_predicted.csv')
        df.to_csv(result_path, index=False)

        return send_file(result_path, as_attachment=True)

    return "File upload failed"

if __name__ == '__main__':
    app.run(debug=True)
