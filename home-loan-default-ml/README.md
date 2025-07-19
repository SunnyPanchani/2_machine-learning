# 🏠 Home Credit Default Risk Prediction App

This project is a **Flask-based web application** that predicts whether a loan applicant is likely to default, using the **Home Credit Default Risk dataset** from Kaggle. It features data preprocessing, model inference with XGBoost, and file-based batch prediction via a user-friendly web UI.

---

### 🔗 Dataset

We used the official dataset from the Kaggle competition:

👉 [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/competitions/home-credit-default-risk/data)

Due to its size (~2.3GB), you must download the dataset manually from Kaggle and place it in the `data/` directory.
which contain 10 files and total 346 columns

---

### 🚀 Features

- Batch prediction via CSV upload
- Preprocessing includes:
  - Missing value imputation
  - Categorical encoding
  - Feature scaling
- Model: XGBoost Classifier
- HTML UI using Flask + Bootstrap
- Prediction threshold: **0.69** (i.e., if probability ≥ 0.69, classified as *Default*)

---

### 📂 Folder Structure

📁 app/
    📄 app.py
    📁 models/
        📄 cat_imputer.pkl
        📄 num_imputer.pkl
        📄 ohe.pkl        
        📄 scaler.pkl     
        📄 xgb_model.pkl  
    📁 static/
        📄 style.css      
    📁 templates/
        📄 base.html      
        📄 index.html     
        📄 single.html    
    📁 uploads/
        📄 home_credit_test_sample.csv
📁 data/
    📄 data.txt
📁 main/
    📄 main.py
📁 notebook/
    📄 house_price_loan_merge_data.ipynb
📄 home_credit.png
📄 project_summary.md
📄 requirements.txt
📄 README.md


### ▶️ Run Locally Or Colab 

pip install -r requirements.txt
Run app.py in coda base 3.12.5

