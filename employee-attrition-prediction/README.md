# Employee Attrition Prediction

This project uses machine learning to predict whether an employee is likely to leave or stay, based on their work habits, job satisfaction, and personal attributes.

## 🔍 Objective
To help HR departments take proactive steps in employee retention by identifying at-risk employees using predictive modeling.

## 📁 Project Structure
```
employee-attrition-prediction/
├── app/                # Streamlit app for live predictions + Trained logistic regression model + scaler
├── data/               # Raw and processed CSV files
├── models/             # Trained logistic regression model + scaler
├── notebook/           # Jupyter Notebook (EDA, preprocessing, training)
├── README.md
├── project_summary.md
└── .gitignore
```

## ⚙️ Tech Stack
- Python
- Pandas, NumPy, Scikit-learn
- Streamlit
- Joblib
- StandardScaler for Logistic Regression

## 📊 Selected Features
- Positively correlated (more likely to leave): `OverTime`, `MaritalStatus_Single`, `JobRole_Sales Representative`, `BusinessTravel_Travel_Frequently`
- Negatively correlated (more likely to stay): `TotalWorkingYears`, `JobLevel`, `YearsInCurrentRole`, `MonthlyIncome`, `Age`, `YearsWithCurrManager`, `YearsAtCompany`, `JobInvolvement`, `JobSatisfaction`, `EnvironmentSatisfaction`

## 🚀 How to Run
1. Clone the repository
2. Navigate to the project folder
3. Install dependencies:
   ```
   pip install -r app/requirements.txt
   ```
4. Run Streamlit app:
   ```
   streamlit run app/app.py
   ```

## 📄 Sample Prediction
Upload or manually enter employee data and get:
```
✅ Likely to Stay or ❌ Likely to Leave
```

## 📌 Why Logistic Regression?
- Balanced F1-Score and interpretability
- Best model during comparison (Threshold = 0.6)

| Model              | Threshold | Accuracy | F1-Score (Class 1) | Recall (Class 1) | Precision (Class 1) | Comments                      |
|-------------------|-----------|----------|--------------------|------------------|----------------------|-------------------------------|
| ✅ Logistic        | 0.60      | 0.8027   | 0.5167             | 0.6596           | 0.4247               | 🎯 Best F1. Interpretable     |
| Random Forest      | 0.20      | 0.7585   | 0.4818             | 0.7021           | 0.3708               | Good recall, lower accuracy   |
| XGBoost            | 0.20      | 0.8129   | 0.4954             | 0.5745           | 0.4355               | Best accuracy, lower F1       |

---