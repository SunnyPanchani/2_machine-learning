# 📊 Bank Marketing Campaign Analysis and Prediction

This project explores and models the UCI Bank Marketing dataset to understand customer behavior and predict whether a client will subscribe to a term deposit.

## 🔍 Project Overview

- **Dataset**: UCI Bank Marketing (Portuguese bank)
- **Goal**: Predict if a customer will say "yes" to a term deposit offer.
- **Type**: Binary classification (Target: `y` → `yes` / `no`)

## 📁 Folder Structure

eda-marketing-strategy/
├── data/
│ └── bank-additional-full.csv # Raw dataset from UCI
├── notebooks/
│ └── bank_marketing_analysis.ipynb # EDA + ML model building
├── models/
│ ├── best_bank_marketing_model.pkl # Final trained pipeline
│ └── best_threshold.pkl # Optimal threshold value
├── app/
│ └── app.py # Streamlit web app
├── README.md
├── requirements.txt
└── .gitignore


---

## 📊 Dataset Overview

- **Source:** UCI Bank Marketing Dataset  
- **Target Variable:** `y` — whether the client subscribed to a term deposit (`yes` / `no`)  
- **Observations:** 41,188  
- **Features:** 20 (mix of numeric & categorical)

---

## 📌 Project Goals

- Perform in-depth EDA to understand subscription behavior
- Preprocess and engineer features based on business insight
- Train and evaluate multiple ML models
- Optimize classification threshold for better recall/f1
- Build a user-friendly prediction app using Streamlit

---



## ✅ Final Model Used

- **Model**: XGBoost Classifier
- **Best Threshold**: 0.65
- **Performance**:

| Model               | ROC AUC | F1 Score |
|--------------------|---------|----------|
| Logistic Regression| 0.7990  | 0.4594   |
| Random Forest      | 0.7823  | 0.3916   |
| **XGBoost**        | **0.7914**  | **0.4647**   |

🔹 At threshold **0.65**, XGBoost provided the best trade-off between precision and recall for this dataset.

---

## 🎯 Features Used for Prediction

| Feature Name          | Description                                            |
|-----------------------|--------------------------------------------------------|
| `age`                 | Age of the client                                      |
| `campaign`            | Number of contacts during this campaign                |
| `pdays`               | Days since the client was last contacted (999 = never) |
| `previous`            | Number of previous contacts                            |
| `cons.price.idx`      | Consumer price index                                   |
| `cons.conf.idx`       | Consumer confidence index                              |
| `euribor3m`           | Euribor 3-month rate                                   |
| `was_contacted`       | Whether client was contacted before                    |
| `job`                 | Job category                                           |
| `marital`             | Marital status                                         |
| `education`           | Education level                                        |
| `housing`             | Housing loan status                                    |
| `contact`             | Contact communication type                             |
| `month`               | Month of last contact                                  |
| `day_of_week`         | Day of the week of last contact                        |
| `poutcome`            | Outcome of previous marketing campaign                 |

---
## 🚀 App Features (Streamlit UI)

- 📥 Upload CSV or use a single form for prediction
- ✅ Batch and single prediction support
- 🎯 Shows subscription probability
- ⬇️ Download results as CSV
- 📈 Uses pre-trained XGBoost model with optimal thresholding

---

## 🛠️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/app.py