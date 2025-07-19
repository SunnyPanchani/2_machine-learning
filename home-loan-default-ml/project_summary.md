
---

## 📝 `project_summary.md`

```markdown
# 📊 Project Summary – Home Credit Default Risk

This project focuses on creating an end-to-end machine learning application using:

- **Dataset**: Home Credit Default Risk (Kaggle)
- **Goal**: Predict whether an applicant will default on their loan
- **Approach**:
  - Data Cleaning
  - Feature Engineering
  - Model Training (XGBoost)
  - Deployment via Flask web app

---

## 🧹 Data Cleaning

- Dropped irrelevant columns
- Handled missing values:
  - Categorical: most frequent
  - Numeric: mean imputation
- One-hot encoding for selected categorical variables
- Scaled numerical features using `StandardScaler`

---

## 🧠 Model Training

- Algorithm: `XGBClassifier`
- Balanced classes using `scale_pos_weight`
- Best parameters (RandomizedSearchCV):
  ```python
  {
      'learning_rate': 0.17,
      'max_depth': 7,
      'n_estimators': 290,
      'scale_pos_weight': ~13.35
  }

Probability threshold: 0.69

    Predicts 1 if P(Default) ≥ 0.69

