import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score,classification_report,confusion_matrix,precision_score
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle

from sklearn.model_selection import StratifiedKFold


from joblib import dump


application_train=pd.read_csv('application_train_merged.csv')
application_test=pd.read_csv('application_test_merged.csv')

missing = application_train.isnull().mean().sort_values(ascending=False)

threshold = 0.4
drop_cols = missing[missing > threshold].index.tolist()

application_train_clean = application_train.drop(columns=drop_cols)
application_test_clean = application_test.drop(columns=drop_cols)

print(f"Dropped {len(drop_cols)} columns with >{threshold*100}% missing values")

app_train_clean = application_train_clean.copy()
app_test_clean = application_test_clean.copy()

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# -------------------------
# STEP 1: Split X and y
# -------------------------
X = app_train_clean.drop(columns=['TARGET'])
y = app_train_clean['TARGET']
X_test = app_test_clean.copy()

# -------------------------
# STEP 2: Separate columns
# -------------------------
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()

# -------------------------
# STEP 3: Impute & Scale numeric columns
# -------------------------
num_imputer = SimpleImputer(strategy='median')
X_num = num_imputer.fit_transform(X[num_cols])
X_test_num = num_imputer.transform(X_test[num_cols])

scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)
X_test_num_scaled = scaler.transform(X_test_num)

# -------------------------
# STEP 4: Impute & One-Hot Encode categorical columns
# -------------------------
cat_imputer = SimpleImputer(strategy='most_frequent')
X_cat = cat_imputer.fit_transform(X[cat_cols])
X_test_cat = cat_imputer.transform(X_test[cat_cols])

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat_encoded = ohe.fit_transform(X_cat)
X_test_cat_encoded = ohe.transform(X_test_cat)

# -------------------------
# STEP 5: Combine
# -------------------------
X_final = np.hstack([X_num_scaled, X_cat_encoded])
X_test_final = np.hstack([X_test_num_scaled, X_test_cat_encoded])

# -------------------------
# STEP 6: Train XGBoost
# -------------------------
best_params = {
    'learning_rate': float(np.float64(0.18276027831785724)),
    'max_depth': 4,
    'n_estimators': 200,
    'scale_pos_weight': float(np.float64(13.0054513608871)),
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

xgb_model = XGBClassifier(**best_params)
xgb_model.fit(X_final, y)

# -------------------------
# STEP 7: Threshold tuning (0.69)
# -------------------------
y_pred_proba = xgb_model.predict_proba(X_final)[:, 1]
y_pred = (y_pred_proba >= 0.69).astype(int)

print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))

# -------------------------
# STEP 8: Save as .pkl
# -------------------------
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

with open('num_imputer.pkl', 'wb') as f:
    pickle.dump(num_imputer, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('cat_imputer.pkl', 'wb') as f:
    pickle.dump(cat_imputer, f)

with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe, f)

print("✅ All models and preprocessing objects saved for Streamlit app.")
