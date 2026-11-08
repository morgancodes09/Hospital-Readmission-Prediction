"""
===============================================================================
HOSPITAL READMISSION PREDICTION - COMPLETE ML PIPELINE
===============================================================================
Dataset    : 11_hospital_readmission_cleaned.csv
Problem    : Binary Classification (Imbalanced ~80/20)
Target     : readmitted_30d (1 = readmitted within 30 days, 0 = not)
Models     : Logistic Regression, Random Forest, XGBoost, ANN (MLPClassifier)
Primary    : ROC-AUC Score (handles class imbalance better than accuracy)
===============================================================================
"""

# ============================================================================
# STEP 0 : IMPORTS
# ============================================================================
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                       # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import os, joblib, time

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     RandomizedSearchCV, StratifiedKFold)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, accuracy_score, confusion_matrix,
                             classification_report, roc_curve)
import sklearn

# Detect OneHotEncoder sparse parameter name (sparse_output for sklearn>=1.2, sparse for older)
_sklearn_version = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
_OHE_SPARSE_PARAM = 'sparse_output' if _sklearn_version >= (1, 2) else 'sparse'
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[INFO] XGBoost not installed. Will skip XGBoost model.")
    print("       Install with: pip install xgboost\n")

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Output directory for plots and saved model
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 80)
print("  HOSPITAL READMISSION PREDICTION - END-TO-END ML PIPELINE")
print("=" * 80)


# ============================================================================
# STEP 1 : LOAD DATA
# ============================================================================
print("\n" + "=" * 80)
print("  STEP 1 : LOADING DATA")
print("=" * 80)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CANDIDATES = [
    os.path.join(BASE_DIR, 'data', '11_hospital_readmission_cleaned.csv'),
    os.path.join(BASE_DIR, '11_hospital_readmission_cleaned.csv'),
]
DATA_PATH = next((path for path in DATA_CANDIDATES if os.path.isfile(path)), None)
if DATA_PATH is None:
    raise FileNotFoundError(
        'Dataset not found. Place 11_hospital_readmission_cleaned.csv '
        'inside the project data/ folder or project root.'
    )
df = pd.read_csv(DATA_PATH)

print(f"  Shape             : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Columns           : {list(df.columns)}")
print(f"  Target distribution:")
target_counts = df['readmitted_30d'].value_counts()
for val, cnt in target_counts.items():
    print(f"    Class {val} : {cnt}  ({cnt / len(df) * 100:.1f}%)")
print(f"  Missing values    : {df.isnull().sum().sum()}")
print(f"\n  First 5 rows:")
print(df.head().to_string(index=False))


# ============================================================================
# STEP 2 : DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("  STEP 2 : DATA PREPROCESSING")
print("=" * 80)

# 2.1  Drop the 'id' column (not a feature)
df = df.drop(columns=['id'])
print("  [2.1] Dropped 'id' column (not a predictive feature)")

# 2.2  Handle missing values using group-wise median imputation
#      BMI is imputed per diagnosis group to avoid bias from non-random missingness
numeric_cols = ['age', 'length_of_stay', 'num_medications', 'prev_admissions', 'bmi']
for col in numeric_cols:
    if df[col].isnull().any():
        # Group-wise median imputation (using diagnosis as the grouping variable)
        df[col] = df.groupby('diagnosis')[col].transform(
            lambda x: x.fillna(x.median())
        )
        # Fallback: if entire group is NaN, use global median
        df[col] = df[col].fillna(df[col].median())
        print(f"  [2.2] Imputed missing values in '{col}' using group-wise median (by diagnosis)")

if df.isnull().sum().sum() == 0:
    print("  [2.2] No missing values found - data is already clean")

# 2.3  Verify data types
print(f"\n  Data types after preprocessing:")
for col in df.columns:
    print(f"    {col:25s} : {df[col].dtype}")


# ============================================================================
# STEP 3 : FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 80)
print("  STEP 3 : FEATURE ENGINEERING")
print("=" * 80)

# 3.1  age_group: Binning age into clinically meaningful categories
def assign_age_group(age):
    if age <= 40:
        return '18-40'
    elif age <= 60:
        return '41-60'
    else:
        return '61+'

df['age_group'] = df['age'].apply(assign_age_group)
print("  [3.1] Created 'age_group' : 18-40, 41-60, 61+")
print(f"         Distribution: {dict(df['age_group'].value_counts())}")

# 3.2  high_risk_flag: Patients with >= 2 previous admissions are high-risk
df['high_risk_flag'] = (df['prev_admissions'] >= 2).astype(int)
print(f"  [3.2] Created 'high_risk_flag' : 1 if prev_admissions >= 2")
print(f"         Distribution: {dict(df['high_risk_flag'].value_counts())}")

# 3.3  medication_per_day: Medication intensity during hospital stay
df['medication_per_day'] = df['num_medications'] / df['length_of_stay'].replace(0, np.nan)
df['medication_per_day'] = df['medication_per_day'].replace([np.inf, -np.inf], np.nan)
df['medication_per_day'] = df['medication_per_day'].fillna(df['medication_per_day'].median())
print(f"  [3.3] Created 'medication_per_day' = num_medications / length_of_stay")
print(f"         Range: {df['medication_per_day'].min():.2f} - {df['medication_per_day'].max():.2f}")

print(f"\n  Final feature set ({df.shape[1] - 1} features + 1 target):")
for col in df.columns:
    print(f"    {col}")


# ============================================================================
# STEP 4 : TRAIN-TEST SPLIT  (Stratified 80/20)
# ============================================================================
print("\n" + "=" * 80)
print("  STEP 4 : STRATIFIED TRAIN-TEST SPLIT (80/20)")
print("=" * 80)

# Separate features and target
TARGET = 'readmitted_30d'
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Identify column types for the preprocessor
categorical_features = ['diagnosis', 'discharge_type', 'age_group']
numerical_features   = [c for c in X.columns if c not in categorical_features]

print(f"  Numerical features  ({len(numerical_features)}): {numerical_features}")
print(f"  Categorical features ({len(categorical_features)}): {categorical_features}")

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print(f"\n  Training set : {X_train.shape[0]} samples")
print(f"  Test set     : {X_test.shape[0]} samples")
print(f"  Train target : {dict(y_train.value_counts())}")
print(f"  Test  target : {dict(y_test.value_counts())}")

# Class imbalance ratio for XGBoost scale_pos_weight
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"\n  Imbalance ratio (neg/pos) : {scale_pos_weight:.2f}")
print(f"  -> This value is used as scale_pos_weight in XGBoost")


# ============================================================================
# STEP 5 : BUILD PREPROCESSING PIPELINES
# ============================================================================
print("\n" + "=" * 80)
print("  STEP 5 : BUILDING PREPROCESSING PIPELINES")
print("=" * 80)

# Build OneHotEncoder kwargs compatible with both old and new sklearn
_ohe_kwargs = {'drop': 'first', 'handle_unknown': 'ignore', _OHE_SPARSE_PARAM: False}

# Pipeline A: WITH scaling (for Logistic Regression & ANN)
preprocessor_scaled = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(**_ohe_kwargs), categorical_features)
    ],
    remainder='drop'
)

# Pipeline B: WITHOUT scaling (for Random Forest & XGBoost - tree-based, scale-invariant)
preprocessor_unscaled = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(**_ohe_kwargs), categorical_features)
    ],
    remainder='drop'
)

print("  [A] Preprocessor WITH StandardScaler    -> Logistic Regression, ANN")
print("  [B] Preprocessor WITHOUT StandardScaler -> Random Forest, XGBoost")
print("  [*] One-Hot Encoding applied to categorical variables (drop='first')")
print("  [*] Outliers are NOT removed (extreme values may be clinically valid)")

# Get feature names after encoding (from the fitted preprocessor to avoid mismatch)
preprocessor_unscaled.fit(X_train)
all_feature_names = list(preprocessor_unscaled.get_feature_names_out())
# Clean up prefixes added by ColumnTransformer (e.g., 'num__age' -> 'age')
all_feature_names = [name.split('__', 1)[-1] if '__' in name else name
                     for name in all_feature_names]
print(f"\n  Total features after encoding: {len(all_feature_names)}")
print(f"  Feature names: {all_feature_names}")


# ============================================================================
# STEP 6 : MODEL TRAINING + HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "=" * 80)
print("  STEP 6 : MODEL TRAINING + HYPERPARAMETER TUNING")
print("=" * 80)

# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Dictionary to store results
results = {}

# ---- Helper function to evaluate a model ----
def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    """Train, predict, and compute all evaluation metrics."""
    start = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - start

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    metrics = {
        'ROC-AUC'   : roc_auc_score(y_te, y_prob),
        'F1'        : f1_score(y_te, y_pred),
        'Precision' : precision_score(y_te, y_pred, zero_division=0),
        'Recall'    : recall_score(y_te, y_pred),
        'Accuracy'  : accuracy_score(y_te, y_pred),
        'Train Time': f"{train_time:.1f}s",
    }
    cm = confusion_matrix(y_te, y_pred)
    return model, y_pred, y_prob, metrics, cm


# --------------------------------------------------------------------------
# MODEL 1 : LOGISTIC REGRESSION  (with StandardScaler, class_weight='balanced')
# --------------------------------------------------------------------------
print("\n" + "-" * 70)
print("  MODEL 1 : LOGISTIC REGRESSION")
print("-" * 70)

lr_pipeline = Pipeline([
    ('preprocessor', preprocessor_scaled),
    ('classifier', LogisticRegression(class_weight='balanced',
                                      random_state=RANDOM_STATE,
                                      max_iter=1000))
])

lr_param_grid = {
    'classifier__C'       : [0.1, 1, 10],
    'classifier__penalty' : ['l1', 'l2'],
    'classifier__solver'  : ['liblinear'],
}

print("  Running GridSearchCV (5-fold) ...")
lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=cv, scoring='roc_auc',
                       n_jobs=2, verbose=0, refit=True)
lr_grid.fit(X_train, y_train)

print(f"  Best params : {lr_grid.best_params_}")
print(f"  Best CV ROC-AUC : {lr_grid.best_score_:.4f}")

# Evaluate on test set
lr_model = lr_grid.best_estimator_
y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

lr_metrics = {
    'ROC-AUC'   : roc_auc_score(y_test, y_prob_lr),
    'F1'        : f1_score(y_test, y_pred_lr),
    'Precision' : precision_score(y_test, y_pred_lr, zero_division=0),
    'Recall'    : recall_score(y_test, y_pred_lr),
    'Accuracy'  : accuracy_score(y_test, y_pred_lr),
}
lr_cm = confusion_matrix(y_test, y_pred_lr)
results['Logistic Regression'] = {
    'model': lr_model, 'y_pred': y_pred_lr, 'y_prob': y_prob_lr,
    'metrics': lr_metrics, 'cm': lr_cm
}

print(f"\n  Test Set Results:")
for k, v in lr_metrics.items():
    print(f"    {k:12s} : {v:.4f}")
print(f"  Confusion Matrix:\n{lr_cm}")


# --------------------------------------------------------------------------
# MODEL 2 : RANDOM FOREST  (no scaling, class_weight='balanced')
# --------------------------------------------------------------------------
print("\n" + "-" * 70)
print("  MODEL 2 : RANDOM FOREST")
print("-" * 70)

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor_unscaled),
    ('classifier', RandomForestClassifier(class_weight='balanced',
                                          random_state=RANDOM_STATE, n_jobs=1))
])

rf_param_grid = {
    'classifier__n_estimators' : [100, 200],
    'classifier__max_depth'    : [5, 10, None],
    'classifier__min_samples_split' : [2, 5],
    'classifier__min_samples_leaf'  : [1, 2],
}

print("  Running GridSearchCV (5-fold) ...")
rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=cv, scoring='roc_auc',
                       n_jobs=2, verbose=0, refit=True)
rf_grid.fit(X_train, y_train)

print(f"  Best params : {rf_grid.best_params_}")
print(f"  Best CV ROC-AUC : {rf_grid.best_score_:.4f}")

rf_model = rf_grid.best_estimator_
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

rf_metrics = {
    'ROC-AUC'   : roc_auc_score(y_test, y_prob_rf),
    'F1'        : f1_score(y_test, y_pred_rf),
    'Precision' : precision_score(y_test, y_pred_rf, zero_division=0),
    'Recall'    : recall_score(y_test, y_pred_rf),
    'Accuracy'  : accuracy_score(y_test, y_pred_rf),
}
rf_cm = confusion_matrix(y_test, y_pred_rf)
results['Random Forest'] = {
    'model': rf_model, 'y_pred': y_pred_rf, 'y_prob': y_prob_rf,
    'metrics': rf_metrics, 'cm': rf_cm
}

print(f"\n  Test Set Results:")
for k, v in rf_metrics.items():
    print(f"    {k:12s} : {v:.4f}")
print(f"  Confusion Matrix:\n{rf_cm}")


# --------------------------------------------------------------------------
# MODEL 3 : XGBOOST  (no scaling, scale_pos_weight for imbalance)
# --------------------------------------------------------------------------
if XGBOOST_AVAILABLE:
    print("\n" + "-" * 70)
    print("  MODEL 3 : XGBOOST")
    print("-" * 70)

    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor_unscaled),
        ('classifier', XGBClassifier(scale_pos_weight=scale_pos_weight,
                                     eval_metric='logloss',
                                     random_state=RANDOM_STATE, n_jobs=1))
    ])

    xgb_param_dist = {
        'classifier__n_estimators'  : [100, 200],
        'classifier__max_depth'     : [3, 5, 7],
        'classifier__learning_rate' : [0.01, 0.05, 0.1, 0.2],
        'classifier__subsample'     : [0.6, 0.8, 1.0],
        'classifier__colsample_bytree' : [0.6, 0.8, 1.0],
        'classifier__min_child_weight' : [1, 3, 5],
        'classifier__gamma'         : [0, 0.1, 0.3],
    }

    print("  Running RandomizedSearchCV (5-fold, 12 iterations) ...")
    xgb_random = RandomizedSearchCV(xgb_pipeline, xgb_param_dist,
                                    n_iter=12, cv=cv, scoring='roc_auc',
                                    n_jobs=2, verbose=0, refit=True,
                                    random_state=RANDOM_STATE)
    xgb_random.fit(X_train, y_train)

    print(f"  Best params : {xgb_random.best_params_}")
    print(f"  Best CV ROC-AUC : {xgb_random.best_score_:.4f}")

    xgb_model = xgb_random.best_estimator_
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    xgb_metrics = {
        'ROC-AUC'   : roc_auc_score(y_test, y_prob_xgb),
        'F1'        : f1_score(y_test, y_pred_xgb),
        'Precision' : precision_score(y_test, y_pred_xgb, zero_division=0),
        'Recall'    : recall_score(y_test, y_pred_xgb),
        'Accuracy'  : accuracy_score(y_test, y_pred_xgb),
    }
    xgb_cm = confusion_matrix(y_test, y_pred_xgb)
    results['XGBoost'] = {
        'model': xgb_model, 'y_pred': y_pred_xgb, 'y_prob': y_prob_xgb,
        'metrics': xgb_metrics, 'cm': xgb_cm
    }

    print(f"\n  Test Set Results:")
    for k, v in xgb_metrics.items():
        print(f"    {k:12s} : {v:.4f}")
    print(f"  Confusion Matrix:\n{xgb_cm}")


# --------------------------------------------------------------------------
# MODEL 4 : ANN / MLPClassifier  (with StandardScaler + sample_weight for imbalance)
# --------------------------------------------------------------------------
print("\n" + "-" * 70)
print("  MODEL 4 : ANN (MLPClassifier)")
print("-" * 70)

# MLPClassifier does NOT support class_weight parameter directly.
# To handle imbalance, we compute sample weights and pass them during fit().
# Since RandomizedSearchCV does not forward sample_weight automatically to
# a pipeline, we preprocess X_train first, then search on the classifier only.

print("  [*] Computing sample weights for class imbalance handling...")
ann_sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Pre-transform training and test data using the scaled preprocessor
preprocessor_scaled.fit(X_train)
X_train_scaled = preprocessor_scaled.transform(X_train)
X_test_scaled  = preprocessor_scaled.transform(X_test)

ann_base = MLPClassifier(random_state=RANDOM_STATE, max_iter=500,
                         early_stopping=True, validation_fraction=0.15)

ann_param_dist = {
    'hidden_layer_sizes' : [(64,), (128,), (64, 32), (128, 64),
                            (128, 64, 32), (64, 32, 16)],
    'activation'         : ['relu', 'tanh'],
    'alpha'              : [0.0001, 0.001, 0.01, 0.1],
    'learning_rate'      : ['constant', 'adaptive'],
    'learning_rate_init' : [0.001, 0.01, 0.005],
    'batch_size'         : [32, 64, 128],
}

print("  Running RandomizedSearchCV (5-fold, 10 iterations) ...")
print("  [*] sample_weight='balanced' applied to handle class imbalance in ANN")
ann_random = RandomizedSearchCV(ann_base, ann_param_dist,
                                n_iter=10, cv=cv, scoring='roc_auc',
                                n_jobs=2, verbose=0, refit=True,
                                random_state=RANDOM_STATE)
ann_random.fit(X_train_scaled, y_train, sample_weight=ann_sample_weights)

print(f"  Best params : {ann_random.best_params_}")
print(f"  Best CV ROC-AUC : {ann_random.best_score_:.4f}")

ann_clf_best = ann_random.best_estimator_
y_pred_ann = ann_clf_best.predict(X_test_scaled)
y_prob_ann = ann_clf_best.predict_proba(X_test_scaled)[:, 1]

# Wrap the best ANN back into a pipeline for consistency with other models
ann_model = Pipeline([
    ('preprocessor', preprocessor_scaled),
    ('classifier', ann_clf_best)
])

ann_metrics = {
    'ROC-AUC'   : roc_auc_score(y_test, y_prob_ann),
    'F1'        : f1_score(y_test, y_pred_ann),
    'Precision' : precision_score(y_test, y_pred_ann, zero_division=0),
    'Recall'    : recall_score(y_test, y_pred_ann),
    'Accuracy'  : accuracy_score(y_test, y_pred_ann),
}
ann_cm = confusion_matrix(y_test, y_pred_ann)
results['ANN (MLP)'] = {
    'model': ann_model, 'y_pred': y_pred_ann, 'y_prob': y_prob_ann,
    'metrics': ann_metrics, 'cm': ann_cm
}

print(f"\n  Test Set Results:")
for k, v in ann_metrics.items():
    print(f"    {k:12s} : {v:.4f}")
print(f"  Confusion Matrix:\n{ann_cm}")


# ============================================================================
# STEP 7 : MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("  STEP 7 : MODEL COMPARISON")
print("=" * 80)

comparison_data = []
for name, res in results.items():
    row = {'Model': name}
    row.update(res['metrics'])
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False).reset_index(drop=True)

print("\n  Performance Summary (sorted by ROC-AUC):")
print("  " + "-" * 76)
print(f"  {'Model':<22s} {'ROC-AUC':>8s} {'F1':>8s} {'Precision':>10s} {'Recall':>8s} {'Accuracy':>10s}")
print("  " + "-" * 76)
for _, row in comparison_df.iterrows():
    print(f"  {row['Model']:<22s} {row['ROC-AUC']:8.4f} {row['F1']:8.4f} "
          f"{row['Precision']:10.4f} {row['Recall']:8.4f} {row['Accuracy']:10.4f}")
print("  " + "-" * 76)

# Identify best model
best_model_name = comparison_df.iloc[0]['Model']
best_roc_auc    = comparison_df.iloc[0]['ROC-AUC']
best_f1         = comparison_df.iloc[0]['F1']
print(f"\n  >>> BEST MODEL: {best_model_name}")
print(f"      ROC-AUC = {best_roc_auc:.4f}  |  F1 = {best_f1:.4f}")

# Note on accuracy
print("\n  NOTE: Accuracy is shown for reference ONLY.")
print("  With ~80/20 class imbalance, a naive majority-class model would get ~80% accuracy.")
print("  ROC-AUC and F1 are the primary evaluation metrics for this problem.")


# ============================================================================
# STEP 8 : MODEL EXPLAINABILITY (Feature Importance)
# ============================================================================
print("\n" + "=" * 80)
print("  STEP 8 : MODEL EXPLAINABILITY")
print("=" * 80)

# --- 8.1 Random Forest Feature Importance ---
print("\n  [8.1] Random Forest - Feature Importance:")
rf_clf = rf_model.named_steps['classifier']
rf_importances = rf_clf.feature_importances_
rf_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': rf_importances
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print("  " + "-" * 50)
for _, row in rf_importance_df.iterrows():
    bar = '#' * int(row['Importance'] * 100)
    print(f"  {row['Feature']:<25s} : {row['Importance']:.4f}  {bar}")
print("  " + "-" * 50)

# --- 8.2 XGBoost Feature Importance ---
if XGBOOST_AVAILABLE:
    print("\n  [8.2] XGBoost - Feature Importance:")
    xgb_clf = xgb_model.named_steps['classifier']
    xgb_importances = xgb_clf.feature_importances_
    xgb_importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': xgb_importances
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    print("  " + "-" * 50)
    for _, row in xgb_importance_df.iterrows():
        bar = '#' * int(row['Importance'] * 100)
        print(f"  {row['Feature']:<25s} : {row['Importance']:.4f}  {bar}")
    print("  " + "-" * 50)

# --- 8.3 Logistic Regression Coefficients ---
print("\n  [8.3] Logistic Regression - Feature Coefficients (absolute values):")
lr_clf = lr_model.named_steps['classifier']
lr_coefs = np.abs(lr_clf.coef_[0])
lr_coef_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Abs Coefficient': lr_coefs
}).sort_values('Abs Coefficient', ascending=False).reset_index(drop=True)

print("  " + "-" * 50)
for _, row in lr_coef_df.iterrows():
    bar = '#' * int(row['Abs Coefficient'] * 10)
    print(f"  {row['Feature']:<25s} : {row['Abs Coefficient']:.4f}  {bar}")
print("  " + "-" * 50)

# --- 8.4 Key Findings ---
print("\n  [8.4] KEY FINDINGS - Top Factors Influencing Readmission:")
print("  " + "-" * 60)
top_rf = rf_importance_df.head(5)['Feature'].tolist()
print(f"  Random Forest Top 5  : {top_rf}")
if XGBOOST_AVAILABLE:
    top_xgb = xgb_importance_df.head(5)['Feature'].tolist()
    print(f"  XGBoost Top 5        : {top_xgb}")
top_lr = lr_coef_df.head(5)['Feature'].tolist()
print(f"  Logistic Reg Top 5   : {top_lr}")
print("  " + "-" * 60)


# ============================================================================
# STEP 9 : GENERATE PLOTS
# ============================================================================
print("\n" + "=" * 80)
print("  STEP 9 : GENERATING PLOTS")
print("=" * 80)

sns.set_style('whitegrid')
sns.set_palette('husl')

# --- Plot 1: ROC Curves for all models ---
fig, ax = plt.subplots(figsize=(10, 7))
colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']
for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    auc_val = res['metrics']['ROC-AUC']
    ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2.5,
            label=f"{name} (AUC = {auc_val:.4f})")

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random Baseline')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves - All Models', fontsize=16, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(OUTPUT_DIR, 'roc_curves.png')
plt.savefig(roc_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [1] ROC Curves saved to: {roc_path}")

# --- Plot 2: Confusion Matrices ---
n_models = len(results)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
if n_models == 1:
    axes = [axes]

for i, (name, res) in enumerate(results.items()):
    sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Readmitted', 'Readmitted'],
                yticklabels=['Not Readmitted', 'Readmitted'],
                ax=axes[i], cbar=False, annot_kws={'size': 14})
    axes[i].set_title(f'{name}', fontsize=13, fontweight='bold')
    axes[i].set_ylabel('Actual' if i == 0 else '', fontsize=12)
    axes[i].set_xlabel('Predicted', fontsize=12)

plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrices.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [2] Confusion Matrices saved to: {cm_path}")

# --- Plot 3: Metric Comparison Bar Chart ---
metric_names = ['ROC-AUC', 'F1', 'Precision', 'Recall']
x = np.arange(len(metric_names))
width = 0.18
fig, ax = plt.subplots(figsize=(12, 6))

for i, (name, res) in enumerate(results.items()):
    values = [res['metrics'][m] for m in metric_names]
    bars = ax.bar(x + i * width, values, width, label=name, color=colors[i % len(colors)],
                  edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x + width * (len(results) - 1) / 2)
ax.set_xticklabels(metric_names, fontsize=12)
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, 1.15)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
bar_path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
plt.savefig(bar_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [3] Model Comparison Chart saved to: {bar_path}")

# --- Plot 4: Feature Importance (Random Forest) ---
fig, ax = plt.subplots(figsize=(10, 7))
top_n = min(15, len(rf_importance_df))
plot_df = rf_importance_df.head(top_n).iloc[::-1]
bars = ax.barh(plot_df['Feature'], plot_df['Importance'],
               color=sns.color_palette('viridis', top_n), edgecolor='white')
ax.set_xlabel('Importance', fontsize=13)
ax.set_title('Feature Importance - Random Forest', fontsize=16, fontweight='bold')
for bar, val in zip(bars, plot_df['Importance']):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
fi_path = os.path.join(OUTPUT_DIR, 'feature_importance_rf.png')
plt.savefig(fi_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [4] Feature Importance (RF) saved to: {fi_path}")

# --- Plot 5: Feature Importance (XGBoost) ---
if XGBOOST_AVAILABLE:
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_df = xgb_importance_df.head(top_n).iloc[::-1]
    bars = ax.barh(plot_df['Feature'], plot_df['Importance'],
                   color=sns.color_palette('magma', top_n), edgecolor='white')
    ax.set_xlabel('Importance', fontsize=13)
    ax.set_title('Feature Importance - XGBoost', fontsize=16, fontweight='bold')
    for bar, val in zip(bars, plot_df['Importance']):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    fi_xgb_path = os.path.join(OUTPUT_DIR, 'feature_importance_xgb.png')
    plt.savefig(fi_xgb_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [5] Feature Importance (XGB) saved to: {fi_xgb_path}")


# ============================================================================
# STEP 10 : SAVE BEST MODEL + DETAILED CLASSIFICATION REPORTS
# ============================================================================
print("\n" + "=" * 80)
print("  STEP 10 : SAVE BEST MODEL + FINAL REPORTS")
print("=" * 80)

# Save the best model
best_model_obj = results[best_model_name]['model']
model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
joblib.dump(best_model_obj, model_path)
joblib.dump(best_model_obj, os.path.join(OUTPUT_DIR, 'best_model.pkl'))
print(f"\n  Best model ({best_model_name}) saved to: {model_path}")

# Print detailed classification report for each model
for name, res in results.items():
    print(f"\n  {'=' * 60}")
    print(f"  Classification Report: {name}")
    print(f"  {'=' * 60}")
    print(classification_report(y_test, res['y_pred'],
                                target_names=['Not Readmitted (0)', 'Readmitted (1)']))


# ============================================================================
# STEP 11 : TRADE-OFFS AND ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("  STEP 11 : TRADE-OFFS AND ANALYSIS")
print("=" * 80)

tradeoffs = """
  +------------------------+-----------------------------------------------+
  | Model                  | Trade-offs                                    |
  +------------------------+-----------------------------------------------+
  | Logistic Regression    | + Highly interpretable (coefficients)         |
  |                        | + Fast training and prediction                |
  |                        | + Good baseline model                        |
  |                        | - May underperform on complex, non-linear     |
  |                        |   patterns in clinical data                   |
  +------------------------+-----------------------------------------------+
  | Random Forest          | + Robust to outliers and noise                |
  |                        | + Captures non-linear relationships           |
  |                        | + Feature importance built-in                 |
  |                        | - Less interpretable than Logistic Regression |
  |                        | - Can overfit with many trees                 |
  +------------------------+-----------------------------------------------+
  | XGBoost                | + High performance, handles imbalance well    |
  |                        | + Built-in regularization (gamma, lambda)     |
  |                        | + scale_pos_weight handles class imbalance    |
  |                        | - Requires careful hyperparameter tuning      |
  |                        | - Less interpretable (black-box)              |
  +------------------------+-----------------------------------------------+
  | ANN (MLPClassifier)    | + Captures complex non-linear relationships   |
  |                        | + Can learn intricate feature interactions     |
  |                        | - Slowest to train                            |
  |                        | - Least interpretable (true black-box)        |
  |                        | - Sensitive to hyperparameters and scaling     |
  +------------------------+-----------------------------------------------+
"""
print(tradeoffs)


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 80)
print("  FINAL PIPELINE SUMMARY")
print("=" * 80)
print(f"""
  Dataset            : {DATA_PATH}
  Total Samples      : {len(df)}
  Features           : {len(all_feature_names)} (after encoding)
  Train / Test       : {len(X_train)} / {len(X_test)}
  Class Distribution : 0={neg_count} ({neg_count/len(y_train)*100:.1f}%) | 1={pos_count} ({pos_count/len(y_train)*100:.1f}%)
  
  Models Trained     : {len(results)}
  Best Model         : {best_model_name}
  Best ROC-AUC       : {best_roc_auc:.4f}
  Best F1 Score      : {best_f1:.4f}
  
  Outputs saved to   : {OUTPUT_DIR}
    - best_model.pkl           (trained model)
    - roc_curves.png           (ROC curves comparison)
    - confusion_matrices.png   (confusion matrices)
    - model_comparison.png     (bar chart comparison)
    - feature_importance_rf.png  (Random Forest importances)
""")
if XGBOOST_AVAILABLE:
    print(f"    - feature_importance_xgb.png (XGBoost importances)")

print("\n" + "=" * 80)
print("  PIPELINE EXECUTION COMPLETE!")
print("=" * 80)
