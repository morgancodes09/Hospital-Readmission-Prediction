"""
===============================================================================
HOSPITAL READMISSION PREDICTION - GUI APPLICATION (Tkinter)
===============================================================================
Desktop GUI with buttons, graphs, prediction, and systematic output display.
===============================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================
import warnings
warnings.filterwarnings('ignore')

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import os, joblib, time, threading

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

# Detect OneHotEncoder sparse parameter name
_sklearn_version = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
_OHE_SPARSE_PARAM = 'sparse_output' if _sklearn_version >= (1, 2) else 'sparse'

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

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
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================================
# MAIN APPLICATION CLASS
# ============================================================================
class HospitalReadmissionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hospital Readmission Prediction - ML Pipeline")
        self.root.geometry("1100x750")
        self.root.configure(bg='#1e1e2e')
        self.root.resizable(True, True)

        # State variables
        self.models_trained = False
        self.results = {}
        self.all_feature_names = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model_name = None
        self.best_model_obj = None
        self.preprocessor_scaled = None
        self.preprocessor_unscaled = None
        self.categorical_features = []
        self.numerical_features = []
        self.scale_pos_weight = 1.0
        self.df = None

        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('Title.TLabel', font=('Segoe UI', 18, 'bold'),
                           foreground='#cdd6f4', background='#1e1e2e')
        self.style.configure('Sub.TLabel', font=('Segoe UI', 10),
                           foreground='#a6adc8', background='#1e1e2e')
        self.style.configure('Action.TButton', font=('Segoe UI', 11, 'bold'),
                           padding=(15, 8))
        self.style.configure('Small.TButton', font=('Segoe UI', 9),
                           padding=(10, 5))
        self.style.configure('Result.TLabel', font=('Consolas', 10),
                           foreground='#a6e3a1', background='#313244')
        self.style.configure('Card.TFrame', background='#313244')
        self.style.configure('Input.TLabel', font=('Segoe UI', 10, 'bold'),
                           foreground='#cdd6f4', background='#313244')

        self._build_ui()
        self._load_saved_model_if_available()

    # ========================================================================
    # UI CONSTRUCTION
    # ========================================================================
    def _build_ui(self):
        # --- Title Bar ---
        title_frame = tk.Frame(self.root, bg='#181825', pady=10)
        title_frame.pack(fill='x')
        tk.Label(title_frame, text="🏥  Hospital Readmission Prediction",
                font=('Segoe UI', 20, 'bold'), fg='#89b4fa', bg='#181825').pack()
        tk.Label(title_frame, text="End-to-End ML Pipeline  |  Logistic Regression • Random Forest • XGBoost • ANN",
                font=('Segoe UI', 10), fg='#6c7086', bg='#181825').pack()

        # --- Main Container ---
        main_frame = tk.Frame(self.root, bg='#1e1e2e')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # LEFT PANEL: Buttons + Prediction
        left_frame = tk.Frame(main_frame, bg='#1e1e2e', width=320)
        left_frame.pack(side='left', fill='y', padx=(0, 10))
        left_frame.pack_propagate(False)

        # RIGHT PANEL: Output Log
        right_frame = tk.Frame(main_frame, bg='#1e1e2e')
        right_frame.pack(side='right', fill='both', expand=True)

        # ---- LEFT: Action Buttons ----
        btn_card = tk.Frame(left_frame, bg='#313244', bd=0, highlightthickness=1,
                           highlightbackground='#45475a')
        btn_card.pack(fill='x', pady=(0, 8))

        tk.Label(btn_card, text="⚡ Pipeline Controls", font=('Segoe UI', 13, 'bold'),
                fg='#f5c2e7', bg='#313244', pady=8).pack(fill='x')

        self.btn_load = tk.Button(btn_card, text="📂  Load & Preprocess Data",
                                 command=self._run_load_data,
                                 font=('Segoe UI', 11, 'bold'), bg='#89b4fa', fg='#1e1e2e',
                                 activebackground='#74c7ec', cursor='hand2',
                                 relief='flat', pady=8)
        self.btn_load.pack(fill='x', padx=12, pady=4)

        self.btn_train = tk.Button(btn_card, text="🧠  Train All Models",
                                  command=self._run_train_models,
                                  font=('Segoe UI', 11, 'bold'), bg='#a6e3a1', fg='#1e1e2e',
                                  activebackground='#94e2d5', cursor='hand2',
                                  relief='flat', pady=8, state='disabled')
        self.btn_train.pack(fill='x', padx=12, pady=4)

        self.btn_load_model = tk.Button(
            btn_card, text="📦  Load Saved Model", command=self._load_saved_model,
            font=('Segoe UI', 10, 'bold'), bg='#cba6f7', fg='#1e1e2e',
            activebackground='#b4befe', cursor='hand2', relief='flat', pady=7
        )
        self.btn_load_model.pack(fill='x', padx=12, pady=4)

        tk.Frame(btn_card, bg='#45475a', height=1).pack(fill='x', padx=12, pady=6)

        tk.Label(btn_card, text="📊 View Graphs", font=('Segoe UI', 11, 'bold'),
                fg='#fab387', bg='#313244', pady=4).pack(fill='x')

        graph_buttons = [
            ("ROC Curves", self._show_roc_curves),
            ("Confusion Matrices", self._show_confusion_matrices),
            ("Model Comparison", self._show_model_comparison),
            ("Feature Importance (RF)", self._show_feature_importance_rf),
            ("Feature Importance (XGB)", self._show_feature_importance_xgb),
        ]
        for text, cmd in graph_buttons:
            b = tk.Button(btn_card, text=f"  📈  {text}", command=cmd,
                         font=('Segoe UI', 9), bg='#45475a', fg='#cdd6f4',
                         activebackground='#585b70', cursor='hand2',
                         relief='flat', anchor='w', pady=4, state='disabled')
            b.pack(fill='x', padx=12, pady=2)
            if not hasattr(self, '_graph_buttons'):
                self._graph_buttons = []
            self._graph_buttons.append(b)

        tk.Frame(btn_card, bg='#313244', height=8).pack()

        # ---- LEFT: Prediction Card ----
        pred_card = tk.Frame(left_frame, bg='#313244', bd=0, highlightthickness=1,
                            highlightbackground='#45475a')
        pred_card.pack(fill='x', pady=(0, 8))

        tk.Label(pred_card, text="🔮 Predict Readmission", font=('Segoe UI', 13, 'bold'),
                fg='#f5c2e7', bg='#313244', pady=8).pack(fill='x')

        # Input fields
        fields_frame = tk.Frame(pred_card, bg='#313244')
        fields_frame.pack(fill='x', padx=12)

        self.input_vars = {}
        input_fields = [
            ("Age", "65"),
            ("Length of Stay", "5"),
            ("Num Medications", "7"),
            ("Prev Admissions", "2"),
            ("BMI", "28.5"),
        ]
        for label_text, default_val in input_fields:
            row = tk.Frame(fields_frame, bg='#313244')
            row.pack(fill='x', pady=2)
            tk.Label(row, text=label_text, font=('Segoe UI', 9, 'bold'),
                    fg='#bac2de', bg='#313244', width=16, anchor='w').pack(side='left')
            var = tk.StringVar(value=default_val)
            entry = tk.Entry(row, textvariable=var, font=('Consolas', 10),
                           bg='#45475a', fg='#cdd6f4', insertbackground='#cdd6f4',
                           relief='flat', width=10)
            entry.pack(side='left', padx=(4, 0))
            self.input_vars[label_text] = var

        # Dropdowns
        drop_frame = tk.Frame(fields_frame, bg='#313244')
        drop_frame.pack(fill='x', pady=2)
        tk.Label(drop_frame, text="Diagnosis", font=('Segoe UI', 9, 'bold'),
                fg='#bac2de', bg='#313244', width=16, anchor='w').pack(side='left')
        self.diagnosis_var = tk.StringVar(value='cardiac')
        diag_combo = ttk.Combobox(drop_frame, textvariable=self.diagnosis_var,
                                 values=['cardiac', 'respiratory', 'diabetes', 'surgical', 'other'],
                                 font=('Consolas', 10), width=12, state='readonly')
        diag_combo.pack(side='left', padx=(4, 0))

        drop_frame2 = tk.Frame(fields_frame, bg='#313244')
        drop_frame2.pack(fill='x', pady=2)
        tk.Label(drop_frame2, text="Discharge Type", font=('Segoe UI', 9, 'bold'),
                fg='#bac2de', bg='#313244', width=16, anchor='w').pack(side='left')
        self.discharge_var = tk.StringVar(value='home')
        disc_combo = ttk.Combobox(drop_frame2, textvariable=self.discharge_var,
                                 values=['home', 'rehab', 'another_facility'],
                                 font=('Consolas', 10), width=12, state='readonly')
        disc_combo.pack(side='left', padx=(4, 0))

        tk.Frame(pred_card, bg='#313244', height=6).pack()

        self.btn_predict = tk.Button(pred_card, text="⚡ Predict",
                                    command=self._predict,
                                    font=('Segoe UI', 11, 'bold'), bg='#f38ba8', fg='#1e1e2e',
                                    activebackground='#eba0ac', cursor='hand2',
                                    relief='flat', pady=6, state='disabled')
        self.btn_predict.pack(fill='x', padx=12, pady=(0, 4))

        # Prediction result
        self.pred_result_label = tk.Label(pred_card, text="", font=('Segoe UI', 12, 'bold'),
                                         fg='#a6e3a1', bg='#313244', pady=6,
                                         wraplength=280)
        self.pred_result_label.pack(fill='x', padx=12)

        tk.Frame(pred_card, bg='#313244', height=6).pack()

        # ---- RIGHT: Output Log ----
        log_card = tk.Frame(right_frame, bg='#313244', bd=0, highlightthickness=1,
                           highlightbackground='#45475a')
        log_card.pack(fill='both', expand=True)

        log_header = tk.Frame(log_card, bg='#313244')
        log_header.pack(fill='x')
        tk.Label(log_header, text="📋 Pipeline Output Log", font=('Segoe UI', 13, 'bold'),
                fg='#f5c2e7', bg='#313244', pady=8).pack(side='left', padx=12)

        self.btn_clear = tk.Button(log_header, text="Clear", command=self._clear_log,
                                  font=('Segoe UI', 8), bg='#45475a', fg='#a6adc8',
                                  relief='flat', cursor='hand2')
        self.btn_clear.pack(side='right', padx=12)

        self.output_text = scrolledtext.ScrolledText(
            log_card, wrap='word', font=('Consolas', 10),
            bg='#11111b', fg='#cdd6f4', insertbackground='#cdd6f4',
            relief='flat', padx=10, pady=8, state='disabled'
        )
        self.output_text.pack(fill='both', expand=True, padx=8, pady=(0, 8))

        # Configure text tags for colored output
        self.output_text.tag_config('header', foreground='#89b4fa', font=('Consolas', 11, 'bold'))
        self.output_text.tag_config('success', foreground='#a6e3a1')
        self.output_text.tag_config('warning', foreground='#fab387')
        self.output_text.tag_config('error', foreground='#f38ba8')
        self.output_text.tag_config('info', foreground='#cdd6f4')
        self.output_text.tag_config('highlight', foreground='#f5c2e7', font=('Consolas', 11, 'bold'))
        self.output_text.tag_config('metric', foreground='#94e2d5')
        self.output_text.tag_config('dim', foreground='#6c7086')

        # --- Status Bar ---
        self.status_var = tk.StringVar(value="Ready. Click 'Load & Preprocess Data' to start.")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                             font=('Segoe UI', 9), fg='#6c7086', bg='#181825',
                             anchor='w', padx=10, pady=4)
        status_bar.pack(fill='x', side='bottom')

    # ========================================================================
    # LOGGING HELPERS
    # ========================================================================
    def _log(self, text, tag='info'):
        self.output_text.config(state='normal')
        self.output_text.insert('end', text + '\n', tag)
        self.output_text.see('end')
        self.output_text.config(state='disabled')
        self.root.update_idletasks()

    def _log_separator(self):
        self._log("=" * 72, 'dim')

    def _clear_log(self):
        self.output_text.config(state='normal')
        self.output_text.delete('1.0', 'end')
        self.output_text.config(state='disabled')

    # ========================================================================
    # STEP 1: LOAD & PREPROCESS DATA
    # ========================================================================
    def _run_load_data(self):
        self.btn_load.config(state='disabled', text="⏳ Loading...")
        self.status_var.set("Loading and preprocessing data...")
        self.root.update_idletasks()
        # Run in thread to keep UI responsive
        threading.Thread(target=self._load_data_thread, daemon=True).start()

    def _load_data_thread(self):
        try:
            self._do_load_data()
            self.root.after(0, self._on_load_complete)
        except Exception as e:
            self.root.after(0, lambda: self._on_error(f"Load Error: {e}"))

    def _do_load_data(self):
        # STEP 1: Load
        self.root.after(0, lambda: self._log("\n  STEP 1 : LOADING DATA", 'header'))
        self.root.after(0, self._log_separator)

        self.df = pd.read_csv(DATA_PATH)
        df = self.df.copy()

        self.root.after(0, lambda: self._log(f"  Shape        : {df.shape[0]} rows x {df.shape[1]} columns", 'info'))
        self.root.after(0, lambda: self._log(f"  Columns      : {list(df.columns)}", 'info'))

        target_counts = df['readmitted_30d'].value_counts()
        for val, cnt in target_counts.items():
            self.root.after(0, lambda v=val, c=cnt: self._log(
                f"  Class {v}      : {c}  ({c / len(df) * 100:.1f}%)", 'info'))

        self.root.after(0, lambda: self._log(f"  Missing vals : {df.isnull().sum().sum()}", 'info'))

        # STEP 2: Preprocess
        self.root.after(0, lambda: self._log("\n  STEP 2 : DATA PREPROCESSING", 'header'))
        self.root.after(0, self._log_separator)

        df = df.drop(columns=['id'])
        self.root.after(0, lambda: self._log("  [2.1] Dropped 'id' column", 'success'))

        numeric_cols = ['age', 'length_of_stay', 'num_medications', 'prev_admissions', 'bmi']
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df.groupby('diagnosis')[col].transform(lambda x: x.fillna(x.median()))
                df[col] = df[col].fillna(df[col].median())
                self.root.after(0, lambda c=col: self._log(
                    f"  [2.2] Imputed '{c}' (group-wise median)", 'success'))

        if df.isnull().sum().sum() == 0:
            self.root.after(0, lambda: self._log("  [2.2] No missing values - data clean ✓", 'success'))

        # STEP 3: Feature Engineering
        self.root.after(0, lambda: self._log("\n  STEP 3 : FEATURE ENGINEERING", 'header'))
        self.root.after(0, self._log_separator)

        df['age_group'] = df['age'].apply(
            lambda a: '18-40' if a <= 40 else ('41-60' if a <= 60 else '61+'))
        self.root.after(0, lambda: self._log("  [3.1] Created 'age_group' : 18-40, 41-60, 61+", 'success'))

        df['high_risk_flag'] = (df['prev_admissions'] >= 2).astype(int)
        self.root.after(0, lambda: self._log("  [3.2] Created 'high_risk_flag' (prev_admissions >= 2)", 'success'))

        df['medication_per_day'] = df['num_medications'] / df['length_of_stay'].replace(0, np.nan)
        df['medication_per_day'] = df['medication_per_day'].replace([np.inf, -np.inf], np.nan)
        df['medication_per_day'] = df['medication_per_day'].fillna(df['medication_per_day'].median())
        self.root.after(0, lambda: self._log("  [3.3] Created 'medication_per_day'", 'success'))

        # STEP 4: Split
        self.root.after(0, lambda: self._log("\n  STEP 4 : STRATIFIED TRAIN-TEST SPLIT (80/20)", 'header'))
        self.root.after(0, self._log_separator)

        TARGET = 'readmitted_30d'
        X = df.drop(columns=[TARGET])
        y = df[TARGET]

        self.categorical_features = ['diagnosis', 'discharge_type', 'age_group']
        self.numerical_features = [c for c in X.columns if c not in self.categorical_features]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

        neg = (self.y_train == 0).sum()
        pos = (self.y_train == 1).sum()
        self.scale_pos_weight = neg / pos

        self.root.after(0, lambda: self._log(f"  Train set    : {self.X_train.shape[0]} samples", 'info'))
        self.root.after(0, lambda: self._log(f"  Test set     : {self.X_test.shape[0]} samples", 'info'))
        self.root.after(0, lambda: self._log(f"  Train target : 0={neg} | 1={pos}", 'info'))
        self.root.after(0, lambda: self._log(f"  Imbalance    : {self.scale_pos_weight:.2f}x", 'warning'))

        # STEP 5: Build preprocessors
        self.root.after(0, lambda: self._log("\n  STEP 5 : BUILDING PREPROCESSORS", 'header'))
        self.root.after(0, self._log_separator)

        _ohe_kwargs = {'drop': 'first', 'handle_unknown': 'ignore', _OHE_SPARSE_PARAM: False}

        self.preprocessor_scaled = ColumnTransformer(transformers=[
            ('num', StandardScaler(), self.numerical_features),
            ('cat', OneHotEncoder(**_ohe_kwargs), self.categorical_features)
        ], remainder='drop')

        self.preprocessor_unscaled = ColumnTransformer(transformers=[
            ('num', 'passthrough', self.numerical_features),
            ('cat', OneHotEncoder(**_ohe_kwargs), self.categorical_features)
        ], remainder='drop')

        self.preprocessor_unscaled.fit(self.X_train)
        self.all_feature_names = list(self.preprocessor_unscaled.get_feature_names_out())
        self.all_feature_names = [n.split('__', 1)[-1] if '__' in n else n
                                  for n in self.all_feature_names]

        self.root.after(0, lambda: self._log(
            f"  Preprocessor A (Scaled)   -> LR, ANN", 'success'))
        self.root.after(0, lambda: self._log(
            f"  Preprocessor B (Unscaled) -> RF, XGBoost", 'success'))
        self.root.after(0, lambda: self._log(
            f"  Features after encoding   : {len(self.all_feature_names)}", 'info'))
        self.root.after(0, lambda: self._log(
            f"  {self.all_feature_names}", 'dim'))

        self.df = df

    def _on_load_complete(self):
        self._log("\n  ✅ Data loaded and preprocessed successfully!", 'highlight')
        self._log("  Click 'Train All Models' to start training.\n", 'dim')
        self.btn_load.config(state='normal', text="📂  Load & Preprocess Data")
        self.btn_train.config(state='normal')
        self.status_var.set("Data ready. Click 'Train All Models' to proceed.")

    def _on_error(self, msg):
        self._log(f"\n  ❌ {msg}", 'error')
        self.btn_load.config(state='normal', text="📂  Load & Preprocess Data")
        self.status_var.set(f"Error: {msg}")

    # ========================================================================
    # STEP 6: TRAIN ALL MODELS
    # ========================================================================
    def _run_train_models(self):
        self.btn_train.config(state='disabled', text="⏳ Training...")
        self.btn_load.config(state='disabled')
        self.status_var.set("Training models... This may take 1-2 minutes.")
        self.root.update_idletasks()
        threading.Thread(target=self._train_models_thread, daemon=True).start()

    def _train_models_thread(self):
        try:
            self._do_train_models()
            self.root.after(0, self._on_train_complete)
        except Exception as e:
            self.root.after(0, lambda: self._on_train_error(str(e)))

    def _do_train_models(self):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        self.results = {}

        # ---- MODEL 1: LOGISTIC REGRESSION ----
        self.root.after(0, lambda: self._log("\n  MODEL 1 : LOGISTIC REGRESSION", 'header'))
        self.root.after(0, self._log_separator)
        self.root.after(0, lambda: self._log("  Running GridSearchCV (5-fold)...", 'dim'))
        self.root.after(0, lambda: self.status_var.set("Training Logistic Regression..."))

        lr_pipeline = Pipeline([
            ('preprocessor', self.preprocessor_scaled),
            ('classifier', LogisticRegression(class_weight='balanced',
                                              random_state=RANDOM_STATE, max_iter=1000))
        ])
        lr_param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear'],
        }
        lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=cv, scoring='roc_auc',
                               n_jobs=2, verbose=0, refit=True)
        lr_grid.fit(self.X_train, self.y_train)

        lr_model = lr_grid.best_estimator_
        y_pred = lr_model.predict(self.X_test)
        y_prob = lr_model.predict_proba(self.X_test)[:, 1]
        lr_m = self._calc_metrics(y_pred, y_prob)
        lr_cm = confusion_matrix(self.y_test, y_pred)
        self.results['Logistic Regression'] = {
            'model': lr_model, 'y_pred': y_pred, 'y_prob': y_prob,
            'metrics': lr_m, 'cm': lr_cm, 'best_params': lr_grid.best_params_,
            'cv_score': lr_grid.best_score_
        }
        self.root.after(0, lambda: self._print_model_results('Logistic Regression'))

        # ---- MODEL 2: RANDOM FOREST ----
        self.root.after(0, lambda: self._log("\n  MODEL 2 : RANDOM FOREST", 'header'))
        self.root.after(0, self._log_separator)
        self.root.after(0, lambda: self._log("  Running GridSearchCV (5-fold)...", 'dim'))
        self.root.after(0, lambda: self.status_var.set("Training Random Forest..."))

        rf_pipeline = Pipeline([
            ('preprocessor', self.preprocessor_unscaled),
            ('classifier', RandomForestClassifier(class_weight='balanced',
                                                  random_state=RANDOM_STATE, n_jobs=1))
        ])
        rf_param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [5, 10, None],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2],
        }
        rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=cv, scoring='roc_auc',
                               n_jobs=2, verbose=0, refit=True)
        rf_grid.fit(self.X_train, self.y_train)

        rf_model = rf_grid.best_estimator_
        y_pred = rf_model.predict(self.X_test)
        y_prob = rf_model.predict_proba(self.X_test)[:, 1]
        rf_m = self._calc_metrics(y_pred, y_prob)
        rf_cm = confusion_matrix(self.y_test, y_pred)
        self.results['Random Forest'] = {
            'model': rf_model, 'y_pred': y_pred, 'y_prob': y_prob,
            'metrics': rf_m, 'cm': rf_cm, 'best_params': rf_grid.best_params_,
            'cv_score': rf_grid.best_score_
        }
        self.root.after(0, lambda: self._print_model_results('Random Forest'))

        # ---- MODEL 3: XGBOOST ----
        if XGBOOST_AVAILABLE:
            self.root.after(0, lambda: self._log("\n  MODEL 3 : XGBOOST", 'header'))
            self.root.after(0, self._log_separator)
            self.root.after(0, lambda: self._log("  Running RandomizedSearchCV (5-fold, 12 iter)...", 'dim'))
            self.root.after(0, lambda: self.status_var.set("Training XGBoost..."))

            xgb_pipeline = Pipeline([
                ('preprocessor', self.preprocessor_unscaled),
                ('classifier', XGBClassifier(scale_pos_weight=self.scale_pos_weight,
                                             eval_metric='logloss',
                                             random_state=RANDOM_STATE, n_jobs=1))
            ])
            xgb_param_dist = {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'classifier__subsample': [0.6, 0.8, 1.0],
                'classifier__colsample_bytree': [0.6, 0.8, 1.0],
                'classifier__min_child_weight': [1, 3, 5],
                'classifier__gamma': [0, 0.1, 0.3],
            }
            xgb_random = RandomizedSearchCV(xgb_pipeline, xgb_param_dist,
                                            n_iter=12, cv=cv, scoring='roc_auc',
                                            n_jobs=2, verbose=0, refit=True,
                                            random_state=RANDOM_STATE)
            xgb_random.fit(self.X_train, self.y_train)

            xgb_model = xgb_random.best_estimator_
            y_pred = xgb_model.predict(self.X_test)
            y_prob = xgb_model.predict_proba(self.X_test)[:, 1]
            xgb_m = self._calc_metrics(y_pred, y_prob)
            xgb_cm = confusion_matrix(self.y_test, y_pred)
            self.results['XGBoost'] = {
                'model': xgb_model, 'y_pred': y_pred, 'y_prob': y_prob,
                'metrics': xgb_m, 'cm': xgb_cm, 'best_params': xgb_random.best_params_,
                'cv_score': xgb_random.best_score_
            }
            self.root.after(0, lambda: self._print_model_results('XGBoost'))

        # ---- MODEL 4: ANN ----
        self.root.after(0, lambda: self._log("\n  MODEL 4 : ANN (MLPClassifier)", 'header'))
        self.root.after(0, self._log_separator)
        self.root.after(0, lambda: self._log("  Running RandomizedSearchCV (5-fold, 10 iter)...", 'dim'))
        self.root.after(0, lambda: self._log("  [*] sample_weight='balanced' for imbalance", 'dim'))
        self.root.after(0, lambda: self.status_var.set("Training ANN (MLPClassifier)..."))

        ann_sample_weights = compute_sample_weight(class_weight='balanced', y=self.y_train)
        self.preprocessor_scaled.fit(self.X_train)
        X_train_scaled = self.preprocessor_scaled.transform(self.X_train)
        X_test_scaled = self.preprocessor_scaled.transform(self.X_test)

        ann_base = MLPClassifier(random_state=RANDOM_STATE, max_iter=500,
                                 early_stopping=True, validation_fraction=0.15)
        ann_param_dist = {
            'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64),
                                   (128, 64, 32), (64, 32, 16)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': ['constant', 'adaptive'],
            'learning_rate_init': [0.001, 0.01, 0.005],
            'batch_size': [32, 64, 128],
        }
        ann_random = RandomizedSearchCV(ann_base, ann_param_dist,
                                        n_iter=10, cv=cv, scoring='roc_auc',
                                        n_jobs=2, verbose=0, refit=True,
                                        random_state=RANDOM_STATE)
        ann_random.fit(X_train_scaled, self.y_train, sample_weight=ann_sample_weights)

        ann_clf_best = ann_random.best_estimator_
        y_pred = ann_clf_best.predict(X_test_scaled)
        y_prob = ann_clf_best.predict_proba(X_test_scaled)[:, 1]

        ann_model = Pipeline([
            ('preprocessor', self.preprocessor_scaled),
            ('classifier', ann_clf_best)
        ])
        ann_m = self._calc_metrics(y_pred, y_prob)
        ann_cm = confusion_matrix(self.y_test, y_pred)
        self.results['ANN (MLP)'] = {
            'model': ann_model, 'y_pred': y_pred, 'y_prob': y_prob,
            'metrics': ann_m, 'cm': ann_cm, 'best_params': ann_random.best_params_,
            'cv_score': ann_random.best_score_
        }
        self.root.after(0, lambda: self._print_model_results('ANN (MLP)'))

    def _calc_metrics(self, y_pred, y_prob):
        return {
            'ROC-AUC': roc_auc_score(self.y_test, y_prob),
            'F1': f1_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred, zero_division=0),
            'Recall': recall_score(self.y_test, y_pred),
            'Accuracy': accuracy_score(self.y_test, y_pred),
        }

    def _print_model_results(self, name):
        res = self.results[name]
        self._log(f"  Best CV ROC-AUC : {res['cv_score']:.4f}", 'success')
        self._log(f"  ┌──────────────┬──────────┐", 'dim')
        self._log(f"  │ Metric       │  Score   │", 'dim')
        self._log(f"  ├──────────────┼──────────┤", 'dim')
        for k, v in res['metrics'].items():
            self._log(f"  │ {k:<12s} │  {v:.4f}  │", 'metric')
        self._log(f"  └──────────────┴──────────┘", 'dim')
        cm = res['cm']
        self._log(f"  Confusion Matrix: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}", 'info')

    def _on_train_complete(self):
        self.models_trained = True

        # Print comparison table
        self._log("\n  MODEL COMPARISON (sorted by ROC-AUC)", 'header')
        self._log_separator()

        sorted_models = sorted(self.results.items(),
                              key=lambda x: x[1]['metrics']['ROC-AUC'], reverse=True)

        self._log(f"  ┌──────────────────────┬──────────┬──────────┬──────────┬──────────┐", 'dim')
        self._log(f"  │ Model                │ ROC-AUC  │    F1    │ Precision│  Recall  │", 'dim')
        self._log(f"  ├──────────────────────┼──────────┼──────────┼──────────┼──────────┤", 'dim')

        for i, (name, res) in enumerate(sorted_models):
            m = res['metrics']
            tag = 'highlight' if i == 0 else 'metric'
            marker = " 🏆" if i == 0 else "   "
            self._log(f"  │ {name:<20s} │  {m['ROC-AUC']:.4f}  │  {m['F1']:.4f}  │  {m['Precision']:.4f}  │  {m['Recall']:.4f}  │{marker}", tag)

        self._log(f"  └──────────────────────┴──────────┴──────────┴──────────┴──────────┘", 'dim')

        self.best_model_name = sorted_models[0][0]
        self.best_model_obj = sorted_models[0][1]['model']
        best_auc = sorted_models[0][1]['metrics']['ROC-AUC']
        best_f1 = sorted_models[0][1]['metrics']['F1']

        self._log(f"\n  🏆 BEST MODEL: {self.best_model_name}", 'highlight')
        self._log(f"     ROC-AUC = {best_auc:.4f}  |  F1 = {best_f1:.4f}", 'success')

        # Save model
        model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
        joblib.dump(self.best_model_obj, model_path)
        joblib.dump(self.best_model_obj, os.path.join(OUTPUT_DIR, 'best_model.pkl'))
        self._log(f"\n  💾 Model saved to: {model_path}", 'success')

        # Feature importance
        self._log("\n  KEY FACTORS INFLUENCING READMISSION:", 'header')
        self._log_separator()
        rf_clf = self.results['Random Forest']['model'].named_steps['classifier']
        importances = rf_clf.feature_importances_
        feat_imp = sorted(zip(self.all_feature_names, importances),
                         key=lambda x: x[1], reverse=True)
        for fname, imp in feat_imp[:5]:
            bar = '█' * int(imp * 80)
            self._log(f"  {fname:<25s} {imp:.4f}  {bar}", 'metric')

        self._log("\n  ✅ All models trained! Use buttons to view graphs or make predictions.", 'highlight')

        # Enable buttons
        self.btn_train.config(state='normal', text="🧠  Train All Models")
        self.btn_load.config(state='normal')
        self.btn_predict.config(state='normal')
        for b in self._graph_buttons:
            b.config(state='normal')
        self.status_var.set(f"Training complete! Best model: {self.best_model_name} (ROC-AUC={best_auc:.4f})")

    def _on_train_error(self, msg):
        self._log(f"\n  ❌ Training Error: {msg}", 'error')
        self.btn_train.config(state='normal', text="🧠  Train All Models")
        self.btn_load.config(state='normal')
        self.status_var.set(f"Error: {msg}")

    # ========================================================================
    # GRAPH DISPLAY FUNCTIONS
    # ========================================================================
    def _show_graph_window(self, title, fig):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.configure(bg='#1e1e2e')
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

    def _show_roc_curves(self):
        if not self.models_trained:
            messagebox.showwarning("Warning", "Train models first!")
            return
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']
        for i, (name, res) in enumerate(self.results.items()):
            fpr, tpr, _ = roc_curve(self.y_test, res['y_prob'])
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2.5,
                    label=f"{name} (AUC = {res['metrics']['ROC-AUC']:.4f})")
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random Baseline')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - All Models', fontsize=15, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves.png'), dpi=150, bbox_inches='tight')
        self._show_graph_window("ROC Curves", fig)

    def _show_confusion_matrices(self):
        if not self.models_trained:
            messagebox.showwarning("Warning", "Train models first!")
            return
        n = len(self.results)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
        if n == 1:
            axes = [axes]
        for i, (name, res) in enumerate(self.results.items()):
            sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Not Readm.', 'Readmitted'],
                        yticklabels=['Not Readm.', 'Readmitted'],
                        ax=axes[i], cbar=False, annot_kws={'size': 14})
            axes[i].set_title(name, fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Actual' if i == 0 else '')
            axes[i].set_xlabel('Predicted')
        plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
        self._show_graph_window("Confusion Matrices", fig)

    def _show_model_comparison(self):
        if not self.models_trained:
            messagebox.showwarning("Warning", "Train models first!")
            return
        metrics = ['ROC-AUC', 'F1', 'Precision', 'Recall']
        x = np.arange(len(metrics))
        width = 0.18
        colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for i, (name, res) in enumerate(self.results.items()):
            vals = [res['metrics'][m] for m in metrics]
            bars = ax.bar(x + i * width, vals, width, label=name,
                         color=colors[i % len(colors)], edgecolor='white')
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_xticks(x + width * (len(self.results)-1)/2)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=15, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
        self._show_graph_window("Model Comparison", fig)

    def _show_feature_importance_rf(self):
        if not self.models_trained:
            messagebox.showwarning("Warning", "Train models first!")
            return
        rf_clf = self.results['Random Forest']['model'].named_steps['classifier']
        imp_df = pd.DataFrame({'Feature': self.all_feature_names,
                              'Importance': rf_clf.feature_importances_}
                             ).sort_values('Importance', ascending=False)
        fig, ax = plt.subplots(figsize=(9, 6))
        top = imp_df.head(15).iloc[::-1]
        bars = ax.barh(top['Feature'], top['Importance'],
                       color=sns.color_palette('viridis', len(top)), edgecolor='white')
        for bar, val in zip(bars, top['Importance']):
            ax.text(val+0.002, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Feature Importance - Random Forest', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_rf.png'), dpi=150, bbox_inches='tight')
        self._show_graph_window("Feature Importance - Random Forest", fig)

    def _show_feature_importance_xgb(self):
        if not self.models_trained or 'XGBoost' not in self.results:
            messagebox.showwarning("Warning", "XGBoost model not available!")
            return
        xgb_clf = self.results['XGBoost']['model'].named_steps['classifier']
        imp_df = pd.DataFrame({'Feature': self.all_feature_names,
                              'Importance': xgb_clf.feature_importances_}
                             ).sort_values('Importance', ascending=False)
        fig, ax = plt.subplots(figsize=(9, 6))
        top = imp_df.head(15).iloc[::-1]
        bars = ax.barh(top['Feature'], top['Importance'],
                       color=sns.color_palette('magma', len(top)), edgecolor='white')
        for bar, val in zip(bars, top['Importance']):
            ax.text(val+0.002, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Feature Importance - XGBoost', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_xgb.png'), dpi=150, bbox_inches='tight')
        self._show_graph_window("Feature Importance - XGBoost", fig)

    # ========================================================================
    # SAVED MODEL LOADING
    # ========================================================================
    def _model_candidates(self):
        return [
            os.path.join(MODEL_DIR, 'best_model.pkl'),
            os.path.join(OUTPUT_DIR, 'best_model.pkl'),
            os.path.join(BASE_DIR, 'best_model.pkl'),
        ]

    def _load_saved_model_if_available(self):
        for model_path in self._model_candidates():
            if os.path.isfile(model_path):
                try:
                    self._set_loaded_model(model_path)
                    return
                except Exception as exc:
                    self._log(f"  Saved model could not be loaded: {exc}", 'warning')

    def _load_saved_model(self):
        for model_path in self._model_candidates():
            if os.path.isfile(model_path):
                try:
                    self._set_loaded_model(model_path)
                    messagebox.showinfo('Model loaded', f'Saved model loaded successfully:\n{model_path}')
                    return
                except Exception as exc:
                    messagebox.showerror('Model loading error', str(exc))
                    return
        messagebox.showwarning(
            'Model not found',
            'No saved model found. Train the models first or place best_model.pkl in models/.'
        )

    def _set_loaded_model(self, model_path):
        model = joblib.load(model_path)
        if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
            raise TypeError('The saved file is not a supported classification model.')
        self.best_model_obj = model
        classifier = model.named_steps.get('classifier') if hasattr(model, 'named_steps') else model
        class_name = classifier.__class__.__name__
        name_map = {
            'LogisticRegression': 'Logistic Regression',
            'RandomForestClassifier': 'Random Forest',
            'XGBClassifier': 'XGBoost',
            'MLPClassifier': 'ANN (MLP)',
        }
        self.best_model_name = name_map.get(class_name, class_name)
        self.btn_predict.config(state='normal')
        self.status_var.set(f'Saved model ready: {self.best_model_name}')
        self._log(f"\n  [OK] Loaded saved model: {self.best_model_name}", 'success')

    # ========================================================================
    # PREDICTION
    # ========================================================================
    def _predict(self):
        if self.best_model_obj is None:
            messagebox.showwarning("Model unavailable", "Load a saved model or train the models first.")
            return
        try:
            age = float(self.input_vars["Age"].get())
            los = float(self.input_vars["Length of Stay"].get())
            meds = float(self.input_vars["Num Medications"].get())
            prev = float(self.input_vars["Prev Admissions"].get())
            bmi = float(self.input_vars["BMI"].get())
            diagnosis = self.diagnosis_var.get()
            discharge = self.discharge_var.get()

            if not (18 <= age <= 120):
                raise ValueError('Age must be between 18 and 120 years.')
            if not (1 <= los <= 365):
                raise ValueError('Length of stay must be between 1 and 365 days.')
            if not (0 <= meds <= 100):
                raise ValueError('Number of medications must be between 0 and 100.')
            if not (0 <= prev <= 100):
                raise ValueError('Previous admissions must be between 0 and 100.')
            if not (10 <= bmi <= 80):
                raise ValueError('BMI must be between 10 and 80.')

            # Engineered features
            age_group = '18-40' if age <= 40 else ('41-60' if age <= 60 else '61+')
            high_risk = 1 if prev >= 2 else 0
            med_per_day = meds / los if los > 0 else 0

            input_data = pd.DataFrame([{
                'age': age, 'length_of_stay': los, 'num_medications': meds,
                'prev_admissions': prev, 'bmi': bmi, 'high_risk_flag': high_risk,
                'medication_per_day': med_per_day, 'diagnosis': diagnosis,
                'discharge_type': discharge, 'age_group': age_group
            }])

            # Predict with best model
            pred = self.best_model_obj.predict(input_data)[0]
            prob = self.best_model_obj.predict_proba(input_data)[0]
            classes = list(self.best_model_obj.classes_)
            positive_index = classes.index(1)
            readmission_probability = float(prob[positive_index])

            if pred == 1:
                result_text = f"⚠️ HIGH RISK\nReadmission Probability: {readmission_probability*100:.1f}%"
                self.pred_result_label.config(fg='#f38ba8')
            else:
                result_text = f"✅ LOW RISK\nReadmission Probability: {readmission_probability*100:.1f}%"
                self.pred_result_label.config(fg='#a6e3a1')

            self.pred_result_label.config(text=result_text)

            # Log
            self._log(f"\n  🔮 PREDICTION ({self.best_model_name}):", 'header')
            self._log(f"  Age={age}, LOS={los}, Meds={meds}, Prev={prev}, BMI={bmi}", 'info')
            self._log(f"  Diagnosis={diagnosis}, Discharge={discharge}", 'info')
            self._log(f"  Result: {'READMITTED (HIGH RISK)' if pred == 1 else 'NOT READMITTED (LOW RISK)'}", 
                      'error' if pred == 1 else 'success')
            self._log(f"  Probability: {readmission_probability*100:.1f}%", 'metric')

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Could not generate prediction: {e}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    root = tk.Tk()
    app = HospitalReadmissionApp(root)
    root.mainloop()
