# Hospital Readmission Prediction

An end-to-end machine-learning project that estimates whether a patient may be readmitted to a hospital within 30 days of discharge. It compares four binary-classification models and provides a Tkinter desktop interface for entering patient attributes and generating a model prediction.

> **Educational/research demonstration only.** This project is not a medical device, does not provide medical advice, and must not be used to make clinical decisions. Model scores are based on the included dataset and one held-out split; they do not establish clinical validity or generalization to another hospital or population.

## Features

- Data loading and validation
- Missing-value handling using diagnosis-wise median imputation with a global-median fallback
- Feature engineering: `age_group`, `high_risk_flag`, and `medication_per_day`
- Stratified 80/20 train-test split
- One-hot encoding for categorical variables
- Standard scaling for Logistic Regression and ANN; unscaled inputs for tree-based models
- Five-fold stratified cross-validation and hyperparameter search
- Class-imbalance handling
- Four models: Logistic Regression, Random Forest, XGBoost, and ANN (MLPClassifier)
- Evaluation using ROC-AUC, F1, precision, recall, accuracy, confusion matrices, and ROC curves
- Feature-importance plots for Random Forest and XGBoost
- Tkinter GUI with saved-model loading and patient-input validation

## Repository structure

```text
Hospital-Readmission-Prediction/
├── data/
│   └── 11_hospital_readmission_cleaned.csv
├── models/
│   └── best_model.pkl
├── output/
│   ├── confusion_matrices.png
│   ├── feature_importance_rf.png
│   ├── feature_importance_xgb.png
│   ├── model_comparison.png
│   └── roc_curves.png
├── hospital_readmission_gui.py
├── hospital_readmission_pipeline.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The included CSV contains 900 rows and these columns:

- `age`
- `length_of_stay`
- `diagnosis`
- `num_medications`
- `prev_admissions`
- `discharge_type`
- `bmi`
- `readmitted_30d` (target: `1` = readmitted within 30 days, `0` = not readmitted)
- `id` (identifier; dropped before model training)

Target distribution in the included file: 722 class-0 records and 178 class-1 records. Confirm that you have permission to publish the dataset and that it contains no identifiable or confidential patient information before making this repository public.

## Setup

Python 3.11 or newer is recommended. The included `best_model.pkl` was generated with the exact dependency versions listed in `requirements.txt`; use those versions when loading the saved model. Tkinter is included with most standard Windows Python installations. On some Linux distributions, install the OS package `python3-tk` if Tkinter is unavailable.

```bash
git clone <YOUR_REPOSITORY_URL>
cd Hospital-Readmission-Prediction

python -m venv .venv
```

Activate the virtual environment:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run model training and evaluation

```bash
python hospital_readmission_pipeline.py
```

The script trains and compares the four classifiers, prints evaluation reports, saves the best pipeline to `models/best_model.pkl`, and writes plots to `output/`.

The hyperparameter search has been reduced from the original very large search space to make a full run more practical on a student laptop. The run still uses stratified five-fold cross-validation. Results may vary slightly across library versions and hardware.

## Run the desktop GUI

```bash
python hospital_readmission_gui.py
```

The GUI checks for a saved model in `models/best_model.pkl`, `output/best_model.pkl`, and the project root. If found, it loads the model automatically so prediction can be used without retraining. You can also use **Load Saved Model**. To train models from the GUI, first click **Load & Preprocess Data**, then **Train All Models**.

## Latest run on the included dataset

The pipeline was executed after the search-space optimization. Results from the held-out test set:

| Model | ROC-AUC | F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|
| ANN (MLP) | **0.6248** | **0.3619** | 0.2754 | 0.5278 | 0.6278 |
| Logistic Regression | 0.6238 | 0.3214 | 0.2368 | 0.5000 | 0.5778 |
| Random Forest | 0.6140 | 0.0541 | 1.0000 | 0.0278 | 0.8056 |
| XGBoost | 0.5712 | 0.3235 | 0.3438 | 0.3056 | 0.7444 |

ANN had the highest ROC-AUC and F1 in this run. Overall discrimination remains modest, and Random Forest's very low recall means it misses most readmission cases at the default classification threshold. Accuracy alone is not an appropriate selection metric for this imbalanced target.

## Important implementation notes

- The preprocessing steps are included in each saved scikit-learn pipeline. The included pickle was generated with scikit-learn 1.8.0 and the package versions pinned in `requirements.txt`; use the pinned versions to avoid model-serialization compatibility issues.
- `best_model.pkl` is a Python pickle artifact. Only load model files from sources you trust.
- Feature importance describes how a fitted model used input features; it does not prove that a feature causes readmission.
- The GUI validates basic input ranges, but these checks are software safeguards rather than clinical rules.

## License

Add a license before redistributing this repository. If you do not intend to grant reuse rights, leave the repository without a license.
