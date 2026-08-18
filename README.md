# Disease Prediction Model

Predicting the presence of heart disease in a patient using classic clinical measurements, comparing three classification models.

![Patient Risk Intake Dashboard](https://github.com/user-attachments/assets/5d8e89d9-bf0b-44f0-af3a-315c3c81cd2a)

## Objective

Predict whether a patient has heart disease (binary: Healthy / Disease) from clinical measurements, and identify which measurements matter most.

## Dataset

- **Source**: [Cleveland Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease), UCI Machine Learning Repository
- **Size**: 303 patients, 14 columns (13 features + target)
- **Target**: originally a 0–4 severity scale, binarized to 0 (Healthy) / 1 (Disease)
- **Missing values**: 6 total (4 in `ca`, 2 in `thal`), handled via imputation rather than dropped

| Feature | Description |
|---|---|
| age, sex | Demographics |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Thalassemia (normal / fixed defect / reversible defect) |

## Approach

1. **EDA** — class balance, distribution shape per feature, correlation heatmap, multicollinearity check
2. **Preprocessing** (all inside a scikit-learn `ColumnTransformer`, to avoid leakage during cross-validation):
   - Binary categorical features (`sex`, `fbs`, `exang`) passed through as-is
   - Multi-category nominal features (`cp`, `restecg`, `slope`, `ca`, `thal`) one-hot encoded
   - Continuous features scaled with `StandardScaler` or `RobustScaler`, chosen automatically per feature based on skewness
   - Missing values median/mode-imputed within the same pipeline
3. **Models**: Logistic Regression, Decision Tree, Random Forest
4. **Evaluation**: held-out 80/20 stratified split (Accuracy, Precision, Recall, F1, ROC-AUC) plus stratified 5-fold cross-validation on ROC-AUC, since the held-out test set alone is only ~60 patients
5. **Interpretability**: both model-native feature importance and permutation importance, to cross-check which features the models are actually relying on

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC (test) | ROC-AUC (5-fold CV mean) |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 86.9% | 83.3% | 89.3% | 86.2% | **96.2%** | **91.8%** |
| Decision Tree | 70.5% | 72.7% | 57.1% | 64.0% | 80.2% | 75.6% |
| Random Forest | 88.5% | 83.9% | 92.9% | 88.1% | 93.7% | 90.0% |

**Logistic Regression** is the strongest model overall. Random Forest edges it out on Accuracy/Precision/Recall/F1 at the default threshold, but Logistic Regression has a higher and more consistent ROC-AUC on both the single test split and across cross-validation folds — meaning it ranks patient risk more reliably regardless of where the decision threshold is set, on top of being more interpretable for a clinical use case.

## Key Predictors

`cp` (chest pain type), `thal` (thalassemia), and `thalach` (max heart rate) consistently rank high across all three models, in both feature importance and permutation importance. `ca` (vessels colored by fluoroscopy) is especially dominant for Logistic Regression specifically.

## Tech Stack

Python, pandas, numpy, scikit-learn, matplotlib, seaborn, scipy

## Running It

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
jupyter notebook disease-prediction.ipynb
```

Place `processed.cleveland.data` (from the UCI link above) in the same directory as the notebook.

## Future Work

- Testing on larger clinical cohorts
- Tuning the decision threshold to further reduce false-negative diagnoses

---

