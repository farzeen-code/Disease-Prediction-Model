import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from scipy.stats import skew
import joblib

Columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
]

df = pd.read_csv('processed.cleveland.data', names=Columns, na_values='?')

df['num']= df['num'].apply(lambda x:1 if x > 0 else 0)

X = df.drop(columns=['num'])
y = df['num']

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def buildPipeline():
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    passthorugh_cols = [c for c in cat_cols if X[c].nunique() == 2]
    nominal_cols = [c for c in cat_cols if X[c].nunique() > 2]

    num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    standard_cols, robust_cols = [], []

    for col in num_features:
        col_skew = abs(skew(X[col].dropna()))
        (robust_cols if col_skew > 0.75 else standard_cols).append(col)

    # Pipelines
    std_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    rob_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('std_num', std_pipeline, standard_cols),
        ('rob_num', rob_pipeline, robust_cols),
        ('cat_num', cat_pipeline, cat_cols),
        ('pass', 'passthrough', passthorugh_cols)
    ])

    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

# Disposable model for sanity check
check_model = buildPipeline()
check_model.fit(X_train, y_train)
print(f'Held-out test accuracy (sanity check): {check_model.score(X_test, y_test):.3f}')

# Real model
model = buildPipeline()
model.fit(X, y)

joblib.dump({'model' : model, 'columns' : Columns[:-1]}, 'model.pkl')
print('Model saved to model.pkl')