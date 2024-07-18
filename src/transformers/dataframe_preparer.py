import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from transformers.one_hot_encoder import CustomOneHotEncoder

# Construcción de un pipeline para los atributos numéricos
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('robust_scaler', RobustScaler()),
])

# Transformador que prepara todo el conjunto de datos llamando pipelines y transformadores personalizados
class DataFramePreparer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.full_pipeline = None
        self.column_names = None

    def fit(self, X):
        numeric_attributes = list(X.select_dtypes(exclude=['object']))
        categorical_attributes = list(X.select_dtypes(include=['object']))

        self.full_pipeline = ColumnTransformer([
            ("numeric", numeric_pipeline, numeric_attributes),
            ("categorical", CustomOneHotEncoder(), categorical_attributes),
        ])

        self.full_pipeline.fit(X)
        self.column_names = pd.get_dummies(X).columns
        return self

    def transform(self, X):
        X_copy = X.copy()
        prepared_data = self.full_pipeline.transform(X_copy)
        return pd.DataFrame(prepared_data, columns=self.column_names, index=X_copy.index)
