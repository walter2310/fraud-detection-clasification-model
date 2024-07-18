import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

# Transformador para codificar únicamente las columnas categóricas y devolver un DataFrame
class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.one_hot_encoder = OneHotEncoder()
        self.column_names = None

    def fit(self, X):
        categorical_columns = X.select_dtypes(include=['object'])
        self.column_names = pd.get_dummies(categorical_columns).columns
        self.one_hot_encoder.fit(categorical_columns)
        return self

    def transform(self, X):
        X_copy = X.copy()
        categorical_columns = X_copy.select_dtypes(include=['object'])
        categorical_one_hot = self.one_hot_encoder.transform(categorical_columns)

        categorical_one_hot_df = pd.DataFrame(categorical_one_hot.toarray(), columns=self.column_names, index=X_copy.index)
        X_copy.drop(list(categorical_columns), axis=1, inplace=True)
        return X_copy.join(categorical_one_hot_df)
