import pandas as pd
from data_loader import split_train_val_test
from transformers.dataframe_preparer import DataFramePreparer
from utils.data_cleaning import split_target_value

DATASET_PATH = '../data/raw/fraud_transaction.csv'

df = pd.read_csv(DATASET_PATH)
train_set, val_set, test_set = split_train_val_test(df)

# Conjunto de datos general
X_df, y_df = split_target_value(df)

X_train, y_train = split_target_value(train_set)
X_val, y_val = split_target_value(val_set)
X_test, y_test = split_target_value(test_set)

data_preparer = DataFramePreparer()
data_preparer.fit(X_df)

X_train_prep = data_preparer.transform(X_train)
X_val_prep = data_preparer.transform(X_val)
X_test_prep = data_preparer.transform(X_test)

X_train_prep.to_csv('../data/processed/x_train_set.csv')
y_train.to_csv('../data/processed/y_train.csv')
X_test_prep.to_csv('../data/processed/x_test_set.csv')
y_test.to_csv('../data/processed/y_test.csv')
X_val_prep.to_csv('../data/processed/x_val_set.csv')
y_val.to_csv('../data/processed/y_val.csv')