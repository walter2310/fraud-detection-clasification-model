import pandas as pd

def split_target_value(df):
    X_df = df.drop("class", axis=1)
    y_df = df["class"].copy()

    return X_df, y_df

# Función para aplicar min-max scaling
def min_max_scaling(df, columns):
    df = df.copy()
    for col in columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

def handle_missing_values(df):
    return df.dropna()

def dummy_coding(df, categorical_column):
    dummies = pd.get_dummies(df[categorical_column]).astype(int)
    df = df.join(dummies).drop([categorical_column], axis=1)
    return df