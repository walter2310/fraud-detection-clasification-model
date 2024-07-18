import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = '../data/raw/fraud_transaction.csv'

def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def split_train_val_test(df, random_state=42, shuffle=True, stratify_column=None):
    stratify_data = df[stratify_column] if stratify_column else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=random_state, shuffle=shuffle, stratify=stratify_data)

    stratify_data = test_set[stratify_column] if stratify_column else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=random_state, shuffle=shuffle, stratify=stratify_data)

    return (train_set, val_set, test_set)

def check_data(df):
    print(df.info())
    print(df.describe())


if __name__ == "__main__":
    df = load_data()
    check_data(df)

    split_train_val_test(df, stratify_column='protocol_type')