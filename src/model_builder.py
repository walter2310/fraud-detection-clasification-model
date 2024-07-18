import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

MODEL_FILENAME = '../models/fraud_detection_model.pkl'
X_TRAIN_SET_PATH = '../data/processed/x_train_set.csv'
Y_TRAIN_PATH = '../data/processed/y_train.csv'

X_train_prep = pd.read_csv(X_TRAIN_SET_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH)

#Se utiliza unicamente la columna con las categorias, eliminando los indices
y_train = y_train.iloc[:, 1]

clf = LogisticRegression(solver="newton-cg", max_iter=1000)
clf.fit(X_train_prep, y_train)

joblib.dump(clf, MODEL_FILENAME)