import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay

MODEL_FILENAME = '../models/fraud_detection_model.pkl'
X_VAL_SET_PATH = '../data/processed/x_val_set.csv'
Y_VAL_PATH = '../data/processed/y_val.csv'

X_val_prep = pd.read_csv(X_VAL_SET_PATH)
y_val = pd.read_csv(Y_VAL_PATH)

# Utilizar únicamente la columna con las categorías, eliminando los índices
y_val = y_val.iloc[:, 1]

# Cargar el modelo entrenado
clf = joblib.load(MODEL_FILENAME)

y_pred = clf.predict(X_val_prep)

ConfusionMatrixDisplay.from_estimator(clf, X_val_prep, y_val, normalize='true', values_format='.2f')
plt.title('Matriz de Confusión Normalizada')
plt.show()

RocCurveDisplay.from_estimator(clf, X_val_prep, y_val)
plt.title('Curva ROC - Validation set')
plt.show()

PrecisionRecallDisplay.from_estimator(clf, X_val_prep, y_val)
plt.title('Curva PR - Validation set')
plt.show()

print("Recall:", recall_score(y_val, y_pred, pos_label='anomaly'))
print("Precisión:", precision_score(y_val, y_pred, pos_label='anomaly'))

# Calcular y mostrar el F1 score
print("F1 score:", f1_score(y_val, y_pred, pos_label='anomaly'))