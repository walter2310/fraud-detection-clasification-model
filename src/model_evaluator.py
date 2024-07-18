import joblib
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

MODEL_FILENAME = '../models/fraud_detection_model.pkl'
X_TEST_SET_PATH = '../data/processed/x_test_set.csv'
Y_TEST_PATH = '../data/processed/y_test.csv'

X_test_prep = pd.read_csv(X_TEST_SET_PATH)
y_test = pd.read_csv(Y_TEST_PATH)

# Utilizar únicamente la columna con las categorías, eliminando los índices
y_val = y_test.iloc[:, 1]

# Cargar el modelo entrenado
clf = joblib.load(MODEL_FILENAME)

y_pred = clf.predict(X_test_prep)

ConfusionMatrixDisplay.from_estimator(clf, X_test_prep, y_val, values_format='d')
plt.title('Matriz de Confusión')
plt.show()

# Calcular y mostrar el F1 score
print("F1 score:", f1_score(y_val, y_pred, pos_label='anomaly'))
