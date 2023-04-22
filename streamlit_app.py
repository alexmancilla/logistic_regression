# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# st.set_page_config(page_title="Regresión Logística Interactiva", layout="wide")

# st.title("Regresión Logística Interactiva")

# # Cargar datos
# uploaded_file = st.sidebar.file_uploader("Carga tu archivo CSV:", type=['csv'])

# if uploaded_file:
#     data = pd.read_csv(uploaded_file)
#     st.sidebar.write("Vista previa de los datos:")
#     st.sidebar.write(data.head())

#     # Seleccionar variables
#     input_features = st.sidebar.multiselect("Selecciona las variables de entrada:", data.columns)
#     target_feature = st.sidebar.selectbox("Selecciona la variable objetivo:", data.columns)

#     if input_features and target_feature:

#         # Dividir los datos en conjuntos de entrenamiento y prueba
#         X = data[input_features]
#         y = data[target_feature]
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Entrenar el modelo
#         logistic_regression = LogisticRegression()
#         logistic_regression.fit(X_train, y_train)

#         # Realizar predicciones
#         y_pred_train = logistic_regression.predict(X_train)
#         y_pred_test = logistic_regression.predict(X_test)

#         # Mostrar resultados
#         st.write("### Resultados del entrenamiento")
#         st.write("Precisión en el conjunto de entrenamiento:", accuracy_score(y_train, y_pred_train))
#         st.write("Precisión en el conjunto de prueba:", accuracy_score(y_test, y_pred_test))
#         st.write("Matriz de confusión:")
#         st.write(confusion_matrix(y_test, y_pred_test))
#         st.write("Informe de clasificación:")
#         st.write(classification_report(y_test, y_pred_test))
# else:
#     st.sidebar.warning("Por favor, carga un archivo CSV.")



import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Regresión Logística Interactiva", layout="wide")

st.title("Regresión Logística Interactiva")


def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]


class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.theta = np.zeros(X.shape[1])

        for _ in range(self.max_iter):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.learning_rate * gradient

    def predict_prob(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) >= threshold).astype(int)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


uploaded_file = st.sidebar.file_uploader("Carga tu archivo CSV:", type=['csv'])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.sidebar.write("Vista previa de los datos:")
    st.sidebar.write(data.head())

    input_features = st.sidebar.multiselect("Selecciona las variables de entrada:", data.columns)
    target_feature = st.sidebar.selectbox("Selecciona la variable objetivo:", data.columns)

    if input_features and target_feature:

        X = data[input_features]
        y = data[target_feature]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        logistic_regression = LogisticRegression()
        logistic_regression.fit(X_train, y_train)

        y_pred_train = logistic_regression.predict(X_train)
        y_pred_test = logistic_regression.predict(X_test)

        st.write("### Resultados del entrenamiento")
        st.write("Precisión en el conjunto de entrenamiento:", accuracy(y_train, y_pred_train))
        st.write("Precisión en el conjunto de prueba:", accuracy(y_test, y_pred_test))
else:
    st.sidebar.warning("Por favor, carga un archivo CSV.")

