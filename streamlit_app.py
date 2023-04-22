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



# import streamlit as st
# import pandas as pd
# import numpy as np

# st.set_page_config(page_title="Regresión Logística Interactiva", layout="wide")

# st.title("Regresión Logística Interactiva")


# def train_test_split(X, y, test_size=0.2, random_state=None):
#     if random_state is not None:
#         np.random.seed(random_state)
#     indices = np.random.permutation(len(X))
#     test_set_size = int(len(X) * test_size)
#     test_indices = indices[:test_set_size]
#     train_indices = indices[test_set_size:]
#     return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]


# class LogisticRegression:
#     def __init__(self, learning_rate=0.01, max_iter=1000):
#         self.learning_rate = learning_rate
#         self.max_iter = max_iter

#     def sigmoid(self, z):
#         return 1 / (1 + np.exp(-z))

#     def fit(self, X, y):
#         X = np.hstack((np.ones((X.shape[0], 1)), X))
#         self.theta = np.zeros(X.shape[1])

#         for _ in range(self.max_iter):
#             z = np.dot(X, self.theta)
#             h = self.sigmoid(z)
#             gradient = np.dot(X.T, (h - y)) / y.size
#             self.theta -= self.learning_rate * gradient

#     def predict_prob(self, X):
#         X = np.hstack((np.ones((X.shape[0], 1)), X))
#         return self.sigmoid(np.dot(X, self.theta))

#     def predict(self, X, threshold=0.5):
#         return (self.predict_prob(X) >= threshold).astype(int)


# def accuracy(y_true, y_pred):
#     return np.mean(y_true == y_pred)


# uploaded_file = st.sidebar.file_uploader("Carga tu archivo CSV:", type=['csv'])

# if uploaded_file:
#     data = pd.read_csv(uploaded_file)
#     st.sidebar.write("Vista previa de los datos:")
#     st.sidebar.write(data.head())

#     input_features = st.sidebar.multiselect("Selecciona las variables de entrada:", data.columns)
#     target_feature = st.sidebar.selectbox("Selecciona la variable objetivo:", data.columns)

#     if input_features and target_feature:

#         X = data[input_features]
#         y = data[target_feature]
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         logistic_regression = LogisticRegression()
#         logistic_regression.fit(X_train, y_train)

#         y_pred_train = logistic_regression.predict(X_train)
#         y_pred_test = logistic_regression.predict(X_test)

#         st.write("### Resultados del entrenamiento")
#         st.write("Precisión en el conjunto de entrenamiento:", accuracy(y_train, y_pred_train))
#         st.write("Precisión en el conjunto de prueba:", accuracy(y_test, y_pred_test))
# else:
#     st.sidebar.warning("Por favor, carga un archivo CSV.")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Leer los datos
titanic_df = pd.read_csv('titanic.csv')

# Convertir la variable 'Sex' en variables ficticias (dummy variables)
dummies_sexo = pd.get_dummies(titanic_df['Sex'], prefix='Sexo')
titanic_df = pd.concat([titanic_df, dummies_sexo], axis=1)

# Seleccionar características y eliminar valores faltantes
caracteristicas_seleccionadas = ['Edad', 'Sexo_female', 'Sexo_male', 'Clase', 'Tarifa']
titanic_df = titanic_df[caracteristicas_seleccionadas + ['Sobrevivio']].dropna()

# Definir la función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Definir la función de costo
def funcion_de_costo(X, y, pesos):
    m = len(y)
    h = sigmoid(X.dot(pesos))
    costo = -(1/m) * np.sum(y * np.log(h) + (1-y) * np.log(1-h))
    gradiente = (1/m) * X.T.dot(h-y)
    return costo, gradiente

# Definir la función de entrenamiento
def entrenar(X, y, tasa_de_aprendizaje, num_iteraciones):
    m, n = X.shape
    pesos = np.zeros(n)
    for i in range(num_iteraciones):
        costo, gradiente = funcion_de_costo(X, y, pesos)
        pesos = pesos - tasa_de_aprendizaje * gradiente
        if i % 1000 == 0:
            print(f'Costo después de la iteración {i}: {costo}')
    return pesos

# Preparar los datos
X = titanic_df[caracteristicas_seleccionadas]
y = titanic_df['Sobrevivio']
X = np.hstack((np.ones((len(X), 1)), X))
pesos = entrenar(X, y, 0.01, 10000)

# Modificar la función de predicción para devolver la probabilidad
def predecir_probabilidad_supervivencia(edad, sexo, clase, tarifa):
    es_hombre = 0
    es_mujer = 0
    if sexo == 'male':
        es_hombre = 1
    elif sexo == 'female':
        es_mujer = 1
    data = np.array([[1, edad, es_mujer, es_hombre, clase, tarifa]])
    probabilidad = sigmoid(data.dot(pesos))
    return probabilidad[0]

# Streamlit app
st.title("Predicción de supervivencia en el Titanic")

# Subir archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")

if uploaded_file:
    titanic_df = pd.read_csv(uploaded_file)

    # Convertir la variable 'Sex' en variables ficticias (dummy variables)
    dummies_sexo = pd.get_dummies(titanic_df['Sex'], prefix='Sexo')
    titanic_df = pd.concat([titanic_df, dummies_sexo], axis=1)

    # Seleccionar características y eliminar valores faltantes
    caracteristicas_seleccionadas = ['Edad', 'Sexo_female', 'Sexo_male', 'Clase', 'Tarifa']
    titanic_df = titanic_df[caracteristicas_seleccionadas + ['Sobrevivio']].dropna()

    # Preparar los datos
    X = titanic_df[caracteristicas_seleccionadas]
    y = titanic_df['Sobrevivio']
    X = np.hstack((np.ones((len(X), 1)), X))
    pesos = entrenar(X, y, 0.01, 10000)

    # Controles interactivos y gráfica
    edad = st.slider("Edad:", min_value=0, max_value=100, value=30, step=1)
    sexo = st.radio("Sexo:", options=['male', 'female'], index=0)
    clase = st.slider("Clase:", min_value=1, max_value=3, value=3, step=1)
    tarifa = st.slider("Tarifa:", min_value=0, max_value=600, value=100, step=1)

    # Crear gráfico
    fig = go.Figure()

    # Añadir traza de gráfico de barras para la probabilidad de supervivencia
    probabilidad = predecir_probabilidad_supervivencia(edad, sexo, clase, tarifa)
    fig.add_trace(go.Bar(x=['Probabilidad de Supervivencia'], y=[probabilidad], text=[f"{probabilidad:.2%}"], textposition='inside', marker=dict(color='rgba(58, 71, 80, 0.6)', line=dict(color='rgba(58, 71, 80, 1.0)', width=3))))

    # Configurar el diseño del gráfico
    fig.update_layout(title_text='Probabilidad de Supervivencia', yaxis=dict(range=[0, 1], tickformat=".0%"))

    # Mostrar el gráfico en la app Streamlit
    st.plotly_chart(fig)

else:
    st.write("Por favor, sube un archivo CSV.")
