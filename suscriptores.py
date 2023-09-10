import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data_set_youtubers = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTjiJn06PS3re6UayHZOHqgDxqegCFAw5ouXeTbFP8sUPRyzC4bWbQPPRpO72NCvg/pubhtml"

df = pd.read_excel(data_set_youtubers)

# características (variables independientes) y variable objetivo (variable dependiente)
X = df[['video views', 'uploads', 'country_rank']]  # Características
y = df['subscribers']  # Variable objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba (por ejemplo, 80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# regresión lineal
modelo_regresion = LinearRegression()

# Entrenar el modelo de regresión lineal en el conjunto de entrenamiento
modelo_regresion.fit(X_train, y_train)
