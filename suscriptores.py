from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import tensorflow as tf
from tensorflow import _keras
from tensorflow.keras import layers


df = pd.read_csv(
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vQYsvkc4bQAxWhXlJLlXTL_N2f61FkaWFXD0MdvxdmMJbCL1IDh7Yc7WhbH8bwFcz2s0Np7pnJDqm-_/pub?output=csv")

# características (variables independientes) y variable objetivo (variable dependiente)
X = df[['video views', 'uploads', 'country_rank']]  # Características
y = df['subscribers']  # Variable objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba (por ejemplo, 80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Crear un objeto de regresión Ridge
modelo_regresion_ridge = Ridge(alpha=1.0)

# Entrenar el modelo de regresión Ridge en el conjunto de entrenamiento
modelo_regresion_ridge.fit(X_train, y_train)

# Red neuronales

# Normalizar las características (opcional, pero a menudo es útil en redes neuronales)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear el modelo de redes neuronales
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Capa de salida para regresión
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=100,
                    batch_size=32, validation_split=0.2, verbose=2)

# Evaluar el modelo en el conjunto de prueba
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f'Error cuadrático medio (MSE): {loss}')
print(f'Error absoluto medio (MAE): {mae}')

# Realizar predicciones
y_pred = model.predict(X_test)
