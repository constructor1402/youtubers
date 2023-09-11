import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim


df = pd.read_csv(
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vTrfTf_O_ci2ym30xdArizt-MoHwNMUiHwdFYqwhH2O2NWOSxO5XBp4KAGrNSxqqJap77uZ_KeFQwAx/pub?output=csv")

# características (variables independientes) y variable objetivo (variable dependiente)
X = df[['video views', 'uploads', 'country_rank']]  # Características
y = df['subscribers']  # Variable objetivo


# Dividir los datos en conjuntos de entrenamiento y prueba (por ejemplo, 80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# y_train = y_train.to_numpy().reshape(-1, 1)
# y_test = y_test.to_numpy().reshape(-1, 1)

# Crear un objeto de regresión Ridge
modelo_regresion_ridge = Ridge(alpha=1.0)

# Entrenar el modelo de regresión Ridge en el conjunto de entrenamiento
modelo_regresion_ridge.fit(X_train, y_train)

# Red neuronales

# Normalizar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir datos a tensores de PyTorch
X_train = torch.from_numpy(X_train).to(dtype=torch.float32)
y_train = torch.from_numpy(y_train.values).view(-1, 1).to(dtype=torch.float32)
X_test = torch.from_numpy(X_test).to(dtype=torch.float32)
y_test = torch.from_numpy(y_test.values).view(-1, 1).to(dtype=torch.float32)


# Definir la arquitectura de la red neuronal


class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 850)  # Capa oculta 1
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(850, 450)  # Capa oculta 2
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(450, 1)  # Capa de salida

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x


# Crear el modelo
input_size = X_train.shape[1]  # Tamaño de entrada según los datos X_train
model = NeuralNet(input_size)

# Definir la función de pérdida y el optimizador
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenar el modelo
num_epochs = 100
batch_size = 32
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i+batch_size]
        targets = y_train[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Evaluar el modelo en el conjunto de prueba
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    mse = criterion(y_pred, y_test)
    mae = torch.abs(y_pred - y_test).mean()

print("Modelo de Red neuronales: ")
print(f'Error cuadrático medio (MSE): {mse.item()}')
print(f'Error absoluto medio (MAE): {mae.item()}')


# modelo de regresion de logistica
tamaño_entrada = X_train.shape[1]
model = NeuralNet(tamaño_entrada)

perdida = nn.MSELoss()
resetear = optim.Adam(model.parameters(), lr=0.001)


numero = 100
tamaño = 32
for j in range(numero):
    for i in range(0, len(X_train), tamaño):
        inputs = X_train[i:i+tamaño]
        targets = y_train[i:i+tamaño]

        resetear.zero_grad()
        outputs = model(inputs)
        loss = perdida(outputs, targets)
        loss.backward()
        resetear.step()

# Evaluar el modelo en el conjunto de prueba
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    mse = perdida(y_pred, y_test)
    mae = torch.abs(y_pred - y_test).mean()


model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
valor = accuracy_score(y_test, y_pred)
informe = classification_report(y_test, y_pred)


# Regresion logistica
print(f'Precisión (valor): {valor}')
print('Informe de clasificación: ')
print(informe)
