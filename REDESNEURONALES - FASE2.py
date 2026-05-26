"""REDESNEURONALES - FASE2.ipynb

# Proyecto Final Fase 2 - Redes Neuronales Feed-Forward
**Pontificia Universidad Javeriana**  
Introducción a la Inteligencia Artificial    

**PRESENTADO POR:**  
Juan David Rincón Muñoz  
Nicolas Torres Roa  
Juan Daniel Ortiz Quecán  
Nicolas Castañeda Vargas

**PRESENTADO A:**  
Ing. Julio Omar Palacio Niño  
Adult (Census Income) – UCI Repository

## 0. Instalación y carga de librerías
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['font.size'] = 11

print(f'TensorFlow: {tf.__version__}')
print('Librerías cargadas correctamente.')

"""## 1. Carga del dataset procesado (continuación Fase 1)

Se retoma el pipeline completo de la Fase 1 para garantizar reproducibilidad.
"""

# Paso 1: Cargar y consolidar dataset
COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

import os
URLDATA = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
URLTEST = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
SRC_DATA = 'adult.data' if os.path.exists('adult.data') else URLDATA
SRC_TEST = 'adult.test' if os.path.exists('adult.test') else URLTEST

dfTrain = pd.read_csv(SRC_DATA, names=COLUMNS, sep=r',\s*', engine='python', na_values='?')
dfTest = pd.read_csv(SRC_TEST, names=COLUMNS, sep=r',\s*', engine='python', skiprows=1, na_values='?')
dfTest['income'] = dfTest['income'].str.rstrip('.')

df = pd.concat([dfTrain, dfTest], ignore_index=True)

# Paso 2: Imputar nulos con moda (reasignación; evita el FutureWarning de inplace encadenado)
for col in ['workclass', 'occupation', 'native_country']:
    df[col] = df[col].fillna(df[col].mode()[0])

# Paso 3: Winsorización de capital_gain / capital_loss
for col in ['capital_gain', 'capital_loss']:
    df[col] = df[col].clip(upper=df[col].quantile(0.99))

# Paso 4: Codificar variable objetivo
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

# Paso 5: One-hot encoding
catFeatures = df.select_dtypes(include='object').columns.tolist()
dfEncoded   = pd.get_dummies(df, columns=catFeatures, drop_first=True)

# Paso 6: Normalización MinMaxScaler
colsNum = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
scaler = MinMaxScaler()
dfEncoded[colsNum] = scaler.fit_transform(dfEncoded[colsNum])

# Paso 7: Particionamiento 80/20 estratificado
X = dfEncoded.drop(columns=['income'])
y = dfEncoded['income']

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

NFEATURES = XTrain.shape[1]
print(f'Dataset consolidado   : {len(dfEncoded):,} registros')
print(f'Variables de entrada  : {NFEATURES}')
print(f'Entrenamiento         : {XTrain.shape[0]:,}  |  Prueba: {XTest.shape[0]:,}')
print(f'Desbalance de clases  : {yTrain.value_counts(normalize=True).round(3).to_dict()}')

"""## 2. Función auxiliar de evaluación

Centraliza el cálculo de métricas y visualización de la matriz de confusión para los tres modelos.
"""

def evaluar_modelo(nombre, y_true, y_pred, guardar=True):
    """Calcula y muestra métricas + matriz de confusión."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f'\n{"═"*52}')
    print(f'  {nombre}')
    print(f'{"═"*52}')
    print(f'  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)')
    print(f'  Precision : {prec:.4f}')
    print(f'  Recall    : {rec:.4f}')
    print(f'  F1-score  : {f1:.4f}')
    print(f'\n{classification_report(y_true, y_pred, target_names=["<=50K",">50K"], zero_division=0)}')

    # Matriz de confusión
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=['<=50K', '>50K'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Matriz de Confusión – {nombre}', fontweight='bold', fontsize=12)
    plt.tight_layout()
    if guardar:
        fname = nombre.lower().replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')
        plt.savefig(f'cm{fname}.png', dpi=150)
    plt.show()

    return {'Modelo': nombre, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1}

resultados = []  # Acumulará métricas de los 3 modelos

"""## 3. Modelo A – Perceptrón

### Descripción arquitectónica
El Perceptrón es el clasificador lineal más simple: aplica una combinación lineal de las entradas
más un término de sesgo (bias) y produce una salida binaria mediante una función escalón.
No tiene capas ocultas.

```
Entrada (107 vars) ──── [w·x + b] ──── Salida (0 / 1)
```
"""

# Entrenamiento
# fit_intercept=True garantiza el término de bias obligatorio
# max_iter=1000 para asegurar convergencia; random_state para reproducibilidad
modeloA = Perceptron(fit_intercept=True, max_iter=1000, random_state=42, tol=1e-4)
modeloA.fit(XTrain, yTrain)

y_pred_A = modeloA.predict(XTest)

print(f'Parámetros del Perceptrón:')
print(f'  Pesos (w): {modeloA.coef_.shape}  |  Bias: {modeloA.intercept_}')
print(f'  Iteraciones de convergencia: {modeloA.n_iter_}')

metricas_A = evaluar_modelo('Modelo A: Perceptrón', yTest, y_pred_A)
resultados.append(metricas_A)

"""## 4. Modelo B – Red Neuronal con una Capa Oculta

### Descripción arquitectónica
Una capa oculta con **n_features = 107 neuronas**, función sigmoide en la capa oculta y de salida, bias en todas las neuronas.

```
Entrada (107) ──── [Hidden: 107 neuronas, sigmoid] ──── [Output: sigmoid] ──── Salida
```
"""

# Implementación con Keras (permite sigmoide explícita y bias visible)
tf.random.set_seed(42)
np.random.seed(42)

modeloB = keras.Sequential([
    # Capa oculta: NFEATURES neuronas, activación sigmoide, bias incluido por defecto
    layers.Dense(NFEATURES, activation='sigmoid', use_bias=True,
                input_shape=(NFEATURES,), name='capa_oculta'),
    # Capa de salida: 1 neurona, sigmoide para clasificación binaria
    layers.Dense(1, activation='sigmoid', use_bias=True, name='capa_salida')
], name='ModeloB_1CapaOculta')

modeloB.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

modeloB.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

histB = modeloB.fit(
    XTrain, yTrain,
    validation_split=0.15,
    epochs=100,
    batch_size=256,
    callbacks=[early_stop],
    verbose=1
)

# Curvas de aprendizaje
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(histB.history['loss'],     label='Train loss', color='#4C72B0')
axes[0].plot(histB.history['val_loss'], label='Val loss',   color='#DD8452', linestyle='--')
axes[0].set_title('Modelo B – Pérdida (Loss)', fontweight='bold')
axes[0].set_xlabel('Época'); axes[0].set_ylabel('Binary Crossentropy'); axes[0].legend()

axes[1].plot(histB.history['accuracy'],     label='Train Acc', color='#4C72B0')
axes[1].plot(histB.history['val_accuracy'], label='Val Acc',   color='#DD8452', linestyle='--')
axes[1].set_title('Modelo B – Exactitud (Accuracy)', fontweight='bold')
axes[1].set_xlabel('Época'); axes[1].set_ylabel('Accuracy'); axes[1].legend()

plt.tight_layout()
plt.savefig('curvasModeloB.png', dpi=150)
plt.show()

# Evaluación
yprobB = modeloB.predict(XTest).flatten()
ypredB = (yprobB >= 0.5).astype(int)

metricas_B = evaluar_modelo('Modelo B: Red 1 Capa Oculta (107 neuronas)', yTest, ypredB)
resultados.append(metricas_B)

"""## 5. Modelo C – Red Neuronal con dos Capas Ocultas

### Descripción arquitectónica
Dos capas ocultas con **exactamente 2 neuronas** cada una, activación sigmoide en capas ocultas y de salida, bias en todas.

```
Entrada (107) ──── [Hidden1: 2 neuronas, sigmoid] ──── [Hidden2: 2 neuronas, sigmoid] ──── [Output: sigmoid] ──── Salida
```

> **Nota técnica:** Este modelo presenta un cuello de botella severo (107 → 2 → 2 → 1). La capacidad representacional es muy limitada, lo que generará un rendimiento inferior.
"""

tf.random.set_seed(42)
np.random.seed(42)

modeloC = keras.Sequential([
    # Primera capa oculta: 2 neuronas, sigmoide, bias
    layers.Dense(2, activation='sigmoid', use_bias=True,
                input_shape=(NFEATURES,), name='capa_oculta_1'),
    # Segunda capa oculta: 2 neuronas, sigmoide, bias
    layers.Dense(2, activation='sigmoid', use_bias=True, name='capa_oculta_2'),
    # Capa de salida: sigmoide
    layers.Dense(1, activation='sigmoid', use_bias=True, name='capa_salida')
], name='ModeloC_2CapasOcultas')

modeloC.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

modeloC.summary()

histC = modeloC.fit(
    XTrain, yTrain,
    validation_split=0.15,
    epochs=150,
    batch_size=256,
    callbacks=[early_stop],
    verbose=1
)

# Curvas de aprendizaje
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(histC.history['loss'],     label='Train loss', color='#4C72B0')
axes[0].plot(histC.history['val_loss'], label='Val loss',   color='#DD8452', linestyle='--')
axes[0].set_title('Modelo C – Pérdida (Loss)', fontweight='bold')
axes[0].set_xlabel('Época'); axes[0].set_ylabel('Binary Crossentropy'); axes[0].legend()

axes[1].plot(histC.history['accuracy'],     label='Train Acc', color='#4C72B0')
axes[1].plot(histC.history['val_accuracy'], label='Val Acc',   color='#DD8452', linestyle='--')
axes[1].set_title('Modelo C – Exactitud (Accuracy)', fontweight='bold')
axes[1].set_xlabel('Época'); axes[1].set_ylabel('Accuracy'); axes[1].legend()

plt.tight_layout()
plt.savefig('curvasmodeloC.png', dpi=150)
plt.show()

# Evaluación
yprobC = modeloC.predict(XTest).flatten()
ypredC = (yprobC >= 0.5).astype(int)

metricas_C = evaluar_modelo('Modelo C: Red 2 Capas Ocultas (2+2 neuronas)', yTest, ypredC)
resultados.append(metricas_C)

"""## 6. Análisis Comparativo de los Tres Modelos"""

# Tabla resumen
dfRes = pd.DataFrame(resultados).set_index('Modelo')
print('\n', '═'*70)
print('  RESUMEN COMPARATIVO DE LOS TRES MODELOS')
print('═'*70)
print(dfRes.round(4).to_string())
print('═'*70)

# Gráfico comparativo de métricas
metricasPlot = ['Accuracy', 'Precision', 'Recall', 'F1-score']
x = np.arange(len(metricasPlot))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
colores = ['#4C72B0', '#DD8452', '#55A868']

for i, (idx, row) in enumerate(dfRes.iterrows()):
    vals = [row[m] for m in metricasPlot]
    bars = ax.bar(x + i*width, vals, width, label=idx.split(':')[0], color=colores[i], alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Métrica')
ax.set_ylabel('Valor')
ax.set_title('Comparación de Métricas – Modelos A, B y C', fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metricasPlot)
ax.set_ylim(0, 1.1)
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('comparativo_modelos.png', dpi=150)
plt.show()

# Análisis del desbalance de clases
print('\n Impacto del desbalance de clases')
print('Distribución en conjunto de prueba:')
print(yTest.value_counts().rename({0:'<=50K (clase negativa)',1:'>50K (clase positiva)'}))
print('\nEl dataset presenta ~75% clase negativa / ~25% clase positiva.')
print('Un clasificador trivial (predice siempre <=50K) alcanzaría ~75% de accuracy.')
print('Por esto, el F1-score para la clase >50K es la métrica más informativa.')

mejor = dfRes['F1-score'].idxmax()
print(f'\n✅ Mejor modelo por F1-score: {mejor}')

# Guardar resultados para Fase 3
dfRes.round(4).to_csv('resultadosFase2.csv')
print('Métricas exportadas a resultadosFase2.csv')
print('\nResumen listo para comparación con algoritmo clásico en Fase 3.')