"""EDA - FASE1.ipynb

# Proyecto Final Fase 1 - Taller EDA
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

# Instalar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Configuración visual
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 11

print('Librerías cargadas correctamente.')

"""## 1. Carga y consolidación del dataset

El dataset Adult (Census Income) del UCI se distribuye en dos archivos:
- `adult.data` - datos de entrenamiento (~32,561 registros)
- `adult.test` - datos de prueba (~16,281 registros)

Se consolidan en un único DataFrame para el análisis.
"""

# Nombres de columnas según documentación UCI
COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

# Fuente de datos: usa archivos locales si existen (adult.data / adult.test),
# de lo contrario descarga desde el repositorio UCI.
import os
URLDATA = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
URLTEST = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
SRC_DATA = 'adult.data' if os.path.exists('adult.data') else URLDATA
SRC_TEST = 'adult.test' if os.path.exists('adult.test') else URLTEST

# Separa coma + espacios opcionales
dfTrain = pd.read_csv(SRC_DATA, names=COLUMNS, sep=r',\s*', engine='python', na_values='?')

# adult.test tiene una línea de cabecera extra y puntos al final de 'income'
dfTest = pd.read_csv(SRC_TEST, names=COLUMNS, sep=r',\s*', engine='python', skiprows=1, na_values='?')

# Normalizar columna 'income': '<=50K.'
dfTest['income'] = dfTest['income'].str.rstrip('.')

# Agregar columna de origen antes de consolidar
dfTrain['_origen'] = 'train'
dfTest['_origen'] = 'test'

# Consolidación
df = pd.concat([dfTrain, dfTest], ignore_index=True)

print(f'Registros de entrenamiento: {len(dfTrain):,}')
print(f'Registros de prueba: {len(dfTest):,}')
print(f'Dataset consolidado: {len(df):,} filas x {df.shape[1]} columnas')

"""## A. Comprensión y Análisis Gráfico (EDA)

### A.1 Vista general del dataset
"""

df.head()

print('Tipos de datos')
print(df.dtypes)
print('\nEstadísticas descriptivas (numéricas)')
df.describe()

print('Estadísticas descriptivas (categóricas)')
df.describe(include='object')

"""### A.2 Variable objetivo: `income`

La columna **`income`** indica si el ingreso anual de una persona supera los US$50,000.  
- `<=50K` - clase negativa  
- `>50K`  - clase positiva

"""

conteo = df['income'].value_counts()
porcentaje = df['income'].value_counts(normalize=True) * 100

print('Distribución de la variable objetivo:')
print(pd.DataFrame({'Conteo': conteo, 'Porcentaje (%)': porcentaje.round(2)}))

fig, ax = plt.subplots()
bars = ax.bar(conteo.index, conteo.values, color=['#4C72B0','#DD8452'], edgecolor='white', width=0.5)
for bar, pct in zip(bars, porcentaje.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{pct:.1f}%', ha='center', fontweight='bold')
ax.set_title('Distribución de la variable objetivo: income', fontsize=13, fontweight='bold')
ax.set_xlabel('Clase de ingreso')
ax.set_ylabel('Cantidad de registros')
plt.tight_layout()
plt.savefig('figIncome.png', dpi=150)
plt.show()

"""### A.3 Distribución de variables numéricas"""

numCols = df.select_dtypes(include=np.number).columns.tolist()
numCols = [c for c in numCols if c != 'fnlwgt']

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for i, col in enumerate(numCols):
    axes[i].hist(df[col].dropna(), bins=30, color='#4C72B0', edgecolor='white', alpha=0.8)
    axes[i].set_title(col, fontweight='bold')
    axes[i].set_xlabel('Valor')
    axes[i].set_ylabel('Frecuencia')

# Ocultar subplots sobrantes
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribución de variables numéricas', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figNumDist.png', dpi=150)
plt.show()

"""### A.4 Variables numéricas por clase de ingreso (boxplots)"""

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for i, col in enumerate(numCols):
    df.boxplot(column=col, by='income', ax=axes[i], grid=False, boxprops=dict(color='#4C72B0'), medianprops=dict(color='red', linewidth=2))
    axes[i].set_title(col, fontweight='bold')
    axes[i].set_xlabel('Clase de ingreso')

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Variables numéricas por clase de ingreso', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figBoxplots.png', dpi=150)
plt.show()

"""### A.5 Distribución de variables categóricas"""

catCols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'sex']

for col in catCols:
    fig, ax = plt.subplots(figsize=(10, 4))
    order = df[col].value_counts().index
    sns.countplot(data=df, y=col, order=order, hue='income', ax=ax, palette='muted')
    ax.set_title(f'Distribución de {col} por clase de ingreso', fontweight='bold')
    ax.set_xlabel('Cantidad')
    ax.set_ylabel(col)
    plt.tight_layout()
    plt.savefig(f'figCat{col}.png', dpi=120)
    plt.show()

"""### A.6 Matriz de correlación (variables numéricas)"""

corr = df[numCols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5)
ax.set_title('Matriz de correlación – variables numéricas', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figCorrelacion.png', dpi=150)
plt.show()

"""## B. Preprocesamiento de la Información

### B.1 Identificación y tratamiento de valores nulos
"""

nulos = df.isnull().sum()
pctNulos = (nulos / len(df) * 100).round(2)

resumenNulos = pd.DataFrame({'Nulos': nulos, '% del total': pctNulos})
resumenNulos = resumenNulos[resumenNulos['Nulos'] > 0].sort_values('Nulos', ascending=False)
print('Columnas con valores nulos:')
print(resumenNulos)

# Visualización
if not resumenNulos.empty:
    fig, ax = plt.subplots(figsize=(8, 4))
    resumenNulos['% del total'].plot(kind='barh', ax=ax, color='#DD8452')
    ax.set_title('Porcentaje de valores nulos por columna', fontweight='bold')
    ax.set_xlabel('% de registros nulos')
    plt.tight_layout()
    plt.savefig('figNulos.png', dpi=150)
    plt.show()

# Estrategia: imputar con la moda en variables categóricas.
# Justificación:
# Los nulos están en workclass, occupation y native_country (<7%).
# La moda preserva la distribución original sin introducir categorías artificiales.
dfClean = df.copy()

# Quitar columna auxiliar
dfClean.drop(columns=['_origen'], inplace=True)

# Imputación
for col in ['workclass', 'occupation', 'native_country']:
    # Obtener moda
    moda = dfClean[col].mode()[0]
    # Reemplazar nulos
    dfClean[col] = dfClean[col].fillna(moda)
    print(f'{col}: nulos imputados con moda → "{moda}"')

# Verificar nulos restantes
print(f'\nNulos restantes: {dfClean.isnull().sum().sum()}')

"""### B.2 Búsqueda y manejo de valores atípicos (outliers)"""

def detectar_outliers_iqr(series):
    """Retorna máscara booleana de outliers usando el método IQR."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (series < lower) | (series > upper)

print(f"{'Columna':<20} {'Outliers':>10} {'%':>8}")
print('-' * 40)
for col in numCols:
    mask = detectar_outliers_iqr(dfClean[col])
    n = mask.sum()
    pct = n / len(dfClean) * 100
    print(f'{col:<20} {n:>10,} {pct:>7.2f}%')

# Decisión justificada: capital_gain y capital_loss tienen distribución extrema
# Se aplicará winsorización al percentil 99 para conservar registros.
for col in ['capital_gain', 'capital_loss']:
    p99 = dfClean[col].quantile(0.99)
    dfClean[col] = dfClean[col].clip(upper=p99)
    print(f'\nWinsorización aplicada a {col} (p99 = {p99:,.0f})')

"""### B.3 Dummificación de variables categóricas"""

# Codificar variable objetivo: <=50K -> 0, >50K -> 1
dfClean['income'] = dfClean['income'].map({'<=50K': 0, '>50K': 1})

catFeatures = dfClean.select_dtypes(include='object').columns.tolist()
print(f'Variables a dummificar: {catFeatures}')

# drop_first=True evita multicolinealidad (trampa de la variable ficticia)
dfEncoded = pd.get_dummies(dfClean, columns=catFeatures, drop_first=True)

print(f'\nDimensiones antes de dummificar : {dfClean.shape}')
print(f'Dimensiones después de dummificar: {dfEncoded.shape}')

"""### B.4 Normalización de variables numéricas"""

# Se aplica MinMaxScaler (escala 0-1) a columnas numéricas continuas.
# Justificación: preserva la distribución original y es compatible con algoritmos sensibles a escala (SVM, KNN, redes neuronales).

colsNormalizar = ['age', 'fnlwgt', 'education_num', 'capital_gain',
                   'capital_loss', 'hours_per_week']

scaler = MinMaxScaler()
dfEncoded[colsNormalizar] = scaler.fit_transform(dfEncoded[colsNormalizar])

print('Estadísticas tras normalización:')
dfEncoded[colsNormalizar].describe().loc[['min','max','mean']].round(4)

"""## C. Construcción del Dataset de Trabajo: Particionamiento"""

X = dfEncoded.drop(columns=['income'])
y = dfEncoded['income']

# Proporción 80/20 - justificación:
# Con ~48,000 registros, el 20% (≈9,600) es suficiente para evaluar generalización
# sin sacrificar capacidad de aprendizaje. stratify=y garantiza la misma
# proporción de clases en ambos subconjuntos (muestreo estratificado).
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print(f'Tamaño del conjunto de entrenamiento: {XTrain.shape[0]:,} ({XTrain.shape[0]/len(X)*100:.1f}%)')
print(f'Tamaño del conjunto de prueba: {XTest.shape[0]:,} ({XTest.shape[0]/len(X)*100:.1f}%)')
print(f'\nBalance de clases – Entrenamiento:')
print(yTrain.value_counts(normalize=True).rename({0:'<=50K',1:'>50K'}).round(3))
print(f'\nBalance de clases – Prueba:')
print(yTest.value_counts(normalize=True).rename({0:'<=50K',1:'>50K'}).round(3))

# Análisis del impacto de diferentes proporciones de partición
proporciones = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
trainSizes = [int((1 - p) * len(X)) for p in proporciones]
testSizes = [int(p * len(X)) for p in proporciones]

fig, ax = plt.subplots(figsize=(10, 5))
pcts = [p*100 for p in proporciones]
ax.plot(pcts, trainSizes, 'o-', color='#4C72B0', label='Entrenamiento', linewidth=2)
ax.plot(pcts, testSizes,  's-', color='#DD8452', label='Prueba', linewidth=2)
ax.axvline(x=20, color='green', linestyle='--', alpha=0.6, label='Selección: 80/20')
ax.set_xlabel('Porcentaje destinado a prueba (%)')
ax.set_ylabel('Cantidad de registros')
ax.set_title('Impacto del particionamiento en el tamaño de los subconjuntos', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figParticion.png', dpi=150)
plt.show()

print('\nResumen del efecto de variar la proporción:')
print('Una proporción de prueba mayor reduce los datos de entrenamiento,')
print('lo que puede empeorar la capacidad del modelo para generalizar.')
print('Una proporción menor da más datos para entrenar, pero puede hacer')
print('la evaluación menos confiable por el menor tamaño de prueba.')

"""## Resumen final del dataset procesado"""

print('=' * 50)
print('RESUMEN DEL DATASET PROCESADO')
print('=' * 50)
print(f'Registros totales: {len(dfEncoded):,}')
print(f'Variables (features): {X.shape[1]}')
print(f'Variable objetivo: income (binaria: 0=<=50K, 1=>50K)')
print(f'Conjunto de entrenamiento: {XTrain.shape[0]:,} registros')
print(f'Conjunto de prueba: {XTest.shape[0]:,} registros')
print(f'Valores nulos restantes: {dfEncoded.isnull().sum().sum()}')
print('=' * 50)

# Exportar datasets particionados
XTrain.to_csv('XTrain.csv', index=False)
XTest.to_csv('XTest.csv',  index=False)
yTrain.to_csv('yTrain.csv', index=False)
yTest.to_csv('yTest.csv',  index=False)
print('Archivos exportados: XTrain.csv, XTest.csv, yTrain.csv, yTest.csv')