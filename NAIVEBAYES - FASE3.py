"""NAIVEBAYES - FASE3.ipynb

# Proyecto Final Fase 3 - Naive Bayes vs Redes Neuronales
**Pontificia Universidad Javeriana**  
Introducción a la Inteligencia Artificial    

**PRESENTADO POR:**  
Juan David Rincón Muñoz  
Nicolas Torres Roa  
Juan Daniel Ortiz Quecán  
Nicolas Castañeda Vargas

**PRESENTADO A:**  
Ing. Julio Omar Palacio Niño  
Técnica de Clasificación: Naive Bayes  
Adult (Census Income) – UCI Repository

## 0. Librerías
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
    ConfusionMatrixDisplay, roc_curve, auc
)

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (9, 5)
print('Librerías cargadas correctamente.')

"""## 1. Pipeline de Fase 1 – Dataset procesado y particionado"""

COLUMNS = [
    'age','workclass','fnlwgt','education','education_num',
    'marital_status','occupation','relationship','race','sex',
    'capital_gain','capital_loss','hours_per_week','native_country','income'
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

for col in ['workclass','occupation','native_country']:
    df[col] = df[col].fillna(df[col].mode()[0])
for col in ['capital_gain','capital_loss']:
    df[col] = df[col].clip(upper=df[col].quantile(0.99))

df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
catFeatures = df.select_dtypes(include='object').columns.tolist()
dfEnc = pd.get_dummies(df, columns=catFeatures, drop_first=True)

colsNum = ['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']
scaler = MinMaxScaler()
dfEnc[colsNum] = scaler.fit_transform(dfEnc[colsNum])

X = dfEnc.drop(columns=['income'])
y = dfEnc['income']

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

NFEATURES = XTrain.shape[1]
print(f'Dataset: {len(dfEnc):,} registros  |  Features: {NFEATURES}')
print(f'Train: {XTrain.shape[0]:,}  |  Test: {XTest.shape[0]:,}')
print(f'Desbalance de clases – train: {yTrain.value_counts(normalize=True).round(3).to_dict()}')

"""## 2. Fundamento matemático – Naive Bayes

### 2.1 Teorema de Bayes
Para una clase $C_k$ y un vector de características $\mathbf{x} = (x_1, x_2, \ldots, x_n)$:

$$P(C_k \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid C_k) \cdot P(C_k)}{P(\mathbf{x})}$$

### 2.2 Supuesto de independencia condicional ("ingenuo")

$$P(\mathbf{x} \mid C_k) = \prod_{i=1}^{n} P(x_i \mid C_k)$$

La regla de clasificación es:

$$\hat{y} = \arg\max_{k} \; P(C_k) \prod_{i=1}^{n} P(x_i \mid C_k)$$

### 2.3 Variantes según distribución de las features
| Variante | Supuesto de $P(x_i \mid C_k)$ | Aplicación |
|---|---|---|
| **GaussianNB** | Distribución normal: $\mathcal{N}(\mu_{ki}, \sigma^2_{ki})$ | Features continuas normalizadas |
| **BernoulliNB** | Distribución de Bernoulli: features binarias | Después de one-hot encoding |
| **ComplementNB** | Complemento de BernoulliNB | Datasets desbalanceados |


**Estrategia elegida:** se evaluarán las tres variantes y se seleccionará la de mayor F1-score.

## 3. Función auxiliar de evaluación
"""

def evaluar(nombre, y_true, y_pred, y_prob=None, guardar=True):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f'\n{"═"*55}')
    print(f'  {nombre}')
    print(f'{"═"*55}')
    print(f'  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)')
    print(f'  Precision : {prec:.4f}')
    print(f'  Recall    : {rec:.4f}')
    print(f'  F1-score  : {f1:.4f}')
    print(f'\n{classification_report(y_true, y_pred, target_names=["<=50K",">50K"], zero_division=0)}')

    # Matriz de confusión
    fig, axes = plt.subplots(1, 2 if y_prob is not None else 1,
                            figsize=(11 if y_prob is not None else 5, 4))
    ax_cm = axes[0] if y_prob is not None else axes
    ConfusionMatrixDisplay(cm, display_labels=['<=50K','>50K']).plot(
        ax=ax_cm, colorbar=False, cmap='Blues')
    ax_cm.set_title(f'Matriz de Confusión – {nombre}', fontweight='bold')

    # Curva ROC (si se proporcionan probabilidades)
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, color='#4C72B0', lw=2, label=f'AUC = {roc_auc:.3f}')
        axes[1].plot([0,1],[0,1],'--', color='gray', lw=1)
        axes[1].set_xlabel('Tasa de Falsos Positivos')
        axes[1].set_ylabel('Tasa de Verdaderos Positivos')
        axes[1].set_title(f'Curva ROC – {nombre}', fontweight='bold')
        axes[1].legend(loc='lower right')

    plt.tight_layout()
    if guardar:
        fname = nombre.lower().replace(' ','_').replace(':','').replace('(','').replace(')','').replace('/','_')
        plt.savefig(f'cm{fname}.png', dpi=150)
    plt.show()

    return {'Modelo': nombre, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1}

resultados = []

"""## 4. Selección de variante: GaussianNB vs BernoulliNB vs ComplementNB"""

# Comparación rápida con validación cruzada estratificada (5-fold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

variantes = {
    'GaussianNB'    : GaussianNB(),
    'BernoulliNB'   : BernoulliNB(alpha=1.0),
    'ComplementNB'  : ComplementNB(alpha=1.0),
}

print(f"{'Variante':<18} {'Accuracy CV':>14} {'F1-score CV':>14}")
print('-' * 48)
cv_results = {}
for nombre, modelo in variantes.items():
    acc_cv = cross_val_score(modelo, XTrain, yTrain, cv=cv, scoring='accuracy').mean()
    f1_cv  = cross_val_score(modelo, XTrain, yTrain, cv=cv, scoring='f1').mean()
    cv_results[nombre] = {'accuracy': acc_cv, 'f1': f1_cv}
    print(f'{nombre:<18} {acc_cv:>14.4f} {f1_cv:>14.4f}')

mejor_variante = max(cv_results, key=lambda k: cv_results[k]['f1'])
print(f'\n✅ Mejor variante por F1-score CV: {mejor_variante}')

"""## 5. Ajuste de hiperparámetros

El principal hiperparámetro de Naive Bayes es:
- **`var_smoothing`** (GaussianNB): añade una fracción de la mayor varianza de las features a todas las varianzas. Evita divisiones por cero y mejora la estabilidad numérica. Equivalente al suavizado de Laplace para distribuciones discretas.
- **`alpha`** (BernoulliNB / ComplementNB): parámetro de suavizado de Laplace. Evita probabilidades cero para categorías no vistas en entrenamiento.
"""

# Ajuste de var_smoothing para GaussianNB
vs_values = np.logspace(-12, 0, 30)
f1_scores_vs = []

for vs in vs_values:
    gnb = GaussianNB(var_smoothing=vs)
    f1 = cross_val_score(gnb, XTrain, yTrain, cv=cv, scoring='f1').mean()
    f1_scores_vs.append(f1)

mejor_vs = vs_values[np.argmax(f1_scores_vs)]
mejor_f1 = max(f1_scores_vs)

fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogx(vs_values, f1_scores_vs, 'o-', color='#4C72B0', linewidth=2)
ax.axvline(mejor_vs, color='red', linestyle='--', alpha=0.7, label=f'Óptimo: {mejor_vs:.2e}')
ax.set_xlabel('var_smoothing (escala logarítmica)')
ax.set_ylabel('F1-score (CV 5-fold)')
ax.set_title('Ajuste de hiperparámetro var_smoothing – GaussianNB', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('nb_var_smoothing.png', dpi=150)
plt.show()

print(f'Mejor var_smoothing : {mejor_vs:.2e}')
print(f'F1-score CV óptimo  : {mejor_f1:.4f}')

# Ajuste de alpha para BernoulliNB
alpha_values = np.logspace(-3, 2, 30)
f1_scores_alpha = []

for alpha in alpha_values:
    bnb = BernoulliNB(alpha=alpha)
    f1 = cross_val_score(bnb, XTrain, yTrain, cv=cv, scoring='f1').mean()
    f1_scores_alpha.append(f1)

mejor_alpha = alpha_values[np.argmax(f1_scores_alpha)]

fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogx(alpha_values, f1_scores_alpha, 's-', color='#DD8452', linewidth=2)
ax.axvline(mejor_alpha, color='red', linestyle='--', alpha=0.7, label=f'Óptimo: alpha={mejor_alpha:.4f}')
ax.set_xlabel('alpha (escala logarítmica)')
ax.set_ylabel('F1-score (CV 5-fold)')
ax.set_title('Ajuste de hiperparámetro alpha – BernoulliNB', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('nb_alpha.png', dpi=150)
plt.show()

print(f'Mejor alpha        : {mejor_alpha:.4f}')

"""## 6. Entrenamiento y evaluación final de los tres Naive Bayes"""

# GaussianNB optimizado
gnb_opt = GaussianNB(var_smoothing=mejor_vs)
gnb_opt.fit(XTrain, yTrain)
ypred_gnb = gnb_opt.predict(XTest)
yprob_gnb = gnb_opt.predict_proba(XTest)[:, 1]

m = evaluar('GaussianNB (optimizado)', yTest, ypred_gnb, yprob_gnb)
resultados.append(m)

# BernoulliNB optimizado
bnb_opt = BernoulliNB(alpha=mejor_alpha)
bnb_opt.fit(XTrain, yTrain)
ypred_bnb = bnb_opt.predict(XTest)
yprob_bnb = bnb_opt.predict_proba(XTest)[:, 1]

m = evaluar('BernoulliNB (optimizado)', yTest, ypred_bnb, yprob_bnb)
resultados.append(m)

# ComplementNB
cnb = ComplementNB(alpha=mejor_alpha)
cnb.fit(XTrain, yTrain)
ypred_cnb = cnb.predict(XTest)
yprob_cnb = cnb.predict_proba(XTest)[:, 1]

m = evaluar('ComplementNB', yTest, ypred_cnb, yprob_cnb)
resultados.append(m)

"""## 7. Comparativo global: Naive Bayes (mejor variante) vs Redes Neuronales Fase 2

> **Instrucción:** reemplaza los valores de Fase 2 con los resultados reales de tu notebook de la Fase 2 una vez ejecutado.
"""

# Cargar métricas reales de Fase 2 (requiere resultadosFase2.csv generado por la Fase 2)
if not os.path.exists('resultadosFase2.csv'):
    raise FileNotFoundError(
        "No se encontró 'resultadosFase2.csv'. Ejecute primero el notebook de la Fase 2 "
        "para generarlo y colóquelo en el mismo directorio.")
dfFase2 = pd.read_csv('resultadosFase2.csv')

# Renombrar columnas para que coincidan con el comparativo
dfFase2['Modelo'] = dfFase2['Modelo'].str.replace('Modelo A:', 'Fase 2 – Modelo A:') \
                                    .str.replace('Modelo B:', 'Fase 2 – Modelo B:') \
                                    .str.replace('Modelo C:', 'Fase 2 – Modelo C:')
metricasFase2 = dfFase2.to_dict('records')

# Mejor variante de Naive Bayes (la de mayor F1-score)
df_nb = pd.DataFrame(resultados)
mejor_nb_idx = df_nb['F1-score'].idxmax()
mejor_nb_nombre = df_nb.loc[mejor_nb_idx, 'Modelo']
print(f'Mejor variante Naive Bayes para comparativo: {mejor_nb_nombre}')

# Tabla comparativa final
comparativo = pd.DataFrame(resultados + metricasFase2).set_index('Modelo')
print('\n', '═'*75)
print('  COMPARATIVO GLOBAL: NAIVE BAYES vs REDES NEURONALES (FASE 2)')
print('═'*75)
print(comparativo.round(4).to_string())
print('═'*75)

# Gráfico comparativo de métricas
# Usar solo los resultados de Naive Bayes (los de Fase 2 se completan después)
metricasCols = ['Accuracy', 'Precision', 'Recall', 'F1-score']
x = np.arange(len(metricasCols))
width = 0.25
colores = ['#4C72B0', '#DD8452', '#55A868']

fig, ax = plt.subplots(figsize=(12, 6))
for i, (_, row) in enumerate(df_nb.iterrows()):
    vals = [row[m] for m in metricasCols]
    bars = ax.bar(x + i*width, vals, width, label=row['Modelo'].split('(')[0].strip(),
                color=colores[i % len(colores)], alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')

ax.set_xlabel('Métrica'); ax.set_ylabel('Valor')
ax.set_title('Comparación entre variantes de Naive Bayes', fontsize=13, fontweight='bold')
ax.set_xticks(x + width); ax.set_xticklabels(metricasCols)
ax.set_ylim(0, 1.12); ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('comparativonb.png', dpi=150)
plt.show()

# Curvas ROC superpuestas
fig, ax = plt.subplots(figsize=(8, 6))
modelos_roc = [
    ('GaussianNB',   yprob_gnb, '#4C72B0'),
    ('BernoulliNB',  yprob_bnb, '#DD8452'),
    ('ComplementNB', yprob_cnb, '#55A868'),
]
for nombre, probs, color in modelos_roc:
    fpr, tpr, _ = roc_curve(yTest, probs)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{nombre} (AUC={roc_auc:.3f})')

ax.plot([0,1],[0,1],'--', color='gray', lw=1, label='Clasificador aleatorio')
ax.set_xlabel('Tasa de Falsos Positivos (FPR)')
ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
ax.set_title('Curvas ROC – Variantes de Naive Bayes', fontweight='bold', fontsize=12)
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_naive_bayes.png', dpi=150)
plt.show()

# Análisis del desbalance de clases
print('\nImpacto del desbalance de clases en Naive Bayes')
print(f'Distribución de clases en test:')
print(yTest.value_counts().rename({0:'<=50K',1:'>50K'}))
print('\nNaive Bayes modela P(C_k) directamente desde los datos de entrenamiento,')
print('por lo que absorbe el prior del desbalance 75/25 de forma natural.')
print('ComplementNB fue diseñado para compensar desbalances: asigna mayor peso')
print('a las muestras de la clase minoritaria al modelar el complemento.')

# Guardar resultados para informe
df_nb.round(4).to_csv('resultadosFase3NB.csv', index=False)
print('\n✅ Resultados exportados a resultadosFase3NB.csv')

"""## 8. Resumen final y conclusiones"""

print('=' * 60)
print('  RESUMEN FASE 3 – NAIVE BAYES')
print('=' * 60)
print(df_nb.round(4).to_string(index=False))
print('=' * 60)
print(f'\nMejor variante: {mejor_nb_nombre}')
print('\nConexión con Fase 2 (completar tras ejecutar Fase 2):')
print('  - Naive Bayes es órdenes más rápido en entrenamiento que las RN')
print('  - El supuesto de independencia limita su accuracy en features correladas')
print('  - Para este dataset, se espera que la Red Neuronal B supere a NB en F1')
print('  - Naive Bayes compensa con interpretabilidad y velocidad de inferencia')