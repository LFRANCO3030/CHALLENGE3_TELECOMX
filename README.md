# Acerca del Proyecto
## Objetivo
 Preparar los datos para el modelado (tratamiento, codificación, normalización).
 Realizar análisis de correlación y selección de variables.
 Entrenar dos o más modelos de clasificación.
 Evaluar el rendimiento de los modelos con métricas.
 Interpretar los resultados, incluyendo la importancia de las variables.
 Crear una conclusión estratégica señalando los principales factores que influyen en la cancelación.

# Descripcion del Caso
Empresa del sector de telecomunicaciones que brinda servicios al mercado, en la actualizada presenta una preocupante tasa de evasion de clientes y desea identificar el motivo y estrategias de solucion 

# Ambiente de Desarrollo
Github : Control de versiones
Google Collab : Editor de Python
### Import para analisis de datos y visualizacion 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
### Import para Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
### Import de oversampling SMOTE
from imblearn.over_sampling import SMOTE

# Extraccion
Clonamos repositorio de Datos de Github desde https://github.com/ingridcristh/challenge3-data-science-LATAM/blob/main/TelecomX_data.csv

# Preparacion de Datos(PR)
## PR1. Eliminacion de Columna customerID
if 'customerID' in df.columns:
    df.drop(columns=['customerID'], inplace=True)

## PR2. Transformacion de Variables categoricas
### Identificacion de Variables Categoricas
valores_unicos = {}
for column in df.columns:
    valores_unicos[column] = df[column].unique()

for column, unique_values in valores_unicos.items():
    print(f"Valores Unicos por columna '{column}':")
    print(unique_values)
    print("-" * 30)

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('/content/TelecomX_data.csv')

### Separación X,y
X = df.drop('Churn', axis=1)
y = df['Churn']

### Columnas categóricas (ajusta según tu dataset)
columnas_categoricas = [
    'gender', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod', 'MultipleLines'

## PR3. Verificacion de Proporcion de Cancelacion
### Calcular conteos
conteo_churn = df['Churn'].value_counts()
print("Conteo de clases:\n", conteo_churn)

### Calcular proporciones
proporcion_churn = df['Churn'].value_counts(normalize=True)
print("\nProporción de clases:\n", proporcion_churn)

## PR4. Balanceo de Clases
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

### 1) 'Churn' es la columna objetivo:

X = df_ml.drop('Churn', axis=1)
y = df_ml['Churn']

### 2) Aplicar SMOTE para balancear
oversampling = SMOTE(random_state=42)
X_balanceada, y_balanceada = oversampling.fit_resample(X, y)

print(y_balanceada.value_counts(normalize=True))

### 3) Modelo
modelo = DecisionTreeClassifier(max_depth=10, random_state=42)

### 4) Validación cruzada estratificada
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_resultados = cross_validate(modelo, X_balanceada, y_balanceada, cv=skf, scoring='recall')

print("Recall en cada fold:", cv_resultados['test_score'])
print("Recall promedio:", cv_resultados['test_score'].mean())

## PR5. Normalizacion o Estandarizacion
Se recomienda aplicar normalización o estandarización solo si se van a usar modelos sensibles a la escala, como KNN, SVM, Regresión Logística o redes neuronales. En cambio, para nuestro caso que usare modelos basados en árboles como Decision Tree, Random Forest o XGBoost, la escala de los datos no afecta el rendimiento, por lo que no es necesario escalar las variables.

# Correlacion y seleccion de Variables (CORR)
## CORR1. Analisis de Correlacion
import seaborn as sns
import matplotlib.pyplot as plt

columnas_numericas = ['tenure', 'Charges.Monthly', 'Charges.Total', 'Churn']
df_corr = df[columnas_numericas].corr()

plt.figure(figsize=(8,6))
sns.heatmap(df_corr, annot=True, cmap='coolwarm')
plt.title('Matriz de correlación - variables numéricas')
plt.show()

## CORR2. Analisis Dirigido
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


### Boxplot 1: Tenure vs Churn 
plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Tiempo de Contrato vs Cancelación (Churn)')
plt.xlabel('Churn (0=Activo, 1=Cancelado)')
plt.ylabel('Meses de contrato (tenure)')
plt.show()

### Boxplot 2: Total Charges vs Churn 
plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='Charges.Total', data=df)
plt.title('Gasto Total vs Cancelación (Churn)')
plt.xlabel('Churn (0=Activo, 1=Cancelado)')
plt.ylabel('Gasto Total')
plt.show()

### Scatter plot combina tenure y total charges vs Churn 
plt.figure(figsize=(8,6))
sns.scatterplot(
    x='tenure',
    y='Charges.Total',
    hue='Churn',
    data=df,
    alpha=0.6
)
plt.title('Scatter Plot: Tenure vs Total Charges (coloreado por Churn)')
plt.show()

## Separacion de Datos (SEP)
### SEP1 Separacion de  Datos:  70% para entrenamiento y 30% para prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
### SEP2  Separacion de  Datos:  80% para entrenamiento y 20% para prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
# Creacion de Modelos (CM)
## CM1 Modelo de Regresión Logística  
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

### 1) Cargar dataset final
df = pd.read_csv('final_ml.csv')

### 2) Eliminar columnas ID si aún existiera
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])
### También elimina cualquier columna no numérica:
df = df.select_dtypes(exclude=['object'])

### 3) Separar X e y
X = df.drop('Churn', axis=1)
y = df['Churn']

### 4) split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

## CM2 Random Forest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

### 1) Cargar archivo
df = pd.read_csv('final_ml.csv')

### 2) Convertir posibles columnas 'ID' o textos en numérico / eliminar
### Elimina customerID si está
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

### 3) Elimina todas las columnas tipo string (por si quedó alguna con formato texto)
df = df.select_dtypes(exclude=['object'])

### 4) X, y
X = df.drop('Churn', axis=1)
y = df['Churn']

### 5) split 70/30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

### 6) RandomForest
rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

### 7) Eval
y_pred = rf_model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 5) seleccionar auto columnas numéricas
col_num = X_train.columns  # todas ya son numéricas

# 6) scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 7) logistic regression
log_reg = LogisticRegression(max_iter=200, class_weight='balanced')
log_reg.fit(X_train_scaled, y_train)

y_pred = log_reg.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Evaluacion de Modelos (EM)

## EM1 Evaluacionde Modelo de Regresion Logistica
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

### 1) Cargar el archivo final ML
df = pd.read_csv('final_ml.csv')

### 2) Quitar columna de texto / ID si existe
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

### 3) Seleccionar solo columnas numéricas (todo debe ser numérico para este modelo)
df = df.select_dtypes(exclude=['object'])

### 4) Separar X e y
X = df.drop('Churn', axis=1)
y = df['Churn']

### 5) Split 70/30 y estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

### 6) Detectar columnas numéricas automáticamente (ahora todas lo son)
columnas_numericas = X_train.columns

### 7) Normalización
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[columnas_numericas] = scaler.fit_transform(X_train[columnas_numericas])
X_test_scaled[columnas_numericas] = scaler.transform(X_test[columnas_numericas])

### 8) Modelo Logístico
log_reg = LogisticRegression(max_iter=300, class_weight='balanced')
log_reg.fit(X_train_scaled, y_train)

### 9) Predicciones
y_pred = log_reg.predict(X_test_scaled)

### 10) Métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
matriz = confusion_matrix(y_test, y_pred)

### 11) Imprimir resultados
print("=== Evaluación Regresión Logística ===")
print("Exactitud (Accuracy):", round(accuracy, 4))
print("Precisión:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
print("\nMatriz de Confusión:")
print(matriz)
print("\nReporte Clasificación:\n")
print(classification_report(y_test, y_pred))

## EM2 Evaluacion de Modelo RandomForest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

### 1) Cargar el archivo
df = pd.read_csv('final_ml.csv')

### 2) Eliminar columna ID si existe
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

### 3) Asegurar que todo sea numérico y quitar columnas tipo texto
df = df.select_dtypes(exclude=['object'])

### 4) Separar X e y
X = df.drop('Churn', axis=1)
y = df['Churn']

### 5) Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

### 6) Crear y entrenar modelo RandomForest
rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)

### 7) Predicciones
y_pred = rf_model.predict(X_test)

### 8) Métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
matriz = confusion_matrix(y_test, y_pred)

### 9) Imprimir resultados
print("=== Evaluación RandomForest ===")
print("Exactitud (Accuracy):", round(accuracy, 4))
print("Precisión:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
print("\nMatriz de Confusión:")
print(matriz)
print("\nReporte Clasificación:\n")
print(classification_report(y_test, y_pred))

## EM3 Analisis Comparativo
Tras comparar el rendimiento de los dos modelos, se observa que Random Forest obtiene mejores resultados en casi todas las métricas (exactitud, precisión y F1-score). Esto indica que el modelo es capaz de capturar patrones más complejos en los datos que la regresión logística, la cual tiende a simplificar el problema.
La regresión logística mostró un rendimiento más modesto, con un posible ligero underfitting, ya que no logra adaptarse a relaciones no lineales entre las variables. Sin embargo, mantiene una interpretación más transparente de los coeficientes y contribuciones de cada variable al churn.
Por otro lado, Random Forest ofrece un mejor equilibrio entre precisión y recall. Aunque podría existir un pequeño riesgo de overfitting si la cantidad de árboles o la profundidad no se regulan, las métricas de prueba no muestran una caída dramática, por lo que el modelo se considera generalizable. En caso de detectarse overfitting, se podrían ajustar hiperparámetros como max_depth, min_samples_leaf o aplicar validación cruzada con búsqueda de parámetros.
En conclusión, Random Forest es el modelo con mejor desempeño en este caso, manteniendo una buena capacidad predictiva sin necesidad de normalizar datos. La regresión logística sigue siendo útil como modelo base de comparación o cuando se requiere mayor interpretabilidad.

# Analisis de la Importancia de la Variables (AVA)
#Coeficiente positivo → aumenta probabilidad de cancelar
#Coeficiente negativo → disminuye probabilidad de cancela
import numpy as np

coeficientes = pd.DataFrame({
    'Variable': X.columns,
    'Coeficiente': log_reg.coef_[0]
}).sort_values(by='Coeficiente', ascending=False)

print(coeficientes.head(10))
print(coeficientes.tail(10))

Interpretacion
1. Contract_Month-to-month: positivo mes a mes eleva el churn
2. tenure: negativo clientes antiguos cancelan menos
3. TechSupport_No: positivo falta de soporte técnico influye en cancelación

## AVA2 KNN
KNN no da coeficientes ni pesos claros. Lo que puedes hacer es:
Ver la distancia en espacio normalizado: las variables con mayor escala o peso dominante influyen más
Usual: las variables más relevantes son tenure, Charges.Total, Charges.Monthly porque definen mejor la proximidad de clientes.
Tip opcional: usar SelectKBest() o métodos de reducción dimensional antes de KNN para ver qué variables tienen mayor varianza o impacto en clasificación

## AVA3 Ramdom Forest
import numpy as np

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]  # ordenar de mayor a menor

# Top 15 variables más importantes
top_vars = [(X.columns[i], importances[i]) for i in indices[:15]]
for nombre, imp in top_vars:
    print(f"{nombre}: {round(imp,4)}")

Interpretacion
tenure y MonthlyCharges suelen ser las más importantes
Variables como Contract_Month-to-month aparecen también arriba
También variables de servicios relacionados con Internet tienden a tener peso semi-alto

## AVA4 SVM
from sklearn.svm import LinearSVC  # o puedes usar SVC(kernel='linear')

# Entrenar el modelo SVM lineal
svm_model = LinearSVC(class_weight='balanced', max_iter=2000)
svm_model.fit(X_train_scaled, y_train)

coef_svm = pd.Series(svm_model.coef_[0], index=X.columns).sort_values(ascending=False)
print(coef_svm.head())

## AVA5 Conclusion
Después de evaluar múltiples modelos, las variables más influyentes en la cancelación de clientes coinciden entre modelos: principalmente tenure, MonthlyCharges, TotalCharges (en sentido inverso), y ciertas variables categóricas como Contract_Month-to-month, TechSupport_No o OnlineSecurity_No.
En Regresión Logística y SVM lineal, estos aparecen con coeficientes significativos. En RandomForest, la variable tenure y las relacionadas al tipo de contrato están entre las más altas en importancia. KNN no provee importancia explícita, pero las variables numéricas con mayor dispersión (como tenure y pagos mensuales) tienen más impacto sobre la cercanía entre observaciones.
Esto confirma que la cancelación está asociada principalmente a clientes con poco tiempo de permanencia, contratos mes a mes, y facturación mensual elevada.

# Informe Final (IF)
## IF2 Modelos Evaluados
Se entrenaron y compararon varios modelos, entre ellos:
Regresión Logística (modelo lineal base, interpretativo)
Random Forest (modelo no lineal robusto con mayor rendimiento)
SVM lineal (opcionalmente)
KNN (opcional: sensible a la normalización)
La comparación se centró en cuatro métricas principales: Accuracy, Precision, Recall y F1-score, además de la matriz de confusión.

## IF4 Variables Influyentes
### IF4.1 Segun Regresion Logistica
Variables con mayor coeficiente positivo :aumentan la probabilidad de cancelación:
Contract_Month-to-month
InternetService_Fiber optic
TechSupport_No
OnlineSecurity_No

Variables con coeficiente negativo :reducen la probabilidad de cancelación:
tenure (tiempo con la empresa)
Having partner / dependable
Contract_Two year

### IF4.2 Segun Random Forest
tenure (meses de contrato)
MonthlyCharges
Contract_Month-to-Month
TotalCharges
TechSupport_No / OnlineSecurity_No
InternetService_Fiber optic

## IF5 Principales Factores
Tiempo de permanencia bajo: Clientes con pocos meses de contrato tienden a cancelar.
Tipo de contrato: Los contratos "Month-to-month" registran una tasa de cancelación muy alta en comparación con contratos de 1 o 2 años.
Servicios adicionales de soporte y seguridad: Los clientes sin Tech Support o sin Online Security cancelan con mayor frecuencia → sensación de falta de valor añadido.
Pagos mensuales altos: Cuotas mensuales elevadas se asocian a mayor insatisfacción o cancelación (especialmente en fibra óptica).
Falta de acompañamiento personalizado y retención temprana: Muchos clientes se van en los primeros meses por no haber fidelización activa.

## Recomendaciones
Estratégicas
Fidelizar a clientes de corta permanencia: Ofrecer promociones específicas o descuentos en los primeros 6 meses.
Incentivar contratos de largo plazo: Ofrecer beneficios adicionales para quienes migran a contratos de 1 año o 2 años.
Mejorar soporte técnico y servicios adicionales: Incluir asistencia técnica y seguridad como parte del paquete base mejora retención.

Segmentación preventiva
Monitorear clientes con MonthlyCharges altos y tenure < 6 meses.
Detectar clientes “en riesgo” por ausencia de servicios complementarios (TechSupport_No).
Segmentación de churn propenso: Listas de clientes de alto riesgo y contactar proactiva/automatizadamente.


