# %% [markdown]
"""
# RAWG Game Success Prediction - Model Training

Este notebook implementa un pipeline completo de machine learning para predecir el éxito de videojuegos usando únicamente información disponible en la fase de diseño.

**Contexto de negocio**: Los estudios de videojuegos necesitan evaluar el potencial de éxito de sus proyectos antes del lanzamiento para optimizar recursos y tomar decisiones estratégicas.

**Objetivo**: Comparar 4 algoritmos (Linear Regression, Random Forest, XGBoost, Red Neuronal) para predecir success_score.
**Dataset**: 76,272 juegos filtrados por calidad con 4 features de diseño y target continuo (0-1).
**Métricas**: RMSE, MAE, R², MAPE para evaluación integral del rendimiento.
**Metodología**: Train/Validation/Test split + Hyperparameter tuning + Análisis interpretable.
"""

# %% [markdown]
"""
## 1. Imports y configuración inicial

Importamos todas las librerías necesarias para el pipeline de machine learning, incluyendo TensorFlow para implementar una red neuronal optimizada con callbacks avanzados.
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os
from scipy.stats import randint, uniform
from scipy import stats

# TensorFlow para red neuronal optimizada
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Configuración de visualización
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
sns.set_palette("husl")
np.random.seed(42)

print("Librerías importadas correctamente")

# %% [markdown]
"""
## 2. Carga y exploración del dataset

Cargamos el dataset final procesado en el EDA que contiene únicamente las 4 features de diseño más predictivas y el target success_score. Este dataset ha sido filtrado por calidad (rating > 0, added > 0) y rango temporal (2010-2024) para asegurar datos confiables. La feature esrb_rating_id fue eliminada debido a que el 77.5% de los registros tenían valores faltantes.

**Features de entrada**: n_genres, n_platforms, n_tags, release_year (4 variables de diseño)
**Target**: success_score (0-1) - métrica compuesta de rating, popularidad y engagement
**Tamaño esperado**: ~76k juegos tras filtros de calidad
**Nota**: esrb_rating_id eliminada por 77.5% de valores faltantes
"""

# %%
# Cargar dataset desde archivo Parquet (más eficiente)
# Intentar múltiples rutas posibles para mayor flexibilidad
possible_paths = [
    "/kaggle/input/rawg-training-dataset/training_dataset_final.parquet",
    "../Data/training_dataset_final.parquet",
    "./training_dataset_final.parquet"
]

df = None
for data_path in possible_paths:
    try:
        if os.path.exists(data_path):
            df = pd.read_parquet(data_path)
            print(f"Dataset cargado desde: {data_path}")
            break
    except Exception as e:
        print(f"Error cargando desde {data_path}: {e}")
        continue

if df is None:
    raise FileNotFoundError("No se pudo cargar el dataset desde ninguna ruta disponible")

print(f"Dataset cargado: {df.shape}")
print(f"Columnas: {list(df.columns)}")
print(f"Tipos de datos:")
print(df.dtypes)

# %%
# Exploración básica del dataset
print("Primeras 5 filas:")
display(df.head())

print("\nEstadísticas descriptivas:")
display(df.describe())

print("\nValores nulos:")
print(df.isnull().sum())

# %% [markdown]
"""
### Análisis de la distribución del target

Verificamos la distribución del success_score para entender el problema de regresión.
"""

# %%
# Visualización de la distribución del target
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Histograma
axes[0].hist(df['success_score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0].set_title('Distribución del Success Score')
axes[0].set_xlabel('Success Score')
axes[0].set_ylabel('Frecuencia')

# Boxplot
axes[1].boxplot(df['success_score'])
axes[1].set_title('Boxplot del Success Score')
axes[1].set_ylabel('Success Score')

# Q-Q plot para normalidad
stats.probplot(df['success_score'], dist="norm", plot=axes[2])
axes[2].set_title('Q-Q Plot (Normalidad)')

plt.tight_layout()
plt.show()

print(f"Success Score - Min: {df['success_score'].min():.4f}, Max: {df['success_score'].max():.4f}")
print(f"Success Score - Media: {df['success_score'].mean():.4f}, Std: {df['success_score'].std():.4f}")

# %% [markdown]
"""
## 3. Preparación de datos

Separamos features y target, dividimos en conjuntos de entrenamiento/validación/test y aplicamos escalado cuando sea necesario.
"""

# %%
# Separar features y target
feature_columns = ['n_genres', 'n_platforms', 'n_tags', 'release_year']
X = df[feature_columns].copy()
y = df['success_score'].copy()

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Verificar que no hay valores nulos en features
print(f"Valores nulos en features: {X.isnull().sum().sum()}")

# %%
# División en train/validation/test (70/20/10)
# Nota: No usamos stratify porque es un problema de regresión, no clasificación
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.222, random_state=42)  # 0.222 * 0.9 = 0.2

print(f"Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")

# %%
# Escalado de features (necesario para Red Neuronal)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Escalado aplicado a todas las particiones")

# %% [markdown]
"""
## 4. Definición de métricas de evaluación

Definimos funciones para calcular todas las métricas de evaluación de forma consistente.
"""

# %%
def calculate_metrics(y_true, y_pred, model_name="Model"):
    """Calcula todas las métricas de evaluación para regresión"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    # Evitamos división por cero añadiendo epsilon
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    metrics = {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }
    
    return metrics

def print_metrics(metrics):
    """Imprime métricas de forma formateada"""
    print(f"Modelo: {metrics['Model']}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAE: {metrics['MAE']:.6f}")
    print(f"R²: {metrics['R²']:.6f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print("-" * 40)

# %% [markdown]
"""
## 5. Modelo Baseline: Linear Regression

Comenzamos con un modelo simple como baseline para establecer una referencia de rendimiento.
"""

# %%
# Entrenar Linear Regression
print("Entrenando Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predicciones
y_train_pred_lr = lr_model.predict(X_train)
y_val_pred_lr = lr_model.predict(X_val)

# Métricas
lr_train_metrics = calculate_metrics(y_train, y_train_pred_lr, "Linear Regression (Train)")
lr_val_metrics = calculate_metrics(y_val, y_val_pred_lr, "Linear Regression (Validation)")

print_metrics(lr_train_metrics)
print_metrics(lr_val_metrics)

# %%
# Análisis de coeficientes de Linear Regression
coefficients = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': lr_model.coef_,
    'Abs_Coefficient': np.abs(lr_model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print("Coeficientes de Linear Regression:")
display(coefficients)

# Visualización de coeficientes
plt.figure(figsize=(10, 6))
plt.barh(coefficients['Feature'], coefficients['Coefficient'])
plt.title('Coeficientes de Linear Regression')
plt.xlabel('Coeficiente')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
## 6. Random Forest con Hyperparameter Tuning

Entrenamos Random Forest con búsqueda aleatoria de hiperparámetros para optimizar el rendimiento.
"""

# %%
# Definir espacio de búsqueda para Random Forest
rf_param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

print("Iniciando búsqueda de hiperparámetros para Random Forest...")

# RandomizedSearchCV
rf_random = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=rf_param_dist,
    n_iter=50,
    cv=3,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_random.fit(X_train, y_train)

print(f"Mejores parámetros RF: {rf_random.best_params_}")
print(f"Mejor score CV: {-rf_random.best_score_:.6f}")

# %%
# Entrenar modelo final con mejores parámetros
rf_model = rf_random.best_estimator_

# Predicciones
y_train_pred_rf = rf_model.predict(X_train)
y_val_pred_rf = rf_model.predict(X_val)

# Métricas
rf_train_metrics = calculate_metrics(y_train, y_train_pred_rf, "Random Forest (Train)")
rf_val_metrics = calculate_metrics(y_val, y_val_pred_rf, "Random Forest (Validation)")

print_metrics(rf_train_metrics)
print_metrics(rf_val_metrics)

# %%
# Feature importance de Random Forest
feature_importance_rf = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Feature Importance - Random Forest:")
display(feature_importance_rf)

# Visualización
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_rf['Feature'], feature_importance_rf['Importance'])
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importancia')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
## 7. XGBoost con Hyperparameter Tuning

Entrenamos XGBoost, conocido por su excelente rendimiento en problemas de regresión con datos tabulares.
"""

# %%
# Definir espacio de búsqueda para XGBoost
xgb_param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}

print("Iniciando búsqueda de hiperparámetros para XGBoost...")

# RandomizedSearchCV
xgb_random = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=1),
    param_distributions=xgb_param_dist,
    n_iter=50,
    cv=3,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

xgb_random.fit(X_train, y_train)

print(f"Mejores parámetros XGB: {xgb_random.best_params_}")
print(f"Mejor score CV: {-xgb_random.best_score_:.6f}")

# %%
# Entrenar modelo final con mejores parámetros
xgb_model = xgb_random.best_estimator_

# Predicciones
y_train_pred_xgb = xgb_model.predict(X_train)
y_val_pred_xgb = xgb_model.predict(X_val)

# Métricas
xgb_train_metrics = calculate_metrics(y_train, y_train_pred_xgb, "XGBoost (Train)")
xgb_val_metrics = calculate_metrics(y_val, y_val_pred_xgb, "XGBoost (Validation)")

print_metrics(xgb_train_metrics)
print_metrics(xgb_val_metrics)

# %%
# Feature importance de XGBoost
feature_importance_xgb = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Feature Importance - XGBoost:")
display(feature_importance_xgb)

# Visualización
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_xgb['Feature'], feature_importance_xgb['Importance'])
plt.title('Feature Importance - XGBoost')
plt.xlabel('Importancia')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
## 8. Red Neuronal (TensorFlow/Keras) con Hyperparameter Tuning

Implementamos una red neuronal optimizada usando TensorFlow/Keras con arquitecturas en múltiplos de 2 (64, 128) para mejor eficiencia computacional. Incluimos callbacks avanzados como EarlyStopping y ReduceLROnPlateau para optimizar el entrenamiento.

**Ventajas sobre sklearn MLPRegressor**:
- Adam optimizer con configuración avanzada
- EarlyStopping más sofisticado con restore_best_weights
- ReduceLROnPlateau para ajuste dinámico del learning rate
- Arquitecturas optimizadas (múltiplos de 2)
- Dropout para regularización
"""

# %%
# Configurar TensorFlow para reproducibilidad
tf.random.set_seed(42)

def create_neural_network(hidden_layers, dropout_rate=0.3, learning_rate=0.001):
    """Crea una red neuronal con la arquitectura especificada"""
    model = Sequential()
    
    # Primera capa oculta
    model.add(Dense(hidden_layers[0], activation='relu', input_shape=(4,)))
    model.add(Dropout(dropout_rate))
    
    # Capas ocultas adicionales
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    # Capa de salida
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid para output 0-1
    
    # Compilar modelo
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# Definir arquitecturas a probar (múltiplos de 2)
architectures = [
    [64],           # 1 capa: 64 neuronas
    [128],          # 1 capa: 128 neuronas  
    [128, 64],      # 2 capas: 128 -> 64
    [128, 64, 32],  # 3 capas: 128 -> 64 -> 32
    [64, 32]        # 2 capas: 64 -> 32
]

dropout_rates = [0.2, 0.3, 0.4]
learning_rates = [0.001, 0.003, 0.01]

print("Iniciando búsqueda de hiperparámetros para Red Neuronal (TensorFlow)...")

# %%
# Búsqueda manual de hiperparámetros (más control que RandomizedSearchCV)
best_val_score = float('inf')
best_params = None
best_model = None
results = []

# Callbacks para entrenamiento
callbacks = [
    EarlyStopping(patience=20, restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7, verbose=0)
]

# Probar diferentes combinaciones
for arch in architectures:
    for dropout in dropout_rates:
        for lr in learning_rates:
            print(f"Probando: arch={arch}, dropout={dropout}, lr={lr}")
            
            # Crear y entrenar modelo
            model = create_neural_network(arch, dropout, lr)
            
            history = model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluar en validación
            val_loss = min(history.history['val_loss'])
            
            results.append({
                'architecture': arch,
                'dropout': dropout,
                'learning_rate': lr,
                'val_loss': val_loss,
                'epochs': len(history.history['loss'])
            })
            
            # Guardar mejor modelo
            if val_loss < best_val_score:
                best_val_score = val_loss
                best_params = {'architecture': arch, 'dropout': dropout, 'learning_rate': lr}
                best_model = model
            
            print(f"Val Loss: {val_loss:.6f}, Epochs: {len(history.history['loss'])}")

print(f"\nMejores parámetros: {best_params}")
print(f"Mejor Val Loss: {best_val_score:.6f}")

# %%
# Entrenar modelo final con mejores parámetros
print("Entrenando modelo final con mejores parámetros...")
nn_model = best_model

# Predicciones
y_train_pred_nn = nn_model.predict(X_train_scaled, verbose=0).flatten()
y_val_pred_nn = nn_model.predict(X_val_scaled, verbose=0).flatten()

# Métricas
nn_train_metrics = calculate_metrics(y_train, y_train_pred_nn, "Neural Network (Train)")
nn_val_metrics = calculate_metrics(y_val, y_val_pred_nn, "Neural Network (Validation)")

print_metrics(nn_train_metrics)
print_metrics(nn_val_metrics)

# Mostrar resumen de la arquitectura final
print("\nArquitectura del modelo final:")
nn_model.summary()

# %% [markdown]
"""
## 9. Comparativa de modelos

Comparamos todos los modelos entrenados usando las métricas de validación para seleccionar el mejor.
"""

# %%
# Recopilar todas las métricas de validación
all_metrics = [lr_val_metrics, rf_val_metrics, xgb_val_metrics, nn_val_metrics]
comparison_df = pd.DataFrame(all_metrics)

print("Comparativa de modelos (Validación):")
display(comparison_df.round(6))

# %%
# Visualización comparativa de métricas
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

metrics_to_plot = ['RMSE', 'MAE', 'R²', 'MAPE']
colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon']

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i//2, i%2]
    bars = ax.bar(comparison_df['Model'], comparison_df[metric], color=colors[i], alpha=0.7)
    ax.set_title(f'Comparación - {metric}')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=45)
    
    # Añadir valores en las barras
    for bar, value in zip(bars, comparison_df[metric]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}' if metric != 'MAPE' else f'{value:.2f}%',
                ha='center', va='bottom')

plt.tight_layout()
plt.show()

# %%
# Identificar el mejor modelo
best_model_idx = comparison_df['R²'].idxmax()  # Mejor R²
best_model_name = comparison_df.iloc[best_model_idx]['Model']
print(f"Mejor modelo según R²: {best_model_name}")

# También por RMSE (menor es mejor)
best_rmse_idx = comparison_df['RMSE'].idxmin()
best_rmse_name = comparison_df.iloc[best_rmse_idx]['Model']
print(f"Mejor modelo según RMSE: {best_rmse_name}")

# %% [markdown]
"""
## 10. Evaluación final en conjunto de test

Evaluamos el mejor modelo en el conjunto de test para obtener una estimación no sesgada del rendimiento.
"""

# %%
# Seleccionar el mejor modelo (por R²)
if best_model_name == "Linear Regression (Validation)":
    best_model = lr_model
    X_test_final = X_test
elif best_model_name == "Random Forest (Validation)":
    best_model = rf_model
    X_test_final = X_test
elif best_model_name == "XGBoost (Validation)":
    best_model = xgb_model
    X_test_final = X_test
else:  # Neural Network
    best_model = nn_model
    X_test_final = X_test_scaled

# Predicción en test
if best_model_name == "Neural Network (Validation)":
    y_test_pred = best_model.predict(X_test_final, verbose=0).flatten()
else:
    y_test_pred = best_model.predict(X_test_final)
test_metrics = calculate_metrics(y_test, y_test_pred, f"{best_model_name.split('(')[0].strip()} (Test)")

print("EVALUACIÓN FINAL EN CONJUNTO DE TEST:")
print_metrics(test_metrics)

# %% [markdown]
"""
## 11. Análisis de predicciones y residuos

Analizamos las predicciones del mejor modelo para entender su comportamiento y posibles mejoras.
"""

# %%
# Gráfico de predicciones vs valores reales
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Success Score Real')
plt.ylabel('Success Score Predicho')
plt.title('Predicciones vs Valores Reales')
plt.grid(True, alpha=0.3)

# Gráfico de residuos
plt.subplot(1, 2, 2)
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals, alpha=0.6, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Success Score Predicho')
plt.ylabel('Residuos')
plt.title('Análisis de Residuos')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Distribución de residuos
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Distribución de Residuos')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot de Residuos')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Estadísticas de residuos:")
print(f"Media: {residuals.mean():.6f}")
print(f"Std: {residuals.std():.6f}")
print(f"Min: {residuals.min():.6f}")
print(f"Max: {residuals.max():.6f}")

# %% [markdown]
"""
## 12. Análisis de feature importance consolidado

Comparamos la importancia de features entre los diferentes modelos para entender qué características son más predictivas.
"""

# %%
# Consolidar feature importance de todos los modelos
importance_comparison = pd.DataFrame({
    'Feature': feature_columns,
    'Linear_Regression': np.abs(lr_model.coef_),
    'Random_Forest': rf_model.feature_importances_,
    'XGBoost': xgb_model.feature_importances_
})

# Normalizar para comparación
for col in ['Linear_Regression', 'Random_Forest', 'XGBoost']:
    importance_comparison[col] = importance_comparison[col] / importance_comparison[col].sum()

print("Feature Importance Comparativa (Normalizada):")
display(importance_comparison.round(4))

# %%
# Visualización comparativa de feature importance
plt.figure(figsize=(12, 8))

x = np.arange(len(feature_columns))
width = 0.25

plt.bar(x - width, importance_comparison['Linear_Regression'], width, label='Linear Regression', alpha=0.8)
plt.bar(x, importance_comparison['Random_Forest'], width, label='Random Forest', alpha=0.8)
plt.bar(x + width, importance_comparison['XGBoost'], width, label='XGBoost', alpha=0.8)

plt.xlabel('Features')
plt.ylabel('Importancia Normalizada')
plt.title('Comparación de Feature Importance entre Modelos')
plt.xticks(x, feature_columns, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
## 13. Conclusiones y recomendaciones

Resumen de resultados y recomendaciones para el modelo de predicción de éxito de videojuegos.
"""

# %%
print("RESUMEN DE RESULTADOS:")
print("=" * 50)
print(f"Mejor modelo: {best_model_name.split('(')[0].strip()}")
print(f"R² en test: {test_metrics['R²']:.4f}")
print(f"RMSE en test: {test_metrics['RMSE']:.6f}")
print(f"MAE en test: {test_metrics['MAE']:.6f}")
print(f"MAPE en test: {test_metrics['MAPE']:.2f}%")
print()

print("INTERPRETACIÓN:")
print(f"- El modelo explica {test_metrics['R²']*100:.1f}% de la varianza en success_score")
print(f"- Error promedio absoluto: {test_metrics['MAE']:.4f} puntos en escala 0-1")
print(f"- Error porcentual promedio: {test_metrics['MAPE']:.1f}%")
print()

print("FEATURES MÁS IMPORTANTES:")
avg_importance = importance_comparison[['Linear_Regression', 'Random_Forest', 'XGBoost']].mean(axis=1)
top_features = importance_comparison.loc[avg_importance.nlargest(3).index]
for i, (_, row) in enumerate(top_features.iterrows(), 1):
    print(f"{i}. {row['Feature']}: {avg_importance[row.name]:.3f}")

# %% [markdown]
"""
## 14. Guardado de modelos

Guardamos el mejor modelo y el scaler para uso posterior en producción.
"""

# %%
# Crear directorio para modelos si no existe
# Intentar múltiples ubicaciones según el entorno
possible_model_dirs = [
    "/kaggle/working/models",
    "../Models",
    "./models"
]

models_dir = None
for dir_path in possible_model_dirs:
    try:
        os.makedirs(dir_path, exist_ok=True)
        models_dir = dir_path
        print(f"Directorio de modelos: {models_dir}")
        break
    except Exception as e:
        print(f"No se pudo crear directorio {dir_path}: {e}")
        continue

if models_dir is None:
    models_dir = "./models"  # Fallback por defecto
    os.makedirs(models_dir, exist_ok=True)
    print(f"Usando directorio por defecto: {models_dir}")

# Guardar el mejor modelo
model_filename = f"{models_dir}/best_model_{best_model_name.split('(')[0].strip().lower().replace(' ', '_')}.joblib"
joblib.dump(best_model, model_filename)
print(f"Modelo guardado en: {model_filename}")

# Guardar scaler (necesario para Red Neuronal)
scaler_filename = f"{models_dir}/feature_scaler.joblib"
joblib.dump(scaler, scaler_filename)
print(f"Scaler guardado en: {scaler_filename}")

# Guardar métricas finales
metrics_filename = f"{models_dir}/model_metrics.joblib"
joblib.dump({
    'test_metrics': test_metrics,
    'comparison_metrics': comparison_df,
    'feature_importance': importance_comparison
}, metrics_filename)
print(f"Métricas guardadas en: {metrics_filename}")

print("\nModelos y artefactos guardados exitosamente para producción.")

# %% [markdown]
"""
## Resumen Final

### Logros del entrenamiento:
1. **Comparación exhaustiva**: 4 modelos evaluados con métricas completas
2. **Optimización de hiperparámetros**: RandomizedSearch aplicado a todos los modelos complejos
3. **Evaluación robusta**: Train/Validation/Test split para estimación no sesgada
4. **Análisis interpretable**: Feature importance y análisis de residuos
5. **Modelo productivo**: Mejor modelo guardado con artefactos necesarios

### Características del modelo final:
- **Algoritmo**: {best_model_name.split('(')[0].strip()}
- **Performance**: R² = {test_metrics['R²']:.4f}, RMSE = {test_metrics['RMSE']:.6f}
- **Features**: 4 variables de diseño (n_genres, n_platforms, n_tags, release_year)
- **Target**: success_score continuo (0-1)

### Próximos pasos:
1. **Validación temporal**: Evaluar modelo con juegos más recientes
2. **Feature engineering**: Explorar interacciones entre variables
3. **Ensemble methods**: Combinar mejores modelos para mayor robustez
4. **Deployment**: Integrar modelo en pipeline de predicción para diseñadores
5. **Monitoreo**: Establecer métricas de drift y reentrenamiento

El modelo está listo para predecir el éxito de videojuegos usando únicamente información de diseño, maximizando la utilidad para estudios de desarrollo.
"""
