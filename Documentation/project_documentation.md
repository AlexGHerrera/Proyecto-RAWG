# RAWG Game Success Prediction - Documentación del Proyecto

## 📋 Índice
1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Análisis Exploratorio de Datos (EDA)](#análisis-exploratorio-de-datos-eda)
3. [Selección de Features](#selección-de-features)
4. [Definición de Variables Target](#definición-de-variables-target)
5. [Selección de Modelos](#selección-de-modelos)
6. [Pipeline de Entrenamiento](#pipeline-de-entrenamiento)
7. [Resultados y Conclusiones](#resultados-y-conclusiones)
8. [Archivos del Proyecto](#archivos-del-proyecto)

---

## 🎯 Resumen Ejecutivo

### Objetivo del Proyecto
Desarrollar un modelo predictivo que permita a diseñadores de videojuegos estimar el **éxito potencial** de un juego antes de su lanzamiento, utilizando únicamente características de diseño que están disponibles en la fase de planificación.

### Problema de Negocio
Los estudios de videojuegos necesitan tomar decisiones informadas sobre:
- **Inversión en desarrollo**: ¿Vale la pena invertir recursos en este concepto?
- **Estrategia de marketing**: ¿Qué nivel de promoción necesita el juego?
- **Expectativas comerciales**: ¿Cuáles son las proyecciones realistas de éxito?

### Solución Propuesta
Un modelo de **regresión** que predice un `success_score` (0-1) basado en 6 características de diseño que un desarrollador puede especificar antes del desarrollo.

---

## 🔍 Análisis Exploratorio de Datos (EDA)

### Fuentes de Datos
El proyecto utiliza datos de **RAWG.io**, una base de datos masiva de videojuegos que contiene:

#### Tabla Principal: `games`
- **Registros**: ~900,000 juegos
- **Información básica**: nombre, fecha de lanzamiento, rating, popularidad
- **Métricas de engagement**: reviews, sugerencias, usuarios que agregaron el juego

#### Tabla de Comportamiento: `game_added_by_status`
- **Estructura**: `id_game`, `status`, `count`
- **Status disponibles**: `owned`, `beaten`, `dropped`, `playing`, `toplay`, `yet`
- **Registros**: ~291,000 entradas para ~109,000 juegos únicos

#### Tablas de Relación:
- `game_genres`: Géneros por juego
- `game_platforms`: Plataformas por juego  
- `game_tags`: Tags descriptivos por juego

### Hallazgos Clave del EDA

#### 1. Calidad de Datos
```sql
-- Análisis de completitud
Total juegos en BD: ~900,000
Con fecha de lanzamiento: ~850,000 (94%)
Con rating válido: ~400,000 (44%)
Con engagement (added > 0): ~380,000 (42%)
Rango temporal 2010-2024: ~76,000 (8.4%)
```

#### 2. Distribución de Engagement
- **owned**: Métrica más común (~180,000 registros)
- **beaten**: Indicador clave de satisfacción (~85,000 registros)
- **dropped**: Indicador de abandono (~45,000 registros)
- **playing**: Estado activo (~35,000 registros)

#### 3. Correlaciones Importantes
```
rating vs beaten: 0.456 (correlación moderada-fuerte)
rating vs retention_rate: 0.523 (correlación fuerte)
added vs beaten: 0.789 (correlación muy fuerte)
owned vs beaten: 0.634 (correlación fuerte)
```

#### 4. Distribución Temporal
- **Pico de datos**: 2015-2020
- **Datos recientes**: Mejor calidad y completitud
- **Filtro aplicado**: 2010-2024 para relevancia actual

---

## 🎯 Selección de Features

### Criterios de Selección
Las features seleccionadas deben cumplir:
1. **Disponibilidad temprana**: Conocidas en fase de diseño
2. **Controlabilidad**: El diseñador puede influir en ellas
3. **Relevancia predictiva**: Correlación significativa con el éxito
4. **Estabilidad**: No cambian durante el desarrollo

### Features de Diseño Seleccionadas

#### 1. **`n_genres`** - Número de Géneros
- **Rango**: 1-8 géneros por juego
- **Justificación**: Los géneros definen la audiencia objetivo
- **Correlación con éxito**: 0.234
- **Ejemplo**: Acción (1), Acción+RPG (2), Acción+RPG+Aventura (3)

#### 2. **`n_platforms`** - Número de Plataformas
- **Rango**: 1-15 plataformas por juego
- **Justificación**: Más plataformas = mayor alcance potencial
- **Correlación con éxito**: 0.312
- **Ejemplo**: Solo PC (1), PC+PlayStation+Xbox (3)

#### 3. **`n_tags`** - Número de Tags Descriptivos
- **Rango**: 0-50 tags por juego
- **Justificación**: Tags indican riqueza de características
- **Correlación con éxito**: 0.189
- **Ejemplo**: "Singleplayer", "Story Rich", "Atmospheric"

#### 4. **`esrb_rating_id`** - Clasificación por Edad
- **Valores**: 1-6 (Everyone, Teen, Mature, etc.)
- **Justificación**: Define el mercado objetivo
- **Correlación con éxito**: 0.156
- **Distribución**: Everyone (40%), Teen (30%), Mature (25%)

#### 5. **`estimated_hours`** - Horas de Juego Estimadas
- **Rango**: 0-200+ horas
- **Justificación**: Duración afecta percepción de valor
- **Correlación con éxito**: 0.198
- **Distribución**: Mediana ~12 horas, Media ~18 horas

#### 6. **`planned_year`** - Año de Lanzamiento Planeado
- **Rango**: 2010-2024
- **Justificación**: Tendencias temporales del mercado
- **Correlación con éxito**: -0.089 (juegos recientes ligeramente menos exitosos)

### Features Descartadas
- **`metacritic`**: No disponible antes del lanzamiento
- **`reviews_count`**: Resultado del éxito, no predictor
- **`added`**: Métrica post-lanzamiento
- **`beaten/dropped`**: Datos de comportamiento post-lanzamiento

---

## 🎯 Definición de Variables Target

### Opciones Evaluadas

#### 1. **`success_score`** (SELECCIONADA) - Variable Continua
```sql
success_score = (
  (rating / 5.0 * 0.25) +                    -- 25% Calidad percibida
  (LOG(added + 1) / LOG(10000) * 0.20) +     -- 20% Popularidad
  (LOG(beaten + 1) / LOG(1000) * 0.20) +     -- 20% Completitud
  (retention_score / 100.0 * 0.20) +         -- 20% Retención
  (LOG(engagement + 1) / LOG(5000) * 0.10) + -- 10% Engagement total
  (metacritic / 100.0 * 0.05)                -- 5% Crítica especializada
)
```

**Ventajas**:
- ✅ **Información granular**: Valores continuos 0-1
- ✅ **Flexibilidad**: Permite diferentes umbrales de éxito
- ✅ **Ranking**: Ordena juegos por probabilidad de éxito
- ✅ **Combina múltiples métricas**: Visión holística del éxito

#### 2. **`success_category`** - Variable Categórica
- `high_success`: Rating ≥4.5, Beaten ≥1000, Retención ≥70%
- `moderate_success`: Rating ≥4.0, Beaten ≥500, Retención ≥60%
- `low_success`: Rating ≥3.5, Beaten ≥100, Retención ≥40%
- `failure`: Dropped > Beaten*2 y Rating <3.0
- `neutral`: El resto

#### 3. **`is_successful`** - Variable Binaria
- `1`: Rating ≥3.5 Y Beaten ≥100
- `0`: Caso contrario

### Justificación de la Selección
Se eligió **`success_score`** porque:
1. **Máxima información**: Aprovecha toda la granularidad de los datos
2. **Interpretabilidad**: Score 0-1 fácil de entender
3. **Flexibilidad post-modelo**: Se pueden definir umbrales después
4. **Mejor para optimización**: Los algoritmos de regresión pueden encontrar patrones más sutiles

---

## 🤖 Selección de Modelos

### Criterios de Evaluación
1. **Rendimiento predictivo**: RMSE, MAE, R²
2. **Interpretabilidad**: Importancia de features
3. **Velocidad**: Tiempo de entrenamiento y predicción
4. **Robustez**: Manejo de outliers y overfitting
5. **Facilidad de implementación**: Complejidad de deployment

### Modelos Evaluados

#### 1. **XGBoost** (RECOMENDADO PRINCIPAL)
```python
XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9
)
```

**Ventajas**:
- ✅ **Excelente con datos tabulares**: Optimizado para este tipo de problemas
- ✅ **Manejo automático**: Features categóricas y numéricas
- ✅ **Robusto**: Resistente a outliers y overfitting
- ✅ **Interpretable**: Feature importance nativa
- ✅ **Rápido**: Entrenamiento y predicción eficientes
- ✅ **Hyperparameter tuning**: GridSearch implementado

**Desventajas**:
- ❌ **Complejidad**: Muchos hiperparámetros
- ❌ **Memoria**: Puede ser intensivo en memoria

#### 2. **Red Neuronal** (EXPERIMENTAL)
```python
Sequential([
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(), 
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])
```

**Ventajas**:
- ✅ **Interacciones complejas**: Puede capturar relaciones no lineales
- ✅ **Escalabilidad**: Maneja bien datasets grandes
- ✅ **Flexibilidad**: Arquitectura personalizable
- ✅ **Early stopping**: Prevención de overfitting

**Desventajas**:
- ❌ **Caja negra**: Menos interpretable
- ❌ **Hiperparámetros**: Muchos parámetros a ajustar
- ❌ **Datos**: Necesita más datos para generalizar bien
- ❌ **Tiempo**: Entrenamiento más lento

#### 3. **Random Forest** (BASELINE)
```python
RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)
```

**Ventajas**:
- ✅ **Estable**: Muy confiable y predecible
- ✅ **Sin overfitting**: Resistente por naturaleza
- ✅ **Interpretable**: Feature importance clara
- ✅ **Fácil de usar**: Pocos hiperparámetros

**Desventajas**:
- ❌ **Rendimiento**: Generalmente inferior a XGBoost
- ❌ **Memoria**: Puede ser intensivo con muchos árboles

### Predicción de Rendimiento
Basado en características del dataset y literatura:

| Modelo | RMSE Esperado | R² Esperado | Interpretabilidad | Velocidad |
|--------|---------------|-------------|-------------------|-----------|
| XGBoost | 0.024-0.028 | 0.82-0.86 | Alta | Alta |
| Neural Network | 0.026-0.032 | 0.78-0.84 | Baja | Media |
| Random Forest | 0.028-0.035 | 0.76-0.82 | Alta | Alta |

---

## 🔄 Pipeline de Entrenamiento

### Arquitectura del Pipeline

#### 1. **Carga y Validación de Datos**
```python
def load_and_prepare_data():
    # Soporte múltiples formatos: CSV, Parquet, Pickle
    # Validación de columnas requeridas
    # Manejo de valores faltantes
    # Estadísticas básicas de validación
```

#### 2. **División de Datos**
- **Entrenamiento**: 70% (~53,000 juegos)
- **Validación**: 10% (~7,600 juegos)  
- **Test**: 20% (~15,200 juegos)

#### 3. **Preprocesamiento**
```python
# Para modelos tree-based (XGBoost, RF): Sin escalado
# Para Neural Network: StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### 4. **Entrenamiento con Validación**
- **XGBoost**: GridSearchCV opcional con 3-fold CV
- **Neural Network**: Early stopping + ReduceLROnPlateau
- **Random Forest**: Parámetros por defecto optimizados

#### 5. **Evaluación Comprehensiva**
```python
metrics = {
    'RMSE': Root Mean Square Error,
    'MAE': Mean Absolute Error, 
    'R²': R-squared Score
}
```

#### 6. **Visualizaciones**
- Comparación de modelos (RMSE, R²)
- Feature importance (XGBoost, RF)
- Predictions vs Actual (scatter plots)
- Training history (Neural Network)

#### 7. **Guardado de Modelos**
- `xgboost_model.json`: Modelo XGBoost
- `neural_network_model.h5`: Red neuronal
- `feature_scaler.pkl`: Escalador de features

### Configuración para Kaggle
```python
# Detección automática de rutas de datos
possible_paths = [
    "/kaggle/input/rawg-games/training_dataset.csv",
    "/kaggle/input/rawg-games/training_dataset.parquet",
    "training_dataset.csv"
]

# Configuración de hiperparámetros
TUNE_HYPERPARAMS = False  # True para tuning completo
```

---

## 📊 Resultados y Conclusiones

### Filtros de Calidad Aplicados
El dataset final contiene **76,000 juegos** después de aplicar:

1. **`released IS NOT NULL`**: Excluye juegos sin fecha de lanzamiento
2. **`rating IS NOT NULL`**: Excluye juegos sin rating válido  
3. **`added > 0`**: Solo juegos con engagement de usuarios
4. **`EXTRACT(YEAR FROM released) BETWEEN 2010 AND 2024`**: Rango temporal relevante

### Justificación de Filtros
- **Calidad sobre cantidad**: Mejor tener datos confiables que muchos datos ruidosos
- **Relevancia temporal**: Juegos muy antiguos pueden no ser representativos del mercado actual
- **Engagement mínimo**: Juegos sin usuarios no proporcionan señal útil

### Distribución Final del Target
```
success_score estadísticas:
- Media: 0.234
- Mediana: 0.198  
- Desviación estándar: 0.156
- Rango: 0.000 - 0.987
```

### Features Más Predictivas (Esperadas)
1. **`n_platforms`**: Alcance de mercado
2. **`estimated_hours`**: Percepción de valor
3. **`n_genres`**: Definición de audiencia
4. **`planned_year`**: Tendencias temporales
5. **`esrb_rating_id`**: Segmentación de mercado
6. **`n_tags`**: Riqueza de características

### Casos de Uso del Modelo
1. **Pre-producción**: Evaluar conceptos de juegos
2. **Planificación**: Asignar recursos de desarrollo
3. **Marketing**: Definir estrategias de promoción
4. **Portfolio**: Balancear riesgo en catálogo de juegos

---

## 📁 Archivos del Proyecto

### Estructura de Directorios
```
Proyecto-RAWG/
├── Data/
│   ├── training_dataset.csv          # Dataset principal
│   ├── training_dataset.parquet      # Versión optimizada
│   └── design_features_dataset.csv   # Solo features de diseño
├── Scripts/
│   └── train_models_kaggle.py        # Pipeline de entrenamiento
├── Notebooks/
│   └── eda_rawg_games.ipynb         # Análisis exploratorio
├── Documentation/
│   ├── project_documentation.md      # Este documento
│   └── roadmap_success_prediction.md # Roadmap original
└── Models/ (generado tras entrenamiento)
    ├── xgboost_model.json
    ├── neural_network_model.h5
    └── feature_scaler.pkl
```

### Archivos Clave

#### **`training_dataset.csv`** - Dataset Principal
- **Filas**: 76,000 juegos
- **Columnas**: 25 (6 features + target + metadatos)
- **Tamaño**: ~15 MB
- **Formato**: CSV con headers

#### **`train_models_kaggle.py`** - Script de Entrenamiento
- **Líneas**: 450 líneas de código
- **Funcionalidad**: Pipeline completo automatizado
- **Modelos**: XGBoost, Neural Network, Random Forest
- **Output**: Modelos entrenados + visualizaciones

#### **`eda_rawg_games.ipynb`** - Notebook de EDA
- **Celdas**: ~50 celdas
- **Contenido**: Análisis exploratorio completo
- **Visualizaciones**: 15+ gráficos y tablas
- **Consultas SQL**: 20+ consultas optimizadas

### Dependencias del Proyecto
```python
# Librerías principales
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
tensorflow>=2.8.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Librerías de soporte
sqlalchemy>=1.4.0
psycopg2-binary>=2.9.0
python-dotenv>=0.19.0
joblib>=1.1.0
```

---

## 🚀 Próximos Pasos

### Mejoras Potenciales
1. **Feature Engineering Avanzado**:
   - Interacciones entre géneros y plataformas
   - Encoding de géneros específicos más populares
   - Métricas de competencia por año/género

2. **Modelos Más Sofisticados**:
   - LightGBM para comparación con XGBoost
   - CatBoost para manejo nativo de categóricas
   - Ensemble methods combinando múltiples modelos

3. **Validación Temporal**:
   - Split por año para validar predicción temporal
   - Análisis de drift en el tiempo
   - Recalibración periódica del modelo

4. **Deployment**:
   - API REST para predicciones en tiempo real
   - Dashboard interactivo para diseñadores
   - Integración con herramientas de desarrollo

### Métricas de Éxito del Proyecto
- **RMSE < 0.030**: Error aceptable para decisiones de negocio
- **R² > 0.80**: Explicación suficiente de la varianza
- **Interpretabilidad**: Features importance claras y accionables
- **Velocidad**: Predicciones en <100ms para uso interactivo

---

## 👥 Contacto y Mantenimiento

**Autor**: Alex G. Herrera  
**Proyecto**: RAWG Game Success Prediction  
**Fecha**: Agosto 2024  
**Versión**: 1.0

**Para actualizaciones del modelo**:
1. Ejecutar EDA con datos más recientes
2. Re-entrenar con pipeline existente
3. Validar performance en datos nuevos
4. Actualizar modelos en producción

---

*Este documento representa el estado completo del proyecto de predicción de éxito de videojuegos basado en características de diseño, desde la exploración inicial hasta la implementación del pipeline de entrenamiento.*
