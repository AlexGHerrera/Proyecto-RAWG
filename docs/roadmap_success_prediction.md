# Proceso de EDA para la tabla `games` de RAWG

Este documento detalla el **Análisis Exploratorio de Datos (EDA)** paso a paso, incluyendo la selección de features de entrada (los que proporcionará un diseñador), la estructura del workflow, librerías utilizadas y justificaciones de cada decisión. 

---

## 1. Objetivos del EDA

- **Evaluar la calidad de los datos**: detectar valores nulos, duplicados y tipos de datos incorrectos.
- **Entender distribuencias univariantes**: rangos, outliers y formas de variables clave como `rating`, `playtime`, `metacritic`.
- **Identificar correlaciones** y relaciones entre variables numéricas.
- **Analizar frecuencias** de categorías (géneros, plataformas, ESRB, tags).
- **Validar que las features de diseño** (las que el diseñador podrá proporcionar) muestren señal suficiente antes de entrenar.

---

## 2. Features de diseño (payload del diseñador)

Estos son los **campos mínimos** que el diseñador de un juego terminado, pero sin lanzar, podrá proporcionar a la API para la predicción de éxito:

1. **`genres`**: lista de géneros (e.g., `["Action", "RPG"]`).
2. **`platforms`**: lista de plataformas objetivo (e.g., `["PC", "PlayStation 5"]`).
3. **`tags`**: etiquetas descriptivas (e.g., `["open world", "multiplayer"]`).
4. **`esrb_rating`**: clasificación por edad (e.g., `"T"`).
5. **`estimated_hours`**: horas de juego estimadas (e.g., `20`).
6. **`planned_year`**: año planeado de lanzamiento (e.g., `2026`).

A partir de estos campos, derivaremos:
- Conteos: `n_genres`, `n_platforms`, `n_tags`.
- One-hot encoding de ESRB: columnas `esrb_E`, `esrb_T`, `esrb_M`, etc.
- Variables numéricas directas: `estimated_hours`, `planned_year`.

---

## 3. Estructura del workflow de EDA

1. **SQL ligero en base de datos**: consultas agregadas para revisar calidad y rangos sin traer todo a memoria.
2. **Muestreo con pandas**: extraer 50 000 filas aleatorias para EDA en memoria.
3. **EDA a escala con Dask**: procesar los ~900 000 registros completos sin saturar RAM.
4. **Visualizaciones finales**: gráficos clave para documentar insights y preparar features.

---

## 4. SQL ligero (Paso 1)

### 4.1 Total de registros y no nulos
```sql
SELECT
  COUNT(*) AS total_juegos,
  COUNT(released)       AS notnull_released,
  COUNT(rating)         AS notnull_rating,
  COUNT(metacritic)     AS notnull_metacritic
FROM games;
```

### 4.2 Top categorías
```sql
-- Géneros
SELECT g.name, COUNT(*) AS cnt
FROM game_genres gg
JOIN genres g ON gg.id_genre=g.id_genre
GROUP BY g.name
ORDER BY cnt DESC
LIMIT 10;

-- Plataformas
SELECT p.name, COUNT(*) AS cnt
FROM game_platforms gp
JOIN platforms p ON gp.id_platform=p.id_platform
GROUP BY p.name
ORDER BY cnt DESC
LIMIT 10;
```

### 4.3 Estadísticas de variables numéricas
```sql
SELECT
  MIN(playtime) AS min_playtime,
  MAX(playtime) AS max_playtime,
  AVG(playtime) AS avg_playtime,
  STDDEV(playtime) AS std_playtime,
  MIN(rating)    AS min_rating,
  MAX(rating)    AS max_rating,
  AVG(rating)    AS avg_rating,
  STDDEV(rating) AS std_rating
FROM games;
```

---

## 5. Muestreo con pandas (Paso 2)

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@host:5432/db")

sql_sample = '''
SELECT *
FROM games
TABLESAMPLE SYSTEM_ROWS(50000);
'''

df_sample = pd.read_sql(sql_sample, engine)

df_sample.info()
df_sample.describe()
```

- **Objetivo**: inspeccionar nulos, tipos de datos y estadísticas descriptivas en memoria.
- **Visualizaciones**: histograma de `rating`, boxplot de `playtime` por `esrb_rating_id`, etc.

---

## 6. EDA a escala con Dask (Paso 3)

Dask permite **procesamiento out-of-core** y **paralelización** usando la misma API de pandas. Ventajas:
- Distribuye datos en **particiones** y opera en disk/cluster.
- **Cálculos lazy**: las operaciones se planifican hasta `.compute()`.
- Escala a datasets que no caben en memoria.

```python
import dask.dataframe as dd

# Leer tabla completa en 20 particiones
ddf = dd.read_sql_table(
    table='games',
    uri='postgresql://user:pass@host:5432/db',
    index_col='id_game',
    npartitions=20
)

# 1) Tipos y total de filas
print(ddf.dtypes)
print("Total filas:", ddf.shape[0].compute())

# 2) Estadísticas de rating
stats_rating = ddf['rating'].describe().compute()
print(stats_rating)

# 3) Conteo de ESRB
esrb_counts = ddf['esrb_rating_id'].value_counts().compute()
print(esrb_counts)

# 4) Correlaciones: muestreo 10%
sample = ddf.sample(frac=0.1).compute()
corr = sample[['playtime','rating','ratings_count','metacritic']].corr()
print(corr)
```

**Por qué Dask:**
- Evita **Out-Of-Memory** al procesar millones de filas.
- Permite **pipeline integrado** con scikit-learn y Dask-ML para modelado.

---

## 7. Visualizaciones finales (Paso 4)

- **Heatmap** de correlaciones numéricas (`playtime`, `rating`, `ratings_count`, `metacritic`).
- **Barplots** de conteos para `esrb_rating_id`, top géneros y plataformas (puede extraerse vía SQL o Dask).
- **Histograma de counts**: `n_genres`, `n_platforms` (derivados de muestreos o SQL).

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlaciones numéricas")
plt.show()
```

---

## 8. Conclusión

Este proceso garantiza:
1. **Validación de la calidad** antes de entrenar.
2. **Comprobación de señal** en las features de diseño reales.
3. **Escalabilidad** con Dask para todo el dataset.
4. **Entrega de insights** visuales claros.

A continuación, estaremos listos para entrenar el modelo utilizando **únicamente** las features que el diseñador puede proporcionar (conteos y binarias derivadas de su payload) y evaluar su capacidad predictiva.

