# Guía Completa de Dask para EDA a Gran Escala

Dask es una librería de Python diseñada para escalar el ecosistema pandas/numpy hacia conjuntos de datos que no caben en memoria o requieren procesamiento paralelo. A continuación profundizamos en su funcionamiento, casos de uso, ventajas y ejemplos.

---

## 1. ¿Qué es Dask?

- Un framework de computación en paralelo y distribuida.
- Proporciona **dataframes** y **arrays** con APIs muy similares a pandas y numpy.
- Permite operaciones **lazy** (perezosas): construye un grafo de tareas y ejecuta solo cuando se invoca `.compute()`.
- Escala desde tu máquina local (multi-hilo, multi-proceso) hasta clusters de cientos de nodos.

---

## 2. Ventajas clave

1. **Out-of-Core Processing**: maneja datos que exceden la RAM, dividendo en particiones que se procesan de forma secuencial o paralela.
2. **Computación Lazy**: optimiza el grafo de tareas para evitar cómputos innecesarios.
3. **Multiprocesamiento y Multihilo**: aprovecha todos los cores de tu CPU por defecto.
4. **Escalabilidad**: desde tu laptop a un cluster en la nube (p. ej. con Dask Distributed).
5. **Interoperabilidad**: integra con scikit-learn (via dask-ml), XGBoost, RAPIDS, y almacena datos en formatos como parquet, CSV, SQL.

---

## 3. Conceptos fundamentales

- **Particiones**: fragmentos de tu DataFrame que viven en memoria por separado. Cada partición es un DataFrame pandas.
- **Dask Scheduler**: coordina la ejecución de tareas. Tiene modos: `single-threaded`, `threads`, `processes`, `distributed`.
- **Delayed**: permite paralelizar funciones arbitrarias de Python decorándolas con `@dask.delayed`.
- **Futuros (Dask Futures)**: para workflows más dinámicos en tiempo real con el cliente distribuido.

---

## 4. Instalación

```bash
pip install dask[complete]  # incluye componentes de distribuidos y compatibilidad con SQL
```

---

## 5. API principal: DataFrame

### 5.1 Crear un DataFrame Dask

- **Desde pandas**:
  ```python
  import dask.dataframe as dd
  import pandas as pd

  df = pd.read_csv('juegos.csv')
  ddf = dd.from_pandas(df, npartitions=10)
  ```
- **Lectura directa (CSV, Parquet)**:
  ```python
  ddf = dd.read_csv('data/games-*.csv')
  ddf = dd.read_parquet('parquet_folder/')
  ```
- **Desde SQL**:
  ```python
  ddf = dd.read_sql_table(
      'games',
      uri='postgresql://user:pass@host:5432/db',
      index_col='id_game',
      npartitions=20
  )
  ```

### 5.2 Operaciones y computación

- Operaciones son **lazy**:
  ```python
  # No lee datos aún
  ```

df2 = ddf[ddf['rating'] > 4] df3 = df2[['name','rating']]

````
- Ejecutar con `.compute()`:
```python
df_small = df3.compute()  # resultado en pandas DataFrame
````

- Agregaciones distribuidas:
  ```python
  mean_rating = ddf['rating'].mean().compute()
  ```

### 5.3 Ejemplo de flujo EDA con Dask

```python
import dask.dataframe as dd

# 1. Leer tabla completa en 10 particiones
ddf = dd.read_sql_table(
    'games',
    uri, index_col='id_game', npartitions=10
)

# 2. Estadísticas descriptivas
desc = ddf.describe().compute()
print(desc)

# 3. Conteos de categoría (ESRB)
esrb_counts = ddf['esrb_rating_id'].value_counts().compute()
print(esrb_counts)

# 4. Muestra 5% para correlaciones
sample = ddf.sample(frac=0.05).compute()
corr = sample.corr()
print(corr)
```

---

## 6. Integración con scikit-learn y Dask-ML

- Permite entrenar modelos en paralelo usando `dask_ml.model_selection` y `dask_ml.wrappers`. Ej:
  ```python
  from dask_ml.model_selection import train_test_split
  from dask_ml.linear_model import LogisticRegression

  X = ddf[features]
  y = ddf['success']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  lr = LogisticRegression()
  lr.fit(X_train, y_train)
  print(lr.score(X_test, y_test))
  ```
- Usa `joblib` con backend `dask` para paralelizar grid search.

---

## 7. Buenas prácticas

1. **Elegir número de particiones**: ideal 2–4 por core disponible.
2. **Persistir resultados**: `ddf = ddf.persist()` si repites varios `.compute()`.
3. **Profiling**: usar Dask Dashboard para visualizar tareas en tiempo real.
4. **Evitar operaciones costosas** sin particiones adecuadas (join, groupby enorme).

---

## 8. Conclusión

Dask es la herramienta perfecta para escalar tu EDA y modelado cuando superas límites de memoria o necesitas paralelizar en tu máquina local/cluster. Su familiaridad con pandas-ni devuelve una curva de aprendizaje suave y resultados eficientes.

