# Entrenamiento de Modelos con Dask

Con Dask puedes preparar y servir tus datos a **casi cualquier** biblioteca de entrenamiento, ya sea XGBoost o frameworks de Deep Learning como TensorFlow o PyTorch. A continuación se explica cómo hacerlo:

---

## 1. XGBoost en paralelo con Dask

### 1.1 Instalación
```bash
pip install dask[complete] xgboost dask-xgboost
```

### 1.2 Ejemplo de entrenamiento distribuido
```python
import dask.dataframe as dd
from dask.distributed import Client
from xgboost.dask import DaskDMatrix, train

# 1) Inicia un cliente Dask (aprovecha cores locales del M2)
client = Client()

# 2) Carga tus datos en un Dask DataFrame
ddf = dd.read_parquet('game_features.parquet', npartitions=20)
X = ddf[feature_cols]
y = ddf['success_label']

# 3) Construye el DaskDMatrix
dtrain = DaskDMatrix(client, X, y)

# 4) Define parámetros de XGBoost
params = {
  'objective': 'multi:softprob',
  'num_class': 3,
  'eval_metric': 'mlogloss'
}

# 5) Entrena en paralelo
output = train(client, params, dtrain, num_boost_round=100)
bst = output['booster']  # tu modelo entrenado
```

### 1.3 Ventajas
- **Paralelismo**: gradientes y actualizaciones distribuidos entre particiones.
- **Escalabilidad**: añade más hilos o nodos sin cambiar código.
- **Compatibilidad**: utiliza el mismo booster de XGBoost que en configuración mononodo.

---

## 2. Redes Neuronales “out-of-core” con TensorFlow

Para entrenar redes neuronales sin cargar todo en memoria:

### 2.1 Convertir particiones de Dask a tf.data.Dataset
```python
import tensorflow as tf
import dask.dataframe as dd

# Definir función generadora de tf.data.Dataset
def df_to_dataset(df, shuffle=True, batch_size=1024):
    ds = tf.data.Dataset.from_tensor_slices((
        dict(df[feature_cols]),
        df['success_label']
    ))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    return ds.batch(batch_size)

# Leer y entrenar por partición
ddf = dd.read_parquet('game_features.parquet', npartitions=20)
model = create_your_keras_model()  # define tu modelo keras

for part in ddf.to_delayed():
    df_part = part.compute()           # convierte a pandas DataFrame
    ds_part = df_to_dataset(df_part)
    model.fit(ds_part, epochs=1)
```

### 2.2 Ventajas
- **Control total** del pipeline de datos (prefetch, cache, shuffle).
- **Escalabilidad**: trabaja con datos que no caben en memoria del MacBook.
- **Integración**: usa tf.data para producción con TF Serving o SageMaker.

---

## 3. PyTorch con DataLoader

De forma idéntica, crea un `Dataset` a partir de pandas y úsalo en un `DataLoader`:
```python
from torch.utils.data import DataLoader, TensorDataset\import torch

def part_to_dataloader(df, batch_size=128):
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df['success_label'].values, dtype=torch.long)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

for part in ddf.to_delayed():
    df_part = part.compute()
    loader = part_to_dataloader(df_part)
    for X_batch, y_batch in loader:
        model(X_batch)  # forward + backward + optimizer step
```

---

## 4. ¿Qué elegir?

- **XGBoost + Dask**: rápido de implementar, excelente en datos tabulares.
- **Red neuronal (tf.keras o PyTorch)**: para arquitecturas personalizadas o embeddings complejos.

---

**En tu MacBook Air M2**, ambos flujos funcionarán localmente aprovechando múltiples cores y particiones Dask, sin saturar la RAM.

