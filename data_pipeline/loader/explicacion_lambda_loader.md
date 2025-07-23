
# Explicación técnica detallada — Lambda `lambda_loader.py`

Esta función AWS Lambda se activa automáticamente cuando se sube un archivo JSON al bucket S3 y carga datos de videojuegos en una base de datos PostgreSQL (RDS). A continuación se describe su funcionamiento de forma detallada y técnica.

---

## 1. Estructura general del flujo Lambda

1. Se activa al subir un archivo JSON a S3.
2. Descarga y parsea ese JSON.
3. Conecta a PostgreSQL (RDS).
4. Por cada juego:
   - Inserta o actualiza en la tabla principal `games`.
   - Inserta o actualiza entidades relacionadas (`platforms`, `tags`, `genres`, etc.).
   - Inserta en tablas puente (relaciones n:m).

---

## 2. Configuración inicial

```python
import json, boto3, psycopg2, logging, os
from urllib.parse import unquote_plus
```

### Funciones clave:
- `boto3`: SDK de AWS en Python para leer archivos de S3.
- `psycopg2`: conector para PostgreSQL.
- `logging`: registra eventos en AWS CloudWatch.
- `os.environ`: recoge credenciales desde variables de entorno.
- `unquote_plus`: decodifica caracteres especiales del nombre del archivo.

```python
logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3_client = boto3.client('s3')
```

---

## 3. Funciones por entidad

Cada función modular inserta y/o actualiza registros según el tipo de entidad, utilizando un `cache` para evitar duplicados innecesarios dentro de una ejecución.

### `procesar_esrb`
- Inserta/actualiza clasificación por edad.
- Evita repetir inserciones con `cache["esrb"]`.

### `procesar_platforms`
- Inserta plataformas y la relación `game_platforms`.
- Inserta requisitos mínimos y fecha de lanzamiento por plataforma.

### `procesar_parent_platforms`
- Inserta plataformas padre y relaciones `game_parent_platforms`.

### `procesar_tags`
- Inserta tags y relaciones `game_tags`.

### `procesar_genres`
- Inserta géneros y relaciones `game_genres`.

### `procesar_stores`
- Inserta tiendas y relaciones `game_stores`.

### `procesar_screenshots`
- Inserta screenshots e imágenes relacionadas.

### `procesar_ratings`
- Inserta ratings y relaciones `game_ratings`.

### `procesar_game_status`
- Inserta estados de usuario como "playing", "toplay", etc.

---

## 4. `lambda_handler(event, context)`

### a. Captura evento de S3

```python
bucket = event['Records'][0]['s3']['bucket']['name']
key = unquote_plus(event['Records'][0]['s3']['object']['key'])
```

### b. Descarga y parseo del JSON

```python
response = s3_client.get_object(Bucket=bucket, Key=key)
data = json.loads(response['Body'].read())
games = data["results"]
```

### c. Conexión a PostgreSQL

```python
conn = psycopg2.connect(...)
```

---

## 5. Procesamiento por juego

```python
for game in games:
    game_id = game["id"]
    cur.execute("SELECT id_game FROM games WHERE id_game = %s", (game_id,))
```

- Si existe: actualiza campos como `rating`, `metacritic`, etc.
- Si no existe: hace un `INSERT` completo.

### Relaciones procesadas:

```python
procesar_platforms(...)
procesar_parent_platforms(...)
procesar_tags(...)
procesar_genres(...)
procesar_stores(...)
procesar_screenshots(...)
procesar_ratings(...)
procesar_game_status(...)
```

---

## 6. Beneficios del diseño

- Código modular, claro y mantenible.
- `ON CONFLICT DO UPDATE` permite mantener datos frescos.
- `cache` evita inserciones duplicadas durante la ejecución.
- CloudWatch muestra trazabilidad detallada del proceso.

---

## 7. Resultado final

Cada archivo JSON subido:
- Se procesa automáticamente.
- Inserta solo lo nuevo.
- Actualiza lo que ha cambiado.
- Registra todo en CloudWatch.
