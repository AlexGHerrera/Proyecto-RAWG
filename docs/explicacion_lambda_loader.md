# Explicación técnica detallada — Lambda `lambda_loader.py`

Esta función AWS Lambda se activa automáticamente al subir un archivo JSON a un bucket S3 y carga datos de videojuegos en una base de datos PostgreSQL (RDS). A continuación se detalla su funcionamiento y diseño actualizado.

---

## 1. Estructura general del flujo Lambda

1. Se activa por eventos de subida (`PUT`) de archivos JSON en S3.
2. Descarga y parsea el JSON detectado.
3. Conecta a PostgreSQL (RDS) usando variables de entorno.
4. Para cada juego:
   - Inserta o actualiza en la tabla principal `games`.
   - Inserta o actualiza entidades relacionadas (`platforms`, `tags`, `genres`, etc.).
   - Inserta en tablas puente (relaciones n:m).

---

## 2. Configuración inicial

```python
import json, boto3, psycopg, logging, os
from urllib.parse import unquote_plus
from data_pipeline.loader.processing import (
    procesar_esrb, procesar_platforms, procesar_parent_platforms,
    procesar_tags, procesar_genres, procesar_stores,
    procesar_screenshots, procesar_ratings, procesar_game_status
)
```

### Puntos clave:
- **Variables de entorno**: Credenciales y parámetros de conexión (`DB_HOST`, `DB_NAME`, etc.).
- **boto3**: SDK de AWS para acceso a S3.
- **psycopg**: Conector eficiente para PostgreSQL.
- **logging**: Registra eventos en AWS CloudWatch.
- **unquote_plus**: Decodifica nombres de archivo con caracteres especiales.

```python
logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3_client = boto3.client('s3')
```

---

## 3. Funciones de procesamiento por entidad

Cada función modular inserta y/o actualiza registros según el tipo de entidad, usando un `cache` (diccionario de sets) para evitar duplicados dentro de una ejecución.

### `procesar_esrb`
- Inserta/actualiza clasificación por edad.
- Evita duplicados con `cache["esrb"]`.

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
raw = response['Body'].read().decode('utf-8')
data = json.loads(raw)
games = data["results"]
```

### c. Conexión a PostgreSQL

```python
conn = psycopg.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT)
```

---

## 5. Procesamiento por juego

```python
with conn:
    with conn.cursor() as cur:
        cache = { ... }
        for game in games:
            game_id = game["id"]
            cur.execute("SELECT id_game FROM games WHERE id_game = %s", (game_id,))
            exists = cur.fetchone()
            esrb = game.get("esrb_rating")
            procesar_esrb(cur, esrb, cache)
            if exists:
                # Actualiza campos dinámicos
                cur.execute("UPDATE games SET ... WHERE id_game = %s", (..., game_id))
            else:
                # Inserta nuevo registro
                cur.execute("INSERT INTO games (...) VALUES (...)", (...))
            # Procesar relaciones
            procesar_platforms(cur, game_id, game.get("platforms"), cache)
            procesar_parent_platforms(cur, game_id, game.get("parent_platforms"), cache)
            procesar_tags(cur, game_id, game.get("tags"), cache)
            procesar_genres(cur, game_id, game.get("genres"), cache)
            procesar_stores(cur, game_id, game.get("stores"), cache)
            procesar_screenshots(cur, game_id, game.get("short_screenshots"), cache)
            procesar_ratings(cur, game_id, game.get("ratings"), cache)
            procesar_game_status(cur, game_id, game.get("added_by_status"))
```

- Si el juego existe: actualiza campos dinámicos (`rating`, `metacritic`, etc.).
- Si no existe: inserta un nuevo registro completo.
- Todas las relaciones n:m se procesan modularmente.

---

## 6. Beneficios del diseño

- **Modularidad**: Cada entidad y relación tiene su propia función, facilitando el mantenimiento y evolución del pipeline.
- **ON CONFLICT DO UPDATE**: Mantiene los datos frescos y sin duplicados.
- **Cache en memoria**: Evita operaciones redundantes en una misma ejecución.
- **Logs detallados**: Toda la trazabilidad queda registrada en CloudWatch.
- **Gestión robusta de errores**: Los errores se capturan y registran con detalles para facilitar el debug.

---

## 7. Resultado final

Cada archivo JSON subido:
- Se procesa automáticamente.
- Inserta solo lo nuevo.
- Actualiza lo que ha cambiado.
- Registra todo en CloudWatch.
- Devuelve un statusCode de éxito o error según el resultado.
