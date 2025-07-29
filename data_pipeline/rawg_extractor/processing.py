
import os
import json
import logging
import psycopg
import requests
import boto3
from datetime import datetime, timezone
from dateutil import parser
import math
from io import BytesIO
from botocore.exceptions import ClientError
import time


# Configuración del logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ===========================
# Conexión y consultas a la DB
# ===========================

def get_conn():
    try:
        conn = psycopg.connect(
            host=os.getenv('DB_HOST'),
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASS'),
            port=os.getenv('DB_PORT', 5432)
        )
        conn.autocommit = False
        return conn
    except psycopg.OperationalError:
        logger.exception("Error al conectar con la base de datos")
        raise

def get_last_updated(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(MAX(updated), NOW() - INTERVAL '1 year') FROM games;")
        return cur.fetchone()[0]

# ===========================
# Llamadas a la API de RAWG
# ===========================

RAWG_API_KEY = os.environ.get('RAWG_API_KEY')

def fetch_new_games(last_updated, page_size=40, max_pages=5):
    page = 1
    new_games = []

    while page <= max_pages:
        try:
            response = requests.get(
                'https://api.rawg.io/api/games',
                params={
                    'key': RAWG_API_KEY,
                    'ordering': '-updated',
                    'page_size': page_size,
                    'page': page
                },
                timeout=10
            )
            response.raise_for_status()
            logger.info(f"Página {page} obtenida correctamente")
        except requests.RequestException:
            logger.exception("Error al conectar con la API de RAWG")
            break

        data = response.json()
        results = data.get('results', [])
        for game in results:
            try:
                updated_at = parser.isoparse(game['updated'])
                # Asegurar que ambos sean naive (sin tzinfo) para la comparación (máxima compatibilidad)
                if updated_at.tzinfo is not None:
                    updated_at = updated_at.replace(tzinfo=None)
                if last_updated.tzinfo is not None:
                    last_updated_naive = last_updated.replace(tzinfo=None)
                else:
                    last_updated_naive = last_updated
                if updated_at > last_updated_naive:
                    new_games.append(game)
            except Exception as e:
                logger.warning(f"Error procesando la fecha de un juego: {e} | Valor: {game['updated']}")
                continue

        if not data.get('next'):
            break
        page += 1

    logger.info(f"Juegos nuevos/actualizados: {len(new_games)}")
    return new_games

# ===========================
# Subida a S3
# ===========================

# Configuración de S3
S3_PREFIX = os.environ.get('S3_PREFIX', 'updated')
s3_client = boto3.client('s3')
S3_BUCKET = os.environ.get('S3_BUCKET')

# Parámetros de reintento
MAX_RETRIES = 3
RETRY_DELAY = 5  # segundos

def upload_json_batches_to_s3(all_new_games, batch_size=100, MAX_RETRIES=3, RETRY_DELAY=5):
    """
    Divide la lista de juegos nuevos en lotes y sube cada lote como un archivo JSON separado a S3.
    Reintenta la subida hasta 3 veces si falla.
    """
    total_uploaded_games = 0
    total_batches = math.ceil(len(all_new_games) / batch_size)
    logger.info(f"Total de lotes a subir: {total_batches} (cada lote contiene hasta {batch_size} juegos)")

    for i in range(total_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(all_new_games))
        batch_games = all_new_games[start_index:end_index]

        if not batch_games:
            continue  # Saltar si el lote está vacío

        # Convertir el lote a JSON como objeto con 'results' y almacenarlo en un buffer
        buf = BytesIO()
        buf.write(json.dumps({"results": batch_games}, ensure_ascii=False).encode('utf-8'))
        buf.seek(0)

        # Generar clave para el archivo
        key = os.path.join(S3_PREFIX, f"games_{i+1}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")

        # Reintento de subida a S3
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                s3_client.upload_fileobj(buf, S3_BUCKET, key)
                logger.info(f"Subido con éxito: s3://{S3_BUCKET}/{key} (Lote {i+1} de {total_batches}, {len(batch_games)} juegos)")
                total_uploaded_games += len(batch_games)
                break  # Salir del bucle si tiene éxito
            except ClientError as e:
                logger.warning(f"[Intento {attempt}/{MAX_RETRIES}] Error al subir lote {i+1}: {e}")
                if attempt == MAX_RETRIES:
                    logger.error(f"Fallo definitivo al subir lote {i+1} tras {MAX_RETRIES} intentos.")
                    raise
                time.sleep(RETRY_DELAY)  # Esperar antes del siguiente intento

    return total_uploaded_games