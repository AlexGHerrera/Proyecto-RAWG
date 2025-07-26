#!/usr/bin/env python3

import os
import glob
import json
import logging
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from tqdm import tqdm
import time
import boto3

# Importar funciones de procesamiento modular
from processing import (
    procesar_esrb,
    procesar_platforms_batch,
    procesar_parent_platforms_batch,
    procesar_tags,
    procesar_genres,
    procesar_stores,
    procesar_screenshots,
    procesar_ratings,
    procesar_game_status
)

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "loader_rawg.log")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
# Añadir logger después de configurar logging.basicConfig
logger = logging.getLogger(__name__)

load_dotenv()

# -----------------------------
# Configuración de la base de datos
# -----------------------------
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_PORT = os.getenv("DB_PORT")
S3_BUCKET = os.getenv("S3_BUCKET")

# Directorio donde están los JSON
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))

def list_all_s3_files(bucket, prefix):
    s3 = boto3.client('s3')
    continuation_token = None
    while True:
        kwargs = {'Bucket': bucket, 'Prefix': prefix}
        if continuation_token:
            kwargs['ContinuationToken'] = continuation_token
        response = s3.list_objects_v2(**kwargs)
        for obj in response.get('Contents', []):
            if obj["Key"].endswith(".json"):
                yield obj["Key"]
        if not response.get('IsTruncated'):
            break
        continuation_token = response['NextContinuationToken']

def main():
    start_time = time.time()
    try:
        with open("resume_state.txt", "r") as f:
            start_index = int(f.read())
            logger.info("Reanudando desde el índice de lote: %d", start_index)
    except:
        start_index = 0
        logger.info("Iniciando desde el principio")
    # Validar variables de entorno críticas
    for var_name, var_value in [("DB_HOST", DB_HOST), ("DB_NAME", DB_NAME), ("DB_USER", DB_USER), ("DB_PASS", DB_PASS)]:
        if not var_value:
            raise EnvironmentError(f"Variable de entorno faltante: {var_name}")

    # Conectar a PostgreSQL
    logger.info("Conectando a PostgreSQL en %s:%s/%s", DB_HOST, DB_PORT, DB_NAME)
    with psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT
    ) as conn:
        logger.info("Conexión a PostgreSQL establecida correctamente.")
        # Inicializar cache para no reinsertar metadatos en la misma ejecución
        cache = {
            "esrb": set(),
            "platforms": set(),
            "parent_platforms": set(),
            "tags": set(),
            "genres": set(),
            "stores": set(),
            "screenshots": set(),
            "ratings": set()
        }

        s3 = boto3.client('s3')
        S3_PREFIX = "games_pages/"
        logger.info("Listando archivos JSON en S3 bajo %s", S3_PREFIX)
        s3_files = list(list_all_s3_files(S3_BUCKET, S3_PREFIX))
        logger.info("Total de archivos JSON encontrados: %d", len(s3_files))

        for i, key in enumerate(tqdm(s3_files[start_index:], desc="Procesando archivos desde S3")):
            try:
                obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
                payload = json.loads(obj["Body"].read().decode("utf-8"))
                all_games = payload.get("results", [])
                logger.info("Archivo %s contiene %d juegos", key, len(all_games))
            except Exception as e:
                logger.error("Error leyendo archivo %s: %s", key, e)
                continue

            with conn.cursor() as cur:
                insert_values = []
                for game in all_games:
                    insert_values.append((
                        game.get("id"),
                        game.get("slug"),
                        game.get("name"),
                        game.get("released"),
                        game.get("playtime"),
                        game.get("tba"),
                        game.get("background_image"),
                        game.get("rating"),
                        game.get("rating_top"),
                        game.get("ratings_count"),
                        game.get("reviews_text_count"),
                        game.get("added"),
                        game.get("suggestions_count"),
                        game.get("metacritic"),
                        game.get("reviews_count"),
                        game.get("saturated_color"),
                        game.get("dominant_color"),
                        game.get("updated"),
                        game.get("user_game"),
                        game.get("clip"),
                        (game.get("esrb_rating") or {}).get("id")
                    ))

                execute_values(cur, """
                    INSERT INTO games (
                        id_game, slug, name, released, playtime, tba,
                        background_image, rating, rating_top, ratings_count,
                        reviews_text_count, added, suggestions_count,
                        metacritic, reviews_count, saturated_color,
                        dominant_color, updated, user_game, clip,
                        esrb_rating_id
                    ) VALUES %s
                    ON CONFLICT (id_game) DO UPDATE SET
                        slug = EXCLUDED.slug,
                        name = EXCLUDED.name,
                        released = EXCLUDED.released,
                        playtime = EXCLUDED.playtime,
                        tba = EXCLUDED.tba,
                        background_image = EXCLUDED.background_image,
                        rating = EXCLUDED.rating,
                        rating_top = EXCLUDED.rating_top,
                        ratings_count = EXCLUDED.ratings_count,
                        reviews_text_count = EXCLUDED.reviews_text_count,
                        added = EXCLUDED.added,
                        suggestions_count = EXCLUDED.suggestions_count,
                        metacritic = EXCLUDED.metacritic,
                        reviews_count = EXCLUDED.reviews_count,
                        saturated_color = EXCLUDED.saturated_color,
                        dominant_color = EXCLUDED.dominant_color,
                        updated = EXCLUDED.updated,
                        user_game = EXCLUDED.user_game,
                        clip = EXCLUDED.clip,
                        esrb_rating_id = EXCLUDED.esrb_rating_id;
                """, insert_values)

                # Procesar ESRB individualmente y agrupar plataformas
                platforms_data = []
                parent_platforms_data = []

                for game in all_games:
                    game_id = game.get("id")
                    try:
                        procesar_esrb(cur, game.get("esrb_rating"), cache)
                    except Exception as e:
                        logger.exception(f"Error procesando ESRB del juego {game_id}: {e}")
                    platforms_data.append((game_id, game.get("platforms")))
                    parent_platforms_data.append((game_id, game.get("parent_platforms")))

                try:
                    procesar_platforms_batch(cur, platforms_data, cache)
                    procesar_parent_platforms_batch(cur, parent_platforms_data, cache)
                except Exception as e:
                    logger.exception(f"Error procesando plataformas por lote: {e}")

                # Agrupación por tipo de entidad relacionada
                from collections import defaultdict

                tags_data = []
                genres_data = []
                stores_data = []
                screenshots_data = []
                ratings_data = []
                game_status_data = []

                for game in all_games:
                    game_id = game.get("id")
                    tags_data.append((game_id, game.get("tags")))
                    genres_data.append((game_id, game.get("genres")))
                    stores_data.append((game_id, game.get("stores")))
                    screenshots_data.append((game_id, game.get("short_screenshots")))
                    ratings_data.append((game_id, game.get("ratings")))
                    game_status_data.append((game_id, game.get("added_by_status")))

                try:
                    procesar_tags(cur, tags_data, cache)
                    procesar_genres(cur, genres_data, cache)
                    procesar_stores(cur, stores_data, cache)
                    procesar_screenshots(cur, screenshots_data, cache)
                    procesar_ratings(cur, ratings_data, cache)
                    procesar_game_status(cur, game_status_data)
                except Exception as e:
                    logger.exception(f"Error procesando entidades relacionadas por lote: {e}")

                conn.commit()
            with open("resume_state.txt", "w") as f:
                f.write(str(i + 1))
            logger.info("Archivo %s procesado correctamente", key)

        logger.info("Carga masiva completada exitosamente.")

    elapsed = time.time() - start_time
    logger.info("Tiempo total de ejecución: %.2f segundos", elapsed)

if __name__ == "__main__":
    main()
