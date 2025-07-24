import json
import boto3
import psycopg
import logging
import os
from urllib.parse import unquote_plus

from data_pipeline.loader.processing import (
    procesar_esrb,
    procesar_platforms,
    procesar_parent_platforms,
    procesar_tags,
    procesar_genres,
    procesar_stores,
    procesar_screenshots,
    procesar_ratings,
    procesar_game_status
)

# Configurar el logger para que los mensajes aparezcan en CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Leer configuración de la base de datos desde las variables de entorno
DB_HOST = os.environ.get('DB_HOST')
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASS = os.environ.get('DB_PASS')
DB_PORT = os.environ.get('DB_PORT', '5432')

# Cliente de S3 para acceder a los archivos subidos
s3_client = boto3.client('s3')

# Lambda handler para procesar JSON de RAWG:
# 1) Leer y parsear JSON desde S3
# 2) Conectar a PostgreSQL
# 3) Para cada juego, orquestar llamadas a funciones de procesamiento
# 4) Devolver statusCode y mensaje

def lambda_handler(event, context):
    logger.info("Inicio de ejecución Lambda.")
    try:
        # Obtener información del archivo subido a S3
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = unquote_plus(event['Records'][0]['s3']['object']['key'])
        logger.info(f"Archivo detectado: s3://{bucket}/{key}")

        # Cargar y parsear JSON directamente desde S3 con logs de depuración
        logger.info("Iniciando get_object para descargar JSON...")
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            logger.info("get_object completado correctamente.")
        except Exception as e:
            logger.error(f"Error en get_object: {e}", exc_info=True)
            raise

        logger.info("Leyendo cuerpo del objeto...")
        try:
            raw = response['Body'].read().decode('utf-8')
            logger.info(f"Longitud del cuerpo: {len(raw)} bytes")
            data = json.loads(raw)
            logger.info("JSON parseado correctamente.")
        except Exception as e:
            logger.error(f"Error parseando JSON: {e}", exc_info=True)
            raise

        games = data["results"]
        logger.info(f"{len(games)} juegos leídos.")

        # Conectar a la base de datos RDS PostgreSQL
        conn = psycopg.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT
        )
        logger.info("Conexión a PostgreSQL establecida correctamente.")

        with conn:
            with conn.cursor() as cur:
                cache = {
                    "esrb": set(), "platforms": set(), "parent_platforms": set(),
                    "tags": set(), "genres": set(), "stores": set(), "screenshots": set(), "ratings": set()
                }

                # Procesar cada juego
                for game in games:
                    game_id = game["id"]

                    # Comprobar si ya existe en la tabla games
                    cur.execute("SELECT id_game FROM games WHERE id_game = %s", (game_id,))
                    exists = cur.fetchone()

                    esrb = game.get("esrb_rating")
                    procesar_esrb(cur, esrb, cache)

                    if exists:
                        # Actualizar campos dinámicos si ya existe
                        cur.execute("""
                            UPDATE games SET
                                rating = %s, rating_top = %s, ratings_count = %s, metacritic = %s,
                                suggestions_count = %s, updated = %s, reviews_count = %s
                            WHERE id_game = %s
                        """, (
                            game.get("rating"), game.get("rating_top"), game.get("ratings_count"),
                            game.get("metacritic"), game.get("suggestions_count"), game.get("updated"),
                            game.get("reviews_count"), game_id
                        ))
                        logger.info(f"Actualizado juego {game_id}")
                    else:
                        # Insertar nuevo registro
                        cur.execute("""
                            INSERT INTO games (
                                id_game, slug, name, released, tba, background_image, rating,
                                rating_top, ratings_count, reviews_text_count, added, metacritic,
                                suggestions_count, updated, user_game, reviews_count, saturated_color,
                                dominant_color, clip, esrb_rating_id
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            game_id, game.get("slug"), game.get("name"), game.get("released"),
                            game.get("tba"), game.get("background_image"), game.get("rating"),
                            game.get("rating_top"), game.get("ratings_count"), game.get("reviews_text_count"),
                            game.get("added"), game.get("metacritic"), game.get("suggestions_count"),
                            game.get("updated"), game.get("user_game"), game.get("reviews_count"),
                            game.get("saturated_color"), game.get("dominant_color"), game.get("clip"),
                            esrb["id"] if esrb else None
                        ))
                        logger.info(f"Insertado juego {game_id}")

                    # Procesar relaciones
                    procesar_platforms(cur, game_id, game.get("platforms"), cache)
                    procesar_parent_platforms(cur, game_id, game.get("parent_platforms"), cache)
                    procesar_tags(cur, game_id, game.get("tags"), cache)
                    procesar_genres(cur, game_id, game.get("genres"), cache)
                    procesar_stores(cur, game_id, game.get("stores"), cache)
                    procesar_screenshots(cur, game_id, game.get("short_screenshots"), cache)
                    procesar_ratings(cur, game_id, game.get("ratings"), cache)
                    procesar_game_status(cur, game_id, game.get("added_by_status"))

        logger.info("Finalizada ejecución de Lambda.")
        return {"statusCode": 200, "body": "Carga y actualización completa"}
    except Exception as e:
        logger.error(f"Error en la ejecución de la Lambda: {e}", exc_info=True)
        return {"statusCode": 500, "body": "Error al procesar el archivo"}