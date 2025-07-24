#!/usr/bin/env python3
"""
massive_loader.py

Carga masiva de todos los JSON locales de RAWG a PostgreSQL.
Utiliza las funciones de data_pipeline/loader/processing.py
para mantener el pipeline idempotente y evitar duplicados.
"""

import os
import glob
import json
import logging
import psycopg

# Importar funciones de procesamiento modular
from processing import (
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

# Configurar logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# -----------------------------
# Configuración de la base de datos
# -----------------------------
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_NAME = os.environ.get("DB_NAME", "rauwgprueba")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASS = os.environ.get("DB_PASS", "SupernenA")
DB_PORT = os.environ.get("DB_PORT", "5432")

# Directorio donde están los JSON
DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    os.pardir, os.pardir, "data", "raw"
)

def main():
    # Conectar a PostgreSQL
    logger.info("Conectando a PostgreSQL en %s:%s/%s", DB_HOST, DB_PORT, DB_NAME)
    conn = psycopg.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT
    )
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

    # Recorrer todos los archivos JSON en DATA_DIR
    pattern = os.path.join(DATA_DIR, "games_page_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning("No se encontraron JSON en %s", DATA_DIR)
        return

    with conn.cursor() as cur:
        for filepath in files:
            logger.info("Procesando archivo %s", filepath)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as e:
                logger.error("Error leyendo JSON %s: %s", filepath, e)
                continue

            games = payload.get("results", [])
            logger.info("Encontrados %d juegos en %s", len(games), os.path.basename(filepath))

            for game in games:
                game_id = game.get("id")
                # Upsert en tabla games con todos los campos relevantes
                cur.execute("""
                    INSERT INTO games (
                        id_game, slug, name, released, playtime, tba,
                        background_image, rating, rating_top, ratings_count,
                        reviews_text_count, added, suggestions_count,
                        metacritic, reviews_count, saturated_color,
                        dominant_color, updated, user_game, clip,
                        esrb_rating_id
                    ) VALUES (
                        %(id)s, %(slug)s, %(name)s, %(released)s, %(playtime)s, %(tba)s,
                        %(background_image)s, %(rating)s, %(rating_top)s, %(ratings_count)s,
                        %(reviews_text_count)s, %(added)s, %(suggestions_count)s,
                        %(metacritic)s, %(reviews_count)s, %(saturated_color)s,
                        %(dominant_color)s, %(updated)s, %(user_game)s, %(clip)s,
                        %(esrb_id)s
                    )
                    ON CONFLICT (id_game) DO UPDATE
                      SET slug = EXCLUDED.slug,
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
                """, {
                    "id": game_id,
                    "slug": game.get("slug"),
                    "name": game.get("name"),
                    "released": game.get("released"),
                    "playtime": game.get("playtime"),
                    "tba": game.get("tba"),
                    "background_image": game.get("background_image"),
                    "rating": game.get("rating"),
                    "rating_top": game.get("rating_top"),
                    "ratings_count": game.get("ratings_count"),
                    "reviews_text_count": game.get("reviews_text_count"),
                    "added": game.get("added"),
                    "suggestions_count": game.get("suggestions_count"),
                    "metacritic": game.get("metacritic"),
                    "reviews_count": game.get("reviews_count"),
                    "saturated_color": game.get("saturated_color"),
                    "dominant_color": game.get("dominant_color"),
                    "updated": game.get("updated"),
                    "user_game": game.get("user_game"),
                    "clip": game.get("clip"),
                    "esrb_id": (game.get("esrb_rating") or {}).get("id")
                })

                # Procesar entidades relacionadas
                procesar_esrb(cur, game.get("esrb_rating"), cache)
                procesar_platforms(cur, game_id, game.get("platforms"), cache)
                procesar_parent_platforms(cur, game_id, game.get("parent_platforms"), cache)
                procesar_tags(cur, game_id, game.get("tags"), cache)
                procesar_genres(cur, game_id, game.get("genres"), cache)
                procesar_stores(cur, game_id, game.get("stores"), cache)
                procesar_screenshots(cur, game_id, game.get("short_screenshots"), cache)
                procesar_ratings(cur, game_id, game.get("ratings"), cache)
                procesar_game_status(cur, game_id, game.get("added_by_status"))

            # Confirmar tras cada archivo para no mantener transacción muy larga
            conn.commit()
            logger.info("Archivo %s procesado y confirmado.", os.path.basename(filepath))

    conn.close()
    logger.info("Carga masiva completada exitosamente.")

if __name__ == "__main__":
    main()
