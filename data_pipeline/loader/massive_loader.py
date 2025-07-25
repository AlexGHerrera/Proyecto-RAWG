#!/usr/bin/env python3

import os
import glob
import json
import logging
import psycopg2
from dotenv import load_dotenv
from tqdm import tqdm
import time

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

# Directorio donde están los JSON
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))

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

        json_file = os.path.join(DATA_DIR, "games_all.json")
        logger.info("Cargando archivo único: %s", json_file)

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            logger.error("Error leyendo JSON principal: %s", e)
            return

        all_games = payload.get("results", [])
        logger.info("Total de juegos a procesar: %d", len(all_games))

        with conn.cursor() as cur:
            for i in tqdm(range(start_index, len(all_games), 5000), desc="Procesando por lotes"):
                batch = all_games[i:i+5000]
                for game in batch:
                    game_id = game.get("id")
                    try:
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
                    except Exception as e:
                        logger.exception(f"Error procesando juego {game_id}: {e}")
                conn.commit()
                with open("resume_state.txt", "w") as f:
                    f.write(str(i + 5000))
                logger.info("Lote %d-%d confirmado", i, i+len(batch))

        logger.info("Carga masiva completada exitosamente.")

    elapsed = time.time() - start_time
    logger.info("Tiempo total de ejecución: %.2f segundos", elapsed)

if __name__ == "__main__":
    main()
