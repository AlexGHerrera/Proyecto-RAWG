import json
import boto3
import psycopg2
import logging
import os
from urllib.parse import unquote_plus

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

# ================= FUNCIONES POR ENTIDAD =================

# Insertar o actualizar ESRB Rating
def procesar_esrb(cur, esrb, cache):
    if esrb and esrb["id"] not in cache["esrb"]:
        cur.execute("""
            INSERT INTO esrb_ratings (id_esrb_rating, name, slug)
            VALUES (%s, %s, %s)
            ON CONFLICT (id_esrb_rating) DO UPDATE
            SET name = EXCLUDED.name, slug = EXCLUDED.slug;
        """, (esrb["id"], esrb["name"], esrb["slug"]))
        cache["esrb"].add(esrb["id"])

# Plataformas y tabla puente
def procesar_platforms(cur, game_id, platforms, cache):
    for p in platforms or []:
        plat = p["platform"]
        if plat["id"] not in cache["platforms"]:
            cur.execute("""
                INSERT INTO platforms (id_platform, name, slug)
                VALUES (%s, %s, %s)
                ON CONFLICT (id_platform) DO UPDATE
                SET name = EXCLUDED.name, slug = EXCLUDED.slug;
            """, (plat["id"], plat["name"], plat["slug"]))
            cache["platforms"].add(plat["id"])
        cur.execute("""
            INSERT INTO game_platforms (id_game, id_platform, released_at, requirements_en, requirements_ru)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id_game, id_platform) DO UPDATE
            SET released_at = EXCLUDED.released_at,
                requirements_en = EXCLUDED.requirements_en,
                requirements_ru = EXCLUDED.requirements_ru;
        """, (
            game_id, plat["id"], p.get("released_at"),
            (p.get("requirements_en") or {}).get("minimum"),
            (p.get("requirements_ru") or {}).get("minimum")
        ))

# Plataformas padre
def procesar_parent_platforms(cur, game_id, parent_platforms, cache):
    for pp in parent_platforms or []:
        ppd = pp["platform"]
        if ppd["id"] not in cache["parent_platforms"]:
            cur.execute("""
                INSERT INTO parent_platforms (id_parent_platform, name, slug)
                VALUES (%s, %s, %s)
                ON CONFLICT (id_parent_platform) DO UPDATE
                SET name = EXCLUDED.name, slug = EXCLUDED.slug;
            """, (ppd["id"], ppd["name"], ppd["slug"]))
            cache["parent_platforms"].add(ppd["id"])
        cur.execute("""
            INSERT INTO game_parent_platforms (id_game, id_parent_platform)
            VALUES (%s, %s) ON CONFLICT DO NOTHING;
        """, (game_id, ppd["id"]))

# Tags y tabla puente
def procesar_tags(cur, game_id, tags, cache):
    for tag in tags or []:
        if tag["id"] not in cache["tags"]:
            cur.execute("""
                INSERT INTO tags (id_tag, name, slug, language_tag, games_count, image_background)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id_tag) DO UPDATE
                SET name = EXCLUDED.name, slug = EXCLUDED.slug,
                    language_tag = EXCLUDED.language_tag,
                    games_count = EXCLUDED.games_count,
                    image_background = EXCLUDED.image_background;
            """, (
                tag["id"], tag["name"], tag["slug"], tag["language"],
                tag["games_count"], tag["image_background"]
            ))
            cache["tags"].add(tag["id"])
        cur.execute("""
            INSERT INTO game_tags (id_game, id_tag)
            VALUES (%s, %s) ON CONFLICT DO NOTHING;
        """, (game_id, tag["id"]))

# Géneros
def procesar_genres(cur, game_id, genres, cache):
    for genre in genres or []:
        if genre["id"] not in cache["genres"]:
            cur.execute("""
                INSERT INTO genres (id_genre, name, slug)
                VALUES (%s, %s, %s)
                ON CONFLICT (id_genre) DO UPDATE
                SET name = EXCLUDED.name, slug = EXCLUDED.slug;
            """, (genre["id"], genre["name"], genre["slug"]))
            cache["genres"].add(genre["id"])
        cur.execute("""
            INSERT INTO game_genres (id_game, id_genre)
            VALUES (%s, %s) ON CONFLICT DO NOTHING;
        """, (game_id, genre["id"]))

def procesar_stores(cur, game_id, stores, cache):
    for store in stores or []:
        store_info = store["store"]
        if store_info["id"] not in cache["stores"]:
            cur.execute("""
                INSERT INTO stores (id_store, name, slug)
                VALUES (%s, %s, %s)
                ON CONFLICT (id_store) DO UPDATE
                SET name = EXCLUDED.name, slug = EXCLUDED.slug;
            """, (store_info["id"], store_info["name"], store_info["slug"]))
            cache["stores"].add(store_info["id"])
        cur.execute("""
            INSERT INTO game_stores (id_game, id_store)
            VALUES (%s, %s) ON CONFLICT DO NOTHING;
        """, (game_id, store_info["id"]))

def procesar_screenshots(cur, game_id, screenshots, cache):
    for screenshot in screenshots or []:
        screenshot_id = screenshot.get("id")
        if screenshot_id and screenshot_id not in cache["screenshots"]:
            cur.execute("""
                INSERT INTO screenshots (id_screenshot, image)
                VALUES (%s, %s)
                ON CONFLICT (id_screenshot) DO UPDATE
                SET image = EXCLUDED.image;
            """, (screenshot_id, screenshot.get("image")))
            cache["screenshots"].add(screenshot_id)
        if screenshot_id:
            cur.execute("""
                INSERT INTO game_screenshots (id_game, id_screenshot)
                VALUES (%s, %s) ON CONFLICT DO NOTHING;
            """, (game_id, screenshot_id))

def procesar_ratings(cur, game_id, ratings, cache):
    for rating in ratings or []:
        rating_id = rating.get("id")
        if rating_id and rating_id not in cache["ratings"]:
            cur.execute("""
                INSERT INTO ratings (id_rating, title, count, percent)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id_rating) DO UPDATE
                SET title = EXCLUDED.title, count = EXCLUDED.count, percent = EXCLUDED.percent;
            """, (rating_id, rating.get("title"), rating.get("count"), rating.get("percent")))
            cache["ratings"].add(rating_id)
        if rating_id:
            cur.execute("""
                INSERT INTO game_ratings (id_game, id_rating)
                VALUES (%s, %s) ON CONFLICT DO NOTHING;
            """, (game_id, rating_id))

# 📊 Estado de usuarios (ej: playing, toplay)
def procesar_game_status(cur, game_id, status_dict):
    for status, count in (status_dict or {}).items():
        cur.execute("""
            INSERT INTO game_added_by_status (id_game, status, count)
            VALUES (%s, %s, %s)
            ON CONFLICT (id_game, status) DO UPDATE
            SET count = EXCLUDED.count;
        """, (game_id, status, count))

# =================== LAMBDA HANDLER ===================

def lambda_handler(event, context):
    logger.info("Inicio de ejecución Lambda.")

    # Obtener información del archivo subido a S3
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = unquote_plus(event['Records'][0]['s3']['object']['key'])
    logger.info(f"Archivo detectado: s3://{bucket}/{key}")

    # Descargar y parsear el JSON
    response = s3_client.get_object(Bucket=bucket, Key=key)
    data = json.loads(response['Body'].read())
    games = data["results"]
    logger.info(f"{len(games)} juegos leídos.")

    # Conectar a la base de datos RDS PostgreSQL
    conn = psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT
    )

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