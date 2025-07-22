
import json
import psycopg2

# Configuración de la conexión
DB_HOST = 'localhost'
DB_NAME = 'rauwgprueba'
DB_USER = 'postgres'
DB_PASS = 'SupernenA'
DB_PORT = '5432'

# Cargar JSON
with open("/Users/alexg.herrera/Desktop/HackABoss/Proyecto-RAWG/data/cajon/games_page_1.json", "r") as file:
    data = json.load(file)

games = data["results"]

# Insertar datos en PostgreSQL usando context managers
with psycopg2.connect(
    host=DB_HOST,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS,
    port=DB_PORT
) as conn:
    with conn.cursor() as cur:
        inserted = {
            "esrb": set(), "platforms": set(), "stores": set(),
            "ratings": set(), "tags": set(), "screenshots": set(),
            "genres": set(), "parent_platforms": set()
        }

        for game in games:
            game_id = game["id"]
            esrb = game.get("esrb_rating")
            if esrb and esrb["id"] not in inserted["esrb"]:
                cur.execute("""
                    INSERT INTO esrb_ratings (id_esrb_rating, name, slug)
                    VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;
                """, (esrb["id"], esrb["name"], esrb["slug"]))
                inserted["esrb"].add(esrb["id"])

            cur.execute("""
                INSERT INTO games (
                    id_game, slug, name, released, tba, background_image, rating,
                    rating_top, ratings_count, reviews_text_count, added, metacritic,
                    suggestions_count, updated, user_game, reviews_count, saturated_color,
                    dominant_color, clip, esrb_rating_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING;
            """, (
                game_id, game.get("slug"), game.get("name"), game.get("released"),
                game.get("tba"), game.get("background_image"), game.get("rating"),
                game.get("rating_top"), game.get("ratings_count"), game.get("reviews_text_count"),
                game.get("added"), game.get("metacritic"), game.get("suggestions_count"),
                game.get("updated"), game.get("user_game"), game.get("reviews_count"),
                game.get("saturated_color"), game.get("dominant_color"), game.get("clip"),
                esrb["id"] if esrb else None
            ))

            for p in game.get("platforms", []):
                plat = p["platform"]
                if plat["id"] not in inserted["platforms"]:
                    cur.execute("""
                        INSERT INTO platforms (id_platform, name, slug)
                        VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;
                    """, (plat["id"], plat["name"], plat["slug"]))
                    inserted["platforms"].add(plat["id"])
                cur.execute("""
                    INSERT INTO game_platforms (id_game, id_platform, released_at, requirements_en, requirements_ru)
                    VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING;
                """, (
                    game_id, plat["id"], p.get("released_at"),
                    (p.get("requirements_en") or {}).get("minimum"),
                    (p.get("requirements_ru") or {}).get("minimum")
                ))

            for pp in game.get("parent_platforms", []):
                ppd = pp["platform"]
                if ppd["id"] not in inserted["parent_platforms"]:
                    cur.execute("""
                        INSERT INTO parent_platforms (id_parent_platform, name, slug)
                        VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;
                    """, (ppd["id"], ppd["name"], ppd["slug"]))
                    inserted["parent_platforms"].add(ppd["id"])
                cur.execute("""
                    INSERT INTO game_parent_platforms (id_game, id_parent_platform)
                    VALUES (%s, %s) ON CONFLICT DO NOTHING;
                """, (game_id, ppd["id"]))

            for s in game.get("stores", []):
                store = s["store"]
                if store["id"] not in inserted["stores"]:
                    cur.execute("""
                        INSERT INTO stores (id_store, name, slug, domain)
                        VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING;
                    """, (store["id"], store["name"], store["slug"], store.get("domain")))
                    inserted["stores"].add(store["id"])
                cur.execute("""
                    INSERT INTO game_stores (id_game, id_store, store_instance_id)
                    VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;
                """, (game_id, store["id"], s["id"]))

            for r in game.get("ratings", []):
                if r["id"] not in inserted["ratings"]:
                    cur.execute("""
                        INSERT INTO ratings (id_rating, title)
                        VALUES (%s, %s) ON CONFLICT DO NOTHING;
                    """, (r["id"], r["title"]))
                    inserted["ratings"].add(r["id"])
                cur.execute("""
                    INSERT INTO game_ratings (id_game, id_rating, count, percent)
                    VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING;
                """, (game_id, r["id"], r["count"], r["percent"]))

            for tag in game.get("tags", []):
                if tag["id"] not in inserted["tags"]:
                    cur.execute("""
                        INSERT INTO tags (id_tag, name, slug, language_tag, games_count, image_background)
                        VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING;
                    """, (
                        tag["id"], tag["name"], tag["slug"], tag["language"],
                        tag["games_count"], tag["image_background"]
                    ))
                    inserted["tags"].add(tag["id"])
                cur.execute("""
                    INSERT INTO game_tags (id_game, id_tag)
                    VALUES (%s, %s) ON CONFLICT DO NOTHING;
                """, (game_id, tag["id"]))

            for ss in game.get("short_screenshots", []):
                if ss["id"] not in inserted["screenshots"]:
                    cur.execute("""
                        INSERT INTO short_screenshots (id_short_screenshot, image)
                        VALUES (%s, %s) ON CONFLICT DO NOTHING;
                    """, (ss["id"], ss["image"]))
                    inserted["screenshots"].add(ss["id"])
                cur.execute("""
                    INSERT INTO game_short_screenshots (id_game, id_short_screenshot)
                    VALUES (%s, %s) ON CONFLICT DO NOTHING;
                """, (game_id, ss["id"]))

            for genre in game.get("genres", []):
                if genre["id"] not in inserted["genres"]:
                    cur.execute("""
                        INSERT INTO genres (id_genre, name, slug)
                        VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;
                    """, (genre["id"], genre["name"], genre["slug"]))
                    inserted["genres"].add(genre["id"])
                cur.execute("""
                    INSERT INTO game_genres (id_game, id_genre)
                    VALUES (%s, %s) ON CONFLICT DO NOTHING;
                """, (game_id, genre["id"]))

            for status, count in game.get("added_by_status", {}).items():
                cur.execute("""
                    INSERT INTO game_added_by_status (id_game, status, count)
                    VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;
                """, (game_id, status, count))

print("Carga completa.")
