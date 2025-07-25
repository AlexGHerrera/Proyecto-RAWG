# data_pipeline/loader/processing.py

import logging

# Logger compartido por todas las funciones de procesamiento
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def procesar_esrb(cur, esrb, cache):
    """Insert or update ESRB rating metadata."""
    if esrb and esrb["id"] not in cache["esrb"]:
        cur.execute(
            """
            INSERT INTO esrb_ratings (id_esrb_rating, name, slug)
            VALUES (%s, %s, %s)
            ON CONFLICT (id_esrb_rating) DO UPDATE
                SET name = EXCLUDED.name,
                    slug = EXCLUDED.slug;
            """,
            (esrb["id"], esrb["name"], esrb["slug"])
        )
        cache["esrb"].add(esrb["id"])


def procesar_platforms(cur, game_id, platforms, cache):
    """Insert or update platform metadata and game-platform relationships."""
    for p in platforms or []:
        plat = p["platform"]
        pid = plat["id"]

        # Inserta o actualiza la plataforma
        if pid not in cache["platforms"]:
            cur.execute(
                """
                INSERT INTO platforms (
                    id_platform, name, slug,
                    games_count, image_background,
                    year_start, year_end
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id_platform) DO UPDATE
                    SET name = EXCLUDED.name,
                        slug = EXCLUDED.slug,
                        games_count = EXCLUDED.games_count,
                        image_background = EXCLUDED.image_background,
                        year_start = EXCLUDED.year_start,
                        year_end = EXCLUDED.year_end;
                """,
                (
                    pid, plat["name"], plat["slug"],
                    p.get("games_count"), plat.get("image_background"),
                    plat.get("year_start"), plat.get("year_end")
                )
            )
            cache["platforms"].add(pid)

        # Inserta o actualiza la relación juego–plataforma
        cur.execute(
            """
            INSERT INTO game_platforms (
                id_game, id_platform, released_at,
                requirements_en_minimum, requirements_en_recommended,
                requirements_ru_minimum, requirements_ru_recommended
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id_game, id_platform) DO UPDATE
                SET released_at = EXCLUDED.released_at,
                    requirements_en_minimum = EXCLUDED.requirements_en_minimum,
                    requirements_en_recommended = EXCLUDED.requirements_en_recommended,
                    requirements_ru_minimum = EXCLUDED.requirements_ru_minimum,
                    requirements_ru_recommended = EXCLUDED.requirements_ru_recommended;
            """,
            (
                game_id, pid, p.get("released_at"),
                (p.get("requirements_en") or {}).get("minimum"),
                (p.get("requirements_en") or {}).get("recommended"),
                (p.get("requirements_ru") or {}).get("minimum"),
                (p.get("requirements_ru") or {}).get("recommended")
            )
        )


def procesar_parent_platforms(cur, game_id, parent_platforms, cache):
    """Insert or update parent platform metadata and relationships."""
    for pp in parent_platforms or []:
        ppd = pp["platform"]
        ppid = ppd["id"]

        # Inserta o actualiza la plataforma padre
        if ppid not in cache["parent_platforms"]:
            cur.execute(
                """
                INSERT INTO parent_platforms (id_parent_platform, name, slug)
                VALUES (%s, %s, %s)
                ON CONFLICT (id_parent_platform) DO UPDATE
                    SET name = EXCLUDED.name,
                        slug = EXCLUDED.slug;
                """,
                (ppid, ppd["name"], ppd["slug"])
            )
            cache["parent_platforms"].add(ppid)

        # Inserta la relación juego–plataforma padre
        cur.execute(
            """
            INSERT INTO game_parent_platforms (id_game, id_parent_platform)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING;
            """,
            (game_id, ppid)
        )


def procesar_tags(cur, game_id, tags, cache):
    """Insert or update tag metadata and game-tag relationships."""
    for tag in tags or []:
        tid = tag["id"]

        # Inserta o actualiza la etiqueta
        if tid not in cache["tags"]:
            cur.execute(
                """
                INSERT INTO tags (
                    id_tag, name, slug, language_tag,
                    games_count, image_background
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id_tag) DO UPDATE
                    SET name = EXCLUDED.name,
                        slug = EXCLUDED.slug,
                        language_tag = EXCLUDED.language_tag,
                        games_count = EXCLUDED.games_count,
                        image_background = EXCLUDED.image_background;
                """,
                (
                    tid, tag["name"], tag["slug"], tag.get("language"),
                    tag.get("games_count"), tag.get("image_background")
                )
            )
            cache["tags"].add(tid)

        # Inserta la relación juego–etiqueta
        cur.execute(
            "INSERT INTO game_tags (id_game, id_tag) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
            (game_id, tid)
        )


def procesar_genres(cur, game_id, genres, cache):
    """Insert or update genre metadata and game-genre relationships."""
    for genre in genres or []:
        gid = genre["id"]

        # Inserta o actualiza el género
        if gid not in cache["genres"]:
            cur.execute(
                """
                INSERT INTO genres (id_genre, name, slug)
                VALUES (%s, %s, %s)
                ON CONFLICT (id_genre) DO UPDATE
                    SET name = EXCLUDED.name,
                        slug = EXCLUDED.slug;
                """,
                (gid, genre["name"], genre["slug"])
            )
            cache["genres"].add(gid)

        # Inserta la relación juego–género
        cur.execute(
            "INSERT INTO game_genres (id_game, id_genre) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
            (game_id, gid)
        )


def procesar_stores(cur, game_id, stores, cache):
    """Insert or update store metadata and game-store relationships."""
    for store in stores or []:
        sid = store["store"]["id"]

        # Inserta o actualiza la tienda
        if sid not in cache["stores"]:
            cur.execute(
                """
                INSERT INTO stores (
                    id_store, name, slug, domain,
                    games_count, image_background
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id_store) DO UPDATE
                    SET name = EXCLUDED.name,
                        slug = EXCLUDED.slug,
                        domain = EXCLUDED.domain,
                        games_count = EXCLUDED.games_count,
                        image_background = EXCLUDED.image_background;
                """,
                (
                    sid, store["store"]["name"], store["store"]["slug"],
                    store["store"].get("domain"), store.get("games_count"),
                    store.get("image_background")
                )
            )
            cache["stores"].add(sid)

        # Inserta la relación juego–tienda
        cur.execute(
            "INSERT INTO game_stores (id_game, id_store) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
            (game_id, sid)
        )


def procesar_screenshots(cur, game_id, screenshots, cache):
    """Insert or update short_screenshots and relationships."""
    for shot in screenshots or []:
        shot_id = shot.get("id")
        if not shot_id or shot_id <= 0:
            continue

        # Inserta o actualiza la captura breve
        if shot_id not in cache["screenshots"]:
            cur.execute(
                """
                INSERT INTO short_screenshots (id_short_screenshot, image)
                VALUES (%s, %s)
                ON CONFLICT (id_short_screenshot) DO UPDATE
                    SET image = EXCLUDED.image;
                """,
                (shot_id, shot.get("image"))
            )
            cache["screenshots"].add(shot_id)

        # Inserta la relación juego–captura
        cur.execute(
            "INSERT INTO game_short_screenshots (id_game, id_short_screenshot) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
            (game_id, shot_id)
        )


def procesar_ratings(cur, game_id, ratings, cache):
    """Insert or update detailed ratings and relationships."""
    for rating in ratings or []:
        rid = rating.get("id")

        # Inserta o actualiza el rating detallado
        if rid and rid not in cache["ratings"]:
            cur.execute(
                """
                INSERT INTO ratings (id_rating, title, count, percent)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id_rating) DO UPDATE
                    SET title = EXCLUDED.title,
                        count = EXCLUDED.count,
                        percent = EXCLUDED.percent;
                """,
                (rid, rating.get("title"), rating.get("count"), rating.get("percent"))
            )
            cache["ratings"].add(rid)

        # Inserta la relación juego–rating
        if rid:
            cur.execute(
                "INSERT INTO game_ratings (id_game, id_rating) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
                (game_id, rid)
            )


def procesar_game_status(cur, game_id, status_dict):
    """Insert or update user game status counts (added_by_status)."""
    for status, count in (status_dict or {}).items():
        cur.execute(
            """
            INSERT INTO game_added_by_status (id_game, status, count)
            VALUES (%s, %s, %s)
            ON CONFLICT (id_game, status) DO UPDATE
                SET count = EXCLUDED.count;
            """,
            (game_id, status, count)
        )