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


def procesar_platforms_batch(cur, platforms_data, cache):
    """Insert or update platform metadata and game-platform relationships from batch."""
    for game_id, platforms in platforms_data:
        for p in platforms or []:
            plat = p["platform"]
            pid = plat["id"]

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


def procesar_parent_platforms_batch(cur, parent_platforms_data, cache):
    """Insert or update parent platform metadata and relationships from batch."""
    for game_id, parent_platforms in parent_platforms_data:
        for pp in parent_platforms or []:
            ppd = pp["platform"]
            ppid = ppd["id"]

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

            cur.execute(
                """
                INSERT INTO game_parent_platforms (id_game, id_parent_platform)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING;
                """,
                (game_id, ppid)
            )


def procesar_tags(cur, tags_data, cache):
    """Insert or update tag metadata and game-tag relationships from batch."""
    for game_id, tags in tags_data:
        for tag in tags or []:
            tid = tag["id"]

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

            cur.execute(
                "INSERT INTO game_tags (id_game, id_tag) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
                (game_id, tid)
            )


def procesar_genres(cur, genres_data, cache):
    """Insert or update genre metadata and game-genre relationships from batch."""
    for game_id, genres in genres_data:
        for genre in genres or []:
            gid = genre["id"]

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

            cur.execute(
                "INSERT INTO game_genres (id_game, id_genre) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
                (game_id, gid)
            )


def procesar_stores(cur, stores_data, cache):
    """Insert or update store metadata and game-store relationships from batch."""
    for game_id, stores in stores_data:
        for store in stores or []:
            sid = store["store"]["id"]

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

            cur.execute(
                "INSERT INTO game_stores (id_game, id_store) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
                (game_id, sid)
            )


def procesar_screenshots(cur, screenshots_data, cache):
    """Insert or update short_screenshots and relationships from batch."""
    for game_id, screenshots in screenshots_data:
        for shot in screenshots or []:
            shot_id = shot.get("id")
            if not shot_id or shot_id <= 0:
                continue

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

            cur.execute(
                "INSERT INTO game_short_screenshots (id_game, id_short_screenshot) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
                (game_id, shot_id)
            )


def procesar_ratings(cur, ratings_data, cache):
    """Insert or update detailed ratings and relationships from batch."""
    for game_id, ratings in ratings_data:
        for rating in ratings or []:
            rid = rating.get("id")

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

            if rid:
                cur.execute(
                    "INSERT INTO game_ratings (id_game, id_rating) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
                    (game_id, rid)
                )


def procesar_game_status(cur, status_data):
    """Insert or update user game status counts (added_by_status) from batch."""
    for game_id, status_dict in status_data:
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