-- ======================================================
--      TABLA PRINCIPAL: Juegos
-- ======================================================
CREATE TABLE games (
    id_game                 BIGINT       PRIMARY KEY,         -- Identificador único de RAWG
    slug                    TEXT         NOT NULL,            -- Versión amigable del nombre
    name                    TEXT         NOT NULL,            -- Nombre del juego
    released                DATE,                           -- Fecha de lanzamiento
    tba                     BOOLEAN      DEFAULT FALSE,       -- “To be announced”
    background_image        TEXT,                           -- URL de la imagen de fondo
    playtime                INT,                            -- Tiempo promedio de juego (horas)
    rating                  REAL,                           -- Puntuación promedio
    rating_top              INT,                            -- Puntuación máxima posible
    ratings_count           INT,                            -- Número de valoraciones
    reviews_text_count      INT,                            -- Número de reseñas con texto
    added                   INT,                            -- Veces añadido a listas de usuarios
    suggestions_count       INT,                            -- Sugerencias generadas para el juego
    metacritic              INT,                            -- Puntuación en Metacritic
    reviews_count           INT,                            -- Conteo total de reseñas
    updated                 TIMESTAMPTZ,                    -- Fecha/hora de última actualización en RAWG
    user_game               JSONB,                          -- Objeto con estado del usuario (ej. { "played": 10 })
    saturated_color         VARCHAR(7),                     -- Color saturado (hex)
    dominant_color          VARCHAR(7),                     -- Color dominante (hex)
    clip                    JSONB,                          -- Objeto con datos de clip (URL, duración, etc.)
    esrb_rating_id          BIGINT       REFERENCES esrb_ratings(id_esrb_rating)
);

-- ======================================================
--      ESPECIAL: Paginación de la API RAWG
-- ======================================================
CREATE TABLE api_pages (
    fetch_id    SERIAL      PRIMARY KEY,                    -- Identificador interno
    count       INT         NOT NULL,                       -- Total de juegos en RAWG
    next        TEXT,                                    -- URL de la siguiente página
    previous    TEXT,                                    -- URL de la página anterior
    fetched_at  TIMESTAMPTZ DEFAULT NOW()                -- Momento de la descarga
);

-- ======================================================
--      CATALOGO: ESRB Ratings
-- ======================================================
CREATE TABLE esrb_ratings (
    id_esrb_rating  BIGINT     PRIMARY KEY,                 -- ID del rating
    name            TEXT       NOT NULL,                   -- Nombre (E, T, M…)
    slug            TEXT       NOT NULL                    -- Versión amigable
);

-- ======================================================
--      ENTIDAD: Géneros
-- ======================================================
CREATE TABLE genres (
    id_genre       BIGINT     PRIMARY KEY,                  -- ID del género
    name           TEXT       NOT NULL,
    slug           TEXT       NOT NULL
);

CREATE TABLE game_genres (
    id_game        BIGINT     REFERENCES games(id_game),
    id_genre       BIGINT     REFERENCES genres(id_genre),
    PRIMARY KEY (id_game, id_genre)
);

-- ======================================================
--      ENTIDAD: Plataformas
-- ======================================================
CREATE TABLE platforms (
    id_platform       BIGINT   PRIMARY KEY,                 -- ID de la plataforma
    name              TEXT     NOT NULL,
    slug              TEXT     NOT NULL,
    games_count       INT,                                 -- Número de juegos disponibles
    image_background  TEXT,                                -- Imagen asociada
    year_start        INT,                                 -- Año de lanzamiento inicial
    year_end          INT                                  -- Año de final de soporte
);

CREATE TABLE game_platforms (
    id_game                   BIGINT   REFERENCES games(id_game),
    id_platform               BIGINT   REFERENCES platforms(id_platform),
    released_at               DATE,                          -- Fecha de lanzamiento en esa plataforma
    requirements_en_minimum   TEXT,                          -- Requisitos mínimos (en)
    requirements_en_recommended TEXT,                        -- Requisitos recomendados (en)
    requirements_ru_minimum   TEXT,                          -- Requisitos mínimos (ru)
    requirements_ru_recommended TEXT,                        -- Requisitos recomendados (ru)
    PRIMARY KEY (id_game, id_platform)
);

-- ======================================================
--      ENTIDAD: Parent Platforms
-- ======================================================
CREATE TABLE parent_platforms (
    id_parent_platform  BIGINT    PRIMARY KEY,
    name                TEXT      NOT NULL,
    slug                TEXT      NOT NULL
);

CREATE TABLE game_parent_platforms (
    id_game             BIGINT    REFERENCES games(id_game),
    id_parent_platform  BIGINT    REFERENCES parent_platforms(id_parent_platform),
    PRIMARY KEY (id_game, id_parent_platform)
);

-- ======================================================
--      ENTIDAD: Tags
-- ======================================================
CREATE TABLE tags (
    id_tag             BIGINT     PRIMARY KEY,
    name               TEXT       NOT NULL,
    slug               TEXT       NOT NULL,
    language_tag       TEXT,                             -- Idioma del tag
    games_count        INT,                              -- Juegos asociados
    image_background   TEXT
);

CREATE TABLE game_tags (
    id_game            BIGINT     REFERENCES games(id_game),
    id_tag             BIGINT     REFERENCES tags(id_tag),
    PRIMARY KEY (id_game, id_tag)
);

-- ======================================================
--      ENTIDAD: Stores
-- ======================================================
CREATE TABLE stores (
    id_store           BIGINT     PRIMARY KEY,
    name               TEXT       NOT NULL,
    slug               TEXT       NOT NULL,
    domain             TEXT,                             -- Dominio de la tienda
    games_count        INT,
    image_background   TEXT
);

CREATE TABLE game_stores (
    id_game            BIGINT     REFERENCES games(id_game),
    id_store           BIGINT     REFERENCES stores(id_store),
    PRIMARY KEY (id_game, id_store)
);

-- ======================================================
--      ENTIDAD: Capturas de pantalla
-- ======================================================
CREATE TABLE short_screenshots (
    id_short_screenshot  BIGINT   PRIMARY KEY,              -- ID de la captura
    image                TEXT                              -- URL de la imagen
);

CREATE TABLE game_short_screenshots (
    id_game              BIGINT   REFERENCES games(id_game),
    id_short_screenshot  BIGINT   REFERENCES short_screenshots(id_short_screenshot),
    PRIMARY KEY (id_game, id_short_screenshot)
);

-- ======================================================
--      ENTIDAD: Ratings detallados
-- ======================================================
CREATE TABLE ratings (
    id_rating   BIGINT     PRIMARY KEY,                    -- ID interno del tipo de rating
    title       TEXT,                                    -- Texto descriptivo (“exceptional”, “meh”…)
    count       INT,                                     -- Veces aplicado ese rating
    percent     REAL                                     -- Porcentaje de usuarios
);

CREATE TABLE game_ratings (
    id_game     BIGINT     REFERENCES games(id_game),
    id_rating   BIGINT     REFERENCES ratings(id_rating),
    PRIMARY KEY (id_game, id_rating)
);

-- ======================================================
--      ENTIDAD: Estado de usuario (added_by_status)
-- ======================================================
CREATE TABLE game_added_by_status (
    id_game     BIGINT     REFERENCES games(id_game),
    status      TEXT,                                    -- Ej: “playing”, “toplay”, “completed”
    count       INT,                                     -- Veces en ese estado
    PRIMARY KEY (id_game, status)
);