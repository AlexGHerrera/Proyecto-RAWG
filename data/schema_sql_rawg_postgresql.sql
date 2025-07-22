
-- Tabla principal de juegos
CREATE TABLE games (
    id_game BIGINT PRIMARY KEY,
    slug VARCHAR(255),
    name VARCHAR(255),
    released DATE,
    tba BOOLEAN,
    background_image TEXT,
    rating FLOAT,
    rating_top INT,
    ratings_count INT,
    reviews_text_count INT,
    added INT,
    metacritic INT,
    suggestions_count INT,
    updated TIMESTAMPTZ,
    user_game VARCHAR(255),
    reviews_count INT,
    saturated_color CHAR(6),
    dominant_color CHAR(6),
    clip TEXT,
    esrb_rating_id INT,
    FOREIGN KEY (esrb_rating_id) REFERENCES esrb_ratings(id_esrb_rating)
);

-- Tabla ESRB
CREATE TABLE esrb_ratings (
    id_esrb_rating INT PRIMARY KEY,
    name VARCHAR(100),
    slug VARCHAR(100)
);

-- Plataformas
CREATE TABLE platforms (
    id_platform BIGINT PRIMARY KEY,
    name VARCHAR(255),
    slug VARCHAR(255)
);

-- Tabla puente juego-plataforma
CREATE TABLE game_platforms (
    id_game BIGINT REFERENCES games(id_game),
    id_platform BIGINT REFERENCES platforms(id_platform),
    released_at DATE,
    requirements_en TEXT,
    requirements_ru TEXT,
    PRIMARY KEY (id_game, id_platform)
);

-- Plataformas padre
CREATE TABLE parent_platforms (
    id_parent_platform BIGINT PRIMARY KEY,
    name VARCHAR(255),
    slug VARCHAR(255)
);

CREATE TABLE game_parent_platforms (
    id_game BIGINT REFERENCES games(id_game),
    id_parent_platform BIGINT REFERENCES parent_platforms(id_parent_platform),
    PRIMARY KEY (id_game, id_parent_platform)
);

-- Tiendas
CREATE TABLE stores (
    id_store BIGINT PRIMARY KEY,
    name VARCHAR(255),
    slug VARCHAR(255),
    domain VARCHAR(255)
);

CREATE TABLE game_stores (
    id_game BIGINT REFERENCES games(id_game),
    id_store BIGINT REFERENCES stores(id_store),
    store_instance_id BIGINT,
    PRIMARY KEY (id_game, id_store)
);

-- Ratings
CREATE TABLE ratings (
    id_rating BIGINT PRIMARY KEY,
    title VARCHAR(255)
);

CREATE TABLE game_ratings (
    id_game BIGINT REFERENCES games(id_game),
    id_rating BIGINT REFERENCES ratings(id_rating),
    count INT,
    percent FLOAT,
    PRIMARY KEY (id_game, id_rating)
);

-- Tags
CREATE TABLE tags (
    id_tag BIGINT PRIMARY KEY,
    name VARCHAR(255),
    slug VARCHAR(255),
    language_tag CHAR(3),
    games_count INT,
    image_background TEXT
);

CREATE TABLE game_tags (
    id_game BIGINT REFERENCES games(id_game),
    id_tag BIGINT REFERENCES tags(id_tag),
    PRIMARY KEY (id_game, id_tag)
);

-- Screenshots
CREATE TABLE short_screenshots (
    id_short_screenshot BIGINT PRIMARY KEY,
    image TEXT
);

CREATE TABLE game_short_screenshots (
    id_game BIGINT REFERENCES games(id_game),
    id_short_screenshot BIGINT REFERENCES short_screenshots(id_short_screenshot),
    PRIMARY KEY (id_game, id_short_screenshot)
);

-- Géneros
CREATE TABLE genres (
    id_genre BIGINT PRIMARY KEY,
    name VARCHAR(255),
    slug VARCHAR(255)
);

CREATE TABLE game_genres (
    id_game BIGINT REFERENCES games(id_game),
    id_genre BIGINT REFERENCES genres(id_genre),
    PRIMARY KEY (id_game, id_genre)
);

-- Estados de adición del juego por usuarios
CREATE TABLE game_added_by_status (
    id_game BIGINT REFERENCES games(id_game),
    status VARCHAR(50),
    count FLOAT,
    PRIMARY KEY (id_game, status)
);
