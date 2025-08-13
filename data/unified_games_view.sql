-- =====================================================================
-- Vista Materializada Única para API v5 - Nombres Naturales Mejorados
-- Basada en el schema real de PostgreSQL
-- =====================================================================

DROP MATERIALIZED VIEW IF EXISTS games_complete CASCADE;

CREATE MATERIALIZED VIEW games_complete AS
SELECT 
    -- Identificación básica
    g.id_game as id,
    g.name,
    g.slug,
    
    -- Fechas y tiempo
    g.released as release_date,
    EXTRACT(YEAR FROM g.released) as year,
    (EXTRACT(YEAR FROM g.released)/10)*10 as decade,
    g.playtime as hours,
    
    -- Puntuaciones y métricas
    g.rating,
    g.rating_top as max_rating,
    g.ratings_count as total_ratings,
    g.metacritic as metacritic_score,
    g.added as popularity,
    g.suggestions_count as suggestions,
    
    -- Información visual
    g.background_image as image,
    
    -- Géneros (concatenados para búsqueda natural)
    STRING_AGG(DISTINCT gen.name, ', ' ORDER BY gen.name) as genres,
    
    -- Plataformas (concatenadas)
    STRING_AGG(DISTINCT plat.name, ', ' ORDER BY plat.name) as platforms,
    
    -- Tags principales (concatenados) - sin ORDER BY para evitar error
    STRING_AGG(DISTINCT t.name, ', ') as tags,
    
    -- ESRB Rating
    esrb.name as age_rating,
    
    -- Categorías útiles para consultas naturales
    CASE 
        WHEN g.rating IS NULL THEN 'unrated'
        WHEN g.rating < 2.5 THEN 'poor'
        WHEN g.rating < 3.5 THEN 'average' 
        WHEN g.rating < 4.2 THEN 'good'
        ELSE 'excellent'
    END as quality,
    
    CASE
        WHEN EXTRACT(YEAR FROM g.released) >= 2020 THEN 'recent'
        WHEN EXTRACT(YEAR FROM g.released) >= 2010 THEN 'modern'
        WHEN EXTRACT(YEAR FROM g.released) >= 2000 THEN 'classic'
        ELSE 'retro'
    END as era,
    
    -- Categoría de popularidad
    CASE 
        WHEN g.added > 50000 THEN 'very_popular'
        WHEN g.added > 10000 THEN 'popular'
        WHEN g.added > 1000 THEN 'known'
        ELSE 'niche'
    END as popularity_level

FROM games g
LEFT JOIN game_genres gg ON g.id_game = gg.id_game
LEFT JOIN genres gen ON gg.id_genre = gen.id_genre
LEFT JOIN game_platforms gp ON g.id_game = gp.id_game
LEFT JOIN platforms plat ON gp.id_platform = plat.id_platform
LEFT JOIN game_tags gt ON g.id_game = gt.id_game
LEFT JOIN tags t ON gt.id_tag = t.id_tag
LEFT JOIN esrb_ratings esrb ON g.esrb_rating_id = esrb.id_esrb_rating
GROUP BY g.id_game, g.name, g.slug, g.released, g.playtime, g.rating, 
         g.rating_top, g.ratings_count, g.metacritic, g.added, 
         g.suggestions_count, g.background_image, esrb.name;

-- Índices para optimización
CREATE INDEX games_complete_name_idx ON games_complete (name);
CREATE INDEX games_complete_rating_idx ON games_complete (rating DESC);
CREATE INDEX games_complete_popularity_idx ON games_complete (popularity DESC);
CREATE INDEX games_complete_year_idx ON games_complete (year);
CREATE INDEX games_complete_genres_idx ON games_complete USING gin(to_tsvector('english', genres));
CREATE INDEX games_complete_platforms_idx ON games_complete USING gin(to_tsvector('english', platforms));