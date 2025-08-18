import os, re, time, logging
from typing import Dict, Any, List, Tuple, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__); logging.basicConfig(level=logging.INFO)

# Cache de consultas para optimización de rendimiento
_query_cache = {}
_cache_ttl = 300  # 5 minutos de cache

# SQLCoder-7B-2 via deployed HuggingFace endpoint
SQLCODER_ENDPOINT = os.getenv("SQLCODER_ENDPOINT")
MODEL_NAME = "defog/sqlcoder-7b-2"

# ---------- Schema ----------
def generate_rawg_schema() -> str:
    """
    Schema completo de la base de datos PostgreSQL RAWG según documentación oficial.
    Optimizado para SQLCoder-7B-2 con tablas reales y relaciones.
    """
    return """-- TABLA PRINCIPAL: Juegos
CREATE TABLE games (
    id_game BIGINT PRIMARY KEY,
    slug TEXT NOT NULL,
    name TEXT NOT NULL,
    released DATE,
    tba BOOLEAN DEFAULT FALSE,
    background_image TEXT,
    playtime INT,
    rating REAL,
    rating_top INT,
    ratings_count INT,
    reviews_text_count INT,
    added INT,
    suggestions_count INT,
    metacritic INT,
    reviews_count INT,
    updated TIMESTAMPTZ,
    user_game JSONB,
    saturated_color VARCHAR(7),
    dominant_color VARCHAR(7),
    clip JSONB,
    esrb_rating_id BIGINT
);

-- CATÁLOGOS
CREATE TABLE esrb_ratings (
    id_esrb_rating BIGINT PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT NOT NULL
);

CREATE TABLE genres (
    id_genre BIGINT PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT NOT NULL
);

CREATE TABLE platforms (
    id_platform BIGINT PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    games_count INT,
    image_background TEXT,
    year_start INT,
    year_end INT
);

CREATE TABLE parent_platforms (
    id_parent_platform BIGINT PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT NOT NULL
);

CREATE TABLE tags (
    id_tag BIGINT PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    language_tag TEXT,
    games_count INT,
    image_background TEXT
);

CREATE TABLE stores (
    id_store BIGINT PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    domain TEXT,
    games_count INT,
    image_background TEXT
);

-- RELACIONES MUCHOS A MUCHOS
CREATE TABLE game_genres (
    id_game BIGINT REFERENCES games(id_game),
    id_genre BIGINT REFERENCES genres(id_genre),
    PRIMARY KEY (id_game, id_genre)
);

CREATE TABLE game_platforms (
    id_game BIGINT REFERENCES games(id_game),
    id_platform BIGINT REFERENCES platforms(id_platform),
    released_at DATE,
    requirements_en_minimum TEXT,
    requirements_en_recommended TEXT,
    PRIMARY KEY (id_game, id_platform)
);

CREATE TABLE game_parent_platforms (
    id_game BIGINT REFERENCES games(id_game),
    id_parent_platform BIGINT REFERENCES parent_platforms(id_parent_platform),
    PRIMARY KEY (id_game, id_parent_platform)
);

CREATE TABLE game_tags (
    id_game BIGINT REFERENCES games(id_game),
    id_tag BIGINT REFERENCES tags(id_tag),
    PRIMARY KEY (id_game, id_tag)
);

CREATE TABLE game_stores (
    id_game BIGINT REFERENCES games(id_game),
    id_store BIGINT REFERENCES stores(id_store),
    PRIMARY KEY (id_game, id_store)
);"""

# ---------- SQLCoder API ----------
def generate_sql_with_sqlcoder(user_question: str, schema: str) -> str:
    """Genera SQL usando SQLCoder-7B-2 via endpoint desplegado"""
    if not REQUESTS_AVAILABLE:
        raise ImportError("Install requests")
    
    hf_token = os.getenv("HF_API_TOKEN")
    if not hf_token:
        raise ValueError("HF_API_TOKEN not found in environment")
    
    # Formato oficial de SQLCoder según documentación de Defog
    prompt = f"""### Task
Generate a SQL query to answer [QUESTION]{user_question}[/QUESTION]

### Instructions
- If you cannot answer the question with the available database schema, return 'I do not know'
- Always include relevant filter columns in SELECT clause when using WHERE conditions
- When filtering by specific values (rating, metacritic, year, etc.), include those columns in the results
- For example: if filtering by "rating > 4.0", include both name AND rating columns in SELECT

### Critical Aggregation Rules
- For "count games by genre" queries: SELECT gen.name, COUNT(g.id_game) FROM genres gen JOIN game_genres gg ON gen.id_genre = gg.id_genre JOIN games g ON gg.id_game = g.id_game GROUP BY gen.name ORDER BY COUNT(g.id_game) DESC
- For "count games by platform" queries: SELECT p.name, COUNT(g.id_game) FROM platforms p JOIN game_platforms gp ON p.id_platform = gp.id_platform JOIN games g ON gp.id_game = g.id_game GROUP BY p.name ORDER BY COUNT(g.id_game) DESC
- For "count games by tag" queries: SELECT t.name, COUNT(g.id_game) FROM tags t JOIN game_tags gt ON t.id_tag = gt.id_tag JOIN games g ON gt.id_game = g.id_game GROUP BY t.name ORDER BY COUNT(g.id_game) DESC
- For "count games by year" queries: SELECT EXTRACT(YEAR FROM g.released) as year, COUNT(g.id_game) FROM games g WHERE g.released IS NOT NULL GROUP BY EXTRACT(YEAR FROM g.released) ORDER BY year DESC
- For "count games by rating" queries: SELECT FLOOR(g.rating) as rating_range, COUNT(g.id_game) FROM games g WHERE g.rating IS NOT NULL GROUP BY FLOOR(g.rating) ORDER BY rating_range DESC
- For "average rating by genre" queries: SELECT gen.name, AVG(g.rating) FROM genres gen JOIN game_genres gg ON gen.id_genre = gg.id_genre JOIN games g ON gg.id_game = g.id_game WHERE g.rating IS NOT NULL GROUP BY gen.name ORDER BY AVG(g.rating) DESC
- When counting by categories (genre, platform, tag), ALWAYS group by the category name, NOT the game name
- When user asks for "games by X", they want to see X categories with game counts, not individual games
- Always add ORDER BY clause to aggregation queries for better results presentation
- Use COUNT(DISTINCT g.id_game) when multiple JOINs might create duplicates

### Important Rules
- For current year queries, use explicit year 2025 instead of CURRENT_DATE
- For genre filtering, use genres table with ILIKE: WHERE gen.name ILIKE '%GenreName%'
- For platform filtering, use platforms table with ILIKE: WHERE p.name ILIKE '%PlatformName%'
- Always complete JOIN statements properly with full table references
- Use proper table aliases consistently (g for games, gen for genres, p for platforms)
- When mentioning specific platforms: PC, PlayStation, Xbox, Nintendo, use ILIKE pattern matching

### Database Schema
The query will run on a database with the following schema:
{schema}

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{user_question}[/QUESTION]
[SQL]"""
    
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    
    # Usar endpoint /generate para text-generation-inference
    payload = {
        "inputs": prompt,
        "parameters": {
            "do_sample": False,
            "max_new_tokens": 150,
            "temperature": 0.1,
            "return_full_text": False
        }
    }
    
    try:
        logger.info(f"Calling SQLCoder endpoint: {SQLCODER_ENDPOINT}")
        response = requests.post(
            f"{SQLCODER_ENDPOINT}/generate",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_sql = ""
            
            # Manejar diferentes formatos de respuesta
            if isinstance(result, list) and len(result) > 0:
                # Formato lista: [{'generated_text': '...'}]
                generated_sql = result[0].get('generated_text', '').strip()
            elif isinstance(result, dict) and 'generated_text' in result:
                # Formato directo: {'generated_text': '...'}
                generated_sql = result['generated_text'].strip()
            else:
                logger.error(f"Unexpected response format: {result}")
                raise ValueError(f"Unexpected response format from SQLCoder")
            
            logger.info(f"SQLCoder generated: {generated_sql[:100]}...")
            return generated_sql
        else:
            logger.error(f"SQLCoder API error {response.status_code}: {response.text}")
            raise ValueError(f"SQLCoder API error: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error to SQLCoder: {e}")
        raise ValueError(f"Failed to connect to SQLCoder endpoint: {e}")

# ---------- Validación y normalización ----------
_VALID_TABLES = {
    "games", "genres", "platforms", "parent_platforms", "tags", "stores", "esrb_ratings",
    "game_genres", "game_platforms", "game_parent_platforms", "game_tags", "game_stores"
}

def normalize_sql(sql: str) -> str:
    """
    Normalización y post-procesamiento del SQL generado por SQLCoder
    """
    if not sql or not sql.strip():
        return ""
    
    sql = sql.strip()
    
    # Limpiar espacios múltiples
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    # POST-PROCESAMIENTO: Corregir consultas de agregación problemáticas
    sql = fix_aggregation_queries(sql)
    
    # Asegurar que termine con punto y coma
    if not sql.endswith(';'):
        sql += ';'
    
    return sql

def fix_aggregation_queries(sql: str) -> str:
    """
    Post-procesamiento generalizado para corregir consultas de agregación incorrectas
    """
    sql_upper = sql.upper()
    
    # DEBUG: Log del SQL original
    logger.info(f"POST-PROCESSING DEBUG: SQL original: {sql}")
    logger.info(f"POST-PROCESSING DEBUG: SQL upper: {sql_upper}")
    
    # Definir patrones de categorías con sus configuraciones
    category_patterns = {
        'GENRE': {
            'table': 'genres',
            'alias': 'gen',
            'join_table': 'game_genres',
            'join_alias': 'gg',
            'pk': 'id_genre',
            'fk': 'id_genre'
        },
        'PLATFORM': {
            'table': 'platforms', 
            'alias': 'p',
            'join_table': 'game_platforms',
            'join_alias': 'gp', 
            'pk': 'id_platform',
            'fk': 'id_platform'
        },
        'TAG': {
            'table': 'tags',
            'alias': 't', 
            'join_table': 'game_tags',
            'join_alias': 'gt',
            'pk': 'id_tag',
            'fk': 'id_tag'
        },
        'STORE': {
            'table': 'stores',
            'alias': 's',
            'join_table': 'game_stores', 
            'join_alias': 'gs',
            'pk': 'id_store',
            'fk': 'id_store'
        }
    }
    
    # Detectar patrón más amplio: COUNT + GROUP BY G.NAME (sin importar el resto)
    if 'COUNT' in sql_upper and 'GROUP BY G.NAME' in sql_upper:
        logger.info("POST-PROCESSING DEBUG: Detectado COUNT + GROUP BY G.NAME")
        
        # Buscar qué categoría está involucrada
        for category, config in category_patterns.items():
            category_table = config['table'].upper()
            
            # Buscar patrones más flexibles
            if (f'JOIN {category_table}' in sql_upper or 
                f'{category_table}' in sql_upper or
                category.lower() in sql.lower()):
                
                logger.info(f"POST-PROCESSING DEBUG: Detectada categoría {category}")
                
                # Generar SQL corregido dinámicamente
                corrected_sql = f"""SELECT {config['alias']}.name, COUNT(DISTINCT g.id_game) as game_count 
                                   FROM {config['table']} {config['alias']}
                                   JOIN {config['join_table']} {config['join_alias']} ON {config['alias']}.{config['pk']} = {config['join_alias']}.{config['fk']}
                                   JOIN games g ON {config['join_alias']}.id_game = g.id_game 
                                   GROUP BY {config['alias']}.name 
                                   ORDER BY game_count DESC"""
                
                logger.info(f"POST-PROCESSING: Corrigiendo agregación por {category.lower()}")
                return corrected_sql
    
    # Detectar agregaciones temporales incorrectas
    if 'COUNT' in sql_upper and 'YEAR' in sql_upper and 'GROUP BY G.NAME' in sql_upper:
        corrected_sql = """SELECT EXTRACT(YEAR FROM g.released) as year, COUNT(g.id_game) as game_count 
                          FROM games g 
                          WHERE g.released IS NOT NULL 
                          GROUP BY EXTRACT(YEAR FROM g.released) 
                          ORDER BY year DESC"""
        logger.info("POST-PROCESSING: Corrigiendo agregación por año")
        return corrected_sql
    
    # Detectar agregaciones por rating incorrectas  
    if 'COUNT' in sql_upper and 'RATING' in sql_upper and 'GROUP BY G.NAME' in sql_upper:
        corrected_sql = """SELECT FLOOR(g.rating) as rating_range, COUNT(g.id_game) as game_count 
                          FROM games g 
                          WHERE g.rating IS NOT NULL 
                          GROUP BY FLOOR(g.rating) 
                          ORDER BY rating_range DESC"""
        logger.info("POST-PROCESSING: Corrigiendo agregación por rating")
        return corrected_sql
    
    logger.info("POST-PROCESSING DEBUG: No se aplicó ninguna corrección")
    return sql

def validate_sql_security(sql: str) -> bool:
    """
    Validación de seguridad para SQL generado
    """
    if not sql or not sql.strip():
        return False
        
    up = sql.upper()
    
    # Solo permitir SELECT
    if not up.startswith("SELECT"):
        return False
    
    # Prohibir operaciones peligrosas
    dangerous_ops = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", 
                    "TRUNCATE", "GRANT", "REVOKE", "EXEC", "EXECUTE"]
    for op in dangerous_ops:
        if re.search(rf'\b{op}\b', up):
            return False
    
    # Validar tablas permitidas
    for match in re.finditer(r'\b(FROM|JOIN)\s+("?[\w\.]+"?)', sql, re.I):
        table = match.group(2).strip('"').split(".")[-1].lower()
        if table.startswith("("):  # subquery
            continue
        if table not in _VALID_TABLES:
            return False
    
    # Verificar paréntesis y comillas balanceados
    if sql.count("(") != sql.count(")"):
        return False
    if (sql.count("'") - sql.count("\\'")) % 2 != 0:
        return False
    
    return True


def is_gaming_related_query(user_question: str) -> bool:
    """
    Determina si la pregunta está relacionada con videojuegos y datos de RAWG
    
    Args:
        user_question: Pregunta del usuario
        
    Returns:
        bool: True si es sobre videojuegos, False si no
    """
    # Palabras clave que indican consultas relacionadas con videojuegos
    gaming_keywords = [
        "game", "games", "gaming", "videogame", "video game",
        "platform", "platforms", "pc", "playstation", "xbox", "nintendo", "steam", "mobile",
        "genre", "genres", "action", "rpg", "strategy", "sports", "racing", "adventure",
        "shooter", "puzzle", "fighting", "horror", "simulation",
        "rating", "ratings", "score", "metacritic", "review", "reviews",
        "rawg", "released", "playtime", "esrb", "developer", "publisher"
    ]
    
    # Términos que claramente NO son sobre videojuegos
    non_gaming_keywords = [
        "weather", "temperature", "stock", "finance", "financial", "money",
        "news", "politics", "political", "recipe", "cooking", "food",
        "medical", "health", "medicine", "travel", "hotel", "flight",
        "movie", "film", "cinema", "music", "song", "book", "literature"
    ]
    
    question_lower = user_question.lower()
    
    # Si contiene palabras claramente no relacionadas con videojuegos, rechazar
    for keyword in non_gaming_keywords:
        if keyword in question_lower:
            return False
    
    # Si contiene palabras relacionadas con videojuegos, aceptar
    for keyword in gaming_keywords:
        if keyword in question_lower:
            return True
    
    # Si no hay palabras clave específicas, asumir que podría ser sobre videojuegos
    return True



# ---------- DB ----------
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"), cursor_factory=RealDictCursor
    )

def execute_sql_query(sql: str, timeout: int = 90) -> List[Dict[str, Any]]:
    # Generar hash de la consulta para cache
    query_hash = hashlib.md5(sql.encode()).hexdigest()
    current_time = time.time()
    
    # Verificar cache
    if query_hash in _query_cache:
        cached_data, cached_time = _query_cache[query_hash]
        if current_time - cached_time < _cache_ttl:
            logger.info(f"Cache HIT para consulta: {sql[:50]}...")
            return cached_data
    
    conn = None
    try:
        logger.info(f"Ejecutando SQL: {sql}")
        start_time = time.time()
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(f"SET statement_timeout = '{timeout}s'")
        cur.execute(sql)
        rows = cur.fetchall() or []
        result = [dict(r) for r in rows]
        
        execution_time = time.time() - start_time
        logger.info(f"Consulta ejecutada en {execution_time:.2f}s")
        
        # Guardar en cache solo si la consulta fue exitosa y no tardó demasiado
        if execution_time < 60:  # Solo cachear consultas que tarden menos de 1 minuto
            _query_cache[query_hash] = (result, current_time)
            logger.info(f"Resultado guardado en cache")
        
        return result
    finally:
        if conn: conn.close()

# ---------- Orquestación ----------
def question_to_sql_finetuned(user_question_en: str) -> Dict[str, Any]:
    t0 = time.time()
    try:
        # Verificar si la pregunta es sobre videojuegos
        if not is_gaming_related_query(user_question_en):
            return {
                "sql": None, 
                "data": [], 
                "error": "Esta API solo responde preguntas sobre videojuegos y datos de RAWG.",
                "metadata": {
                    "model": MODEL_NAME,
                    "question_original": user_question_en,
                    "rows_returned": 0,
                    "columns": [],
                    "elapsed_s": round(time.time()-t0,3),
                    "error_type": "non_gaming_query"
                }
            }
        
        # Bypass directo para consultas explícitas de nombre de juego
        game_name = detect_direct_game_query(user_question_en)
        if game_name:
            logger.info(f"Bypass directo activado para: {game_name}")
            return execute_direct_game_query(game_name, user_question_en, t0)
        
        
        # Generar SQL con SQLCoder-7B-2
        schema = generate_rawg_schema()
        raw_sql = generate_sql_with_sqlcoder(user_question_en, schema)
        sql = normalize_sql(raw_sql)
        
        # Solo validación de seguridad básica
        if not sql or not validate_sql_security(sql):
            return {
                "sql": None,
                "data": [],
                "error": "No se pudo generar una consulta SQL válida.",
                "metadata": {
                    "model": MODEL_NAME,
                    "question_original": user_question_en,
                    "elapsed_s": round(time.time()-t0,3),
                    "error_type": "invalid_sql"
                }
            }
        
        # Ejecutar SQL tal como lo genera el modelo
        data = execute_sql_query(sql, timeout=90)
        return {
            "sql": sql, 
            "data": data,
            "success": True,
            "metadata": {
                "model": MODEL_NAME,
                "question_original": user_question_en,
                "rows_returned": len(data),
                "columns": list(data[0].keys()) if data else [],
                "elapsed_s": round(time.time()-t0,3)
            }
        }
    except Exception as e:
        logger.exception("Error en text2sql")
        return {
            "sql": None, 
            "data": [], 
            "error": f"Error interno: {str(e)}",
            "metadata": {
                "model": MODEL_NAME, 
                "question_original": user_question_en,
                "elapsed_s": round(time.time()-t0,3),
                "error_type": "internal_error"
            }
        }


def detect_direct_game_query(user_question: str) -> str:
    """
    Detecta si la pregunta del usuario es un comando directo "name = juego".
    Solo detecta el patrón exacto para evitar falsos positivos.
    Retorna el nombre del juego si se detecta, None en caso contrario.
    """
    # Solo detectar el comando directo exacto: "name = juego"
    pattern = r'^name\s*=\s*(.+)$'
    
    match = re.search(pattern, user_question.strip(), re.IGNORECASE)
    if match:
        game_name = match.group(1).strip()
        # Limpiar comillas si las tiene
        game_name = game_name.strip('\'"')
        # Verificar que no sea vacío
        if game_name:
            logger.info(f"Comando directo detectado: 'name = {game_name}'")
            return game_name
    
    return None

def execute_direct_game_query(game_name: str, original_question: str, start_time: float) -> Dict[str, Any]:
    """
    Ejecuta una consulta directa para un nombre de juego específico usando el schema real.
    """
    try:
        # Escapar comillas simples para evitar inyección SQL
        safe_game_name = game_name.replace("'", "''")
        
        # SQL usando las tablas reales con información básica
        sql = f"""SELECT g.name, g.rating, g.metacritic, g.playtime, g.released,
                        STRING_AGG(DISTINCT gen.name, ', ') as genres,
                        STRING_AGG(DISTINCT p.name, ', ') as platforms
                 FROM games g
                 LEFT JOIN game_genres gg ON g.id_game = gg.id_game
                 LEFT JOIN genres gen ON gg.id_genre = gen.id_genre
                 LEFT JOIN game_platforms gp ON g.id_game = gp.id_game
                 LEFT JOIN platforms p ON gp.id_platform = p.id_platform
                 WHERE g.name ILIKE '%{safe_game_name}%'
                 GROUP BY g.id_game, g.name, g.rating, g.metacritic, g.playtime, g.released
                 ORDER BY g.rating DESC NULLS LAST
                 LIMIT 10;"""
        
        logger.info(f"Ejecutando consulta directa: {sql}")
        
        # Ejecutar la consulta
        data = execute_sql_query(sql)
        
        elapsed_time = time.time() - start_time
        
        return {
            "sql": sql,
            "data": data,
            "success": True,
            "metadata": {
                "model": "BYPASS_DIRECTO",
                "question_original": original_question,
                "rows_returned": len(data),
                "columns": list(data[0].keys()) if data else [],
                "elapsed_s": round(elapsed_time, 3),
                "bypass_activated": True,
                "game_searched": game_name
            }
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error en consulta directa para '{game_name}': {e}")
        
        return {
            "sql": None,
            "data": [],
            "error": f"Error ejecutando consulta directa para '{game_name}': {str(e)}",
            "metadata": {
                "model": "BYPASS_DIRECTO",
                "question_original": original_question,
                "elapsed_s": round(elapsed_time, 3),
                "error_type": "direct_query_error",
                "game_searched": game_name
            }
        }

def get_model_info() -> Dict[str, Any]:
    """
    Información del modelo SQLCoder-7B-2 via HuggingFace endpoint.
    """
    return {
        "model_name": MODEL_NAME,
        "endpoint": SQLCODER_ENDPOINT,
        "transformers_available": REQUESTS_AVAILABLE,
        "status": "SQLCoder-7B-2 via HuggingFace Inference API"
    }

def test_model_connection() -> Dict[str, Any]:
    """
    Test de conexión al endpoint de SQLCoder.
    """
    if not SQLCODER_ENDPOINT:
        return {"status": "error", "message": "SQLCODER_ENDPOINT not configured"}
    
    try:
        # Test simple al endpoint
        response = requests.get(SQLCODER_ENDPOINT.replace("/generate", "/health"), timeout=5)
        return {"status": "ok", "endpoint": SQLCODER_ENDPOINT}
    except Exception as e:
        return {"status": "error", "message": str(e)}