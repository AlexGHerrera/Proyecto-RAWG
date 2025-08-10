import os, re, time, logging
from typing import Dict, Any, List, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import hashlib
from langdetect import detect, LangDetectError
load_dotenv()

logger = logging.getLogger(__name__); logging.basicConfig(level=logging.INFO)

# Cache de consultas para optimización de rendimiento
_query_cache = {}
_cache_ttl = 300  # 5 minutos de cache

MODEL_NAME = "cssupport/t5-small-awesome-text-to-sql"
TOKENIZER_NAME = "t5-small"

_model = _tokenizer = None
_device = "cpu"

try:
    import torch
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# ---------- Schema + few-shot ----------
def generate_rawg_schema() -> str:
    return """CREATE TABLE v_games (
  id BIGINT PRIMARY KEY,
  name TEXT NOT NULL,
  slug TEXT,
  released DATE,
  release_year INT,
  rating REAL,
  ratings_count INT,
  metacritic INT,
  playtime INT,
  background_image TEXT,
  quality_tier TEXT,
  release_period TEXT
);

CREATE TABLE v_game_genres (
  game_id BIGINT,
  game_name TEXT,
  game_slug TEXT,
  rating REAL,
  ratings_count INT,
  released DATE,
  release_year INT,
  metacritic INT,
  playtime INT,
  genre_id BIGINT,
  genre_name TEXT
);

CREATE TABLE v_game_platforms (
  game_id BIGINT,
  game_name TEXT,
  game_slug TEXT,
  rating REAL,
  ratings_count INT,
  released DATE,
  release_year INT,
  metacritic INT,
  playtime INT,
  platform_id BIGINT,
  platform_name TEXT
);

CREATE TABLE v_game_details (
  id BIGINT PRIMARY KEY,
  name TEXT NOT NULL,
  slug TEXT,
  released DATE,
  release_year INT,
  rating REAL,
  rating_top INT,
  ratings_count INT,
  metacritic INT,
  playtime INT,
  suggestions_count INT,
  updated TIMESTAMPTZ,
  background_image TEXT,
  quality_tier TEXT,
  release_period TEXT,
  popularity_score REAL
);"""

def build_finetuned_prompt(user_question_en: str) -> str:
    schema = generate_rawg_schema()
    return f"""tables:
{schema}

query for: {user_question_en}"""

# ---------- Modelo ----------
def load_model():
    global _model, _tokenizer, _device
    if _model is not None:
        return _model, _tokenizer, _device
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Install transformers torch")

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {_device}")
    _tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_NAME)
    _model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    if _device == "cuda":
        _model = _model.half().to("cuda")
    torch.set_num_threads(1)
    _model.eval()
    return _model, _tokenizer, _device

def generate_sql_with_finetuned(prompt: str) -> str:
    model, tok, device = load_model()
    import torch
    
    # Prompt construido correctamente
    
    inputs = tok(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    if device == "cuda": inputs = {k:v.to("cuda") for k,v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            max_length=256,              
            do_sample=False,             
            num_beams=5,                 
            early_stopping=True,         
            no_repeat_ngram_size=2,      
            repetition_penalty=1.1,      
            length_penalty=0.8,          # ← Favorece respuestas más cortas
            pad_token_id=tok.eos_token_id
        )
    sql = tok.decode(out[0], skip_special_tokens=True).strip().rstrip(";")
    return sql

# ---------- Normalización / validación ligera ----------
_VALID_TABLES = {"v_games","v_game_details","v_game_genres","v_game_platforms"}

def normalize_postgres(sql: str, user_question: str = "") -> str:
    s = sql.strip().rstrip(";")
    
    # Reparar patrones problemáticos WHERE rating = 'texto'
    # El modelo confunde rating (numérico) con genre_name/platform_name (texto)
    
    # Corrección específica: rating = 'Action' → genre_name = 'Action'
    # Manejar tanto comillas simples como dobles
    genre_patterns = ['Action', 'Racing', 'Adventure', 'RPG', 'Strategy', 'Simulation', 'Sports', 'Shooter', 'Puzzle']
    for genre in genre_patterns:
        # Comillas simples
        if f"rating = '{genre}'" in s:
            s = s.replace(f"rating = '{genre}'", f"genre_name = '{genre}'")
        # Comillas dobles
        if f'rating = "{genre}"' in s:
            s = s.replace(f'rating = "{genre}"', f"genre_name = '{genre}'")
    
    # Corrección específica: rating = 'PC' → platform_name = 'PC' 
    platform_patterns = ['PC', 'PlayStation', 'Xbox', 'Nintendo', 'iOS', 'Android', 'Windows']
    for platform in platform_patterns:
        # Comillas simples
        if f"rating = '{platform}'" in s:
            s = s.replace(f"rating = '{platform}'", f"platform_name = '{platform}'")
        # Comillas dobles
        if f'rating = "{platform}"' in s:
            s = s.replace(f'rating = "{platform}"', f"platform_name = '{platform}'")
    
    # CORRECCIONES CRÍTICAS PARA CONSULTAS DE PLATAFORMAS PC
    # Problema 1: platform_id = 'PC' → platform_name = 'PC'
    s = re.sub(r"\bplatform_id\s*=\s*['\"]PC['\"]", "platform_name = 'PC'", s, flags=re.I)
    
    # Problema 2: game_name = 'PC' → platform_name = 'PC' (cuando es consulta de plataforma)
    if re.search(r"\b(PC|platform)\b", s, flags=re.I):
        s = re.sub(r"\bgame_name\s*=\s*['\"]PC['\"]", "platform_name = 'PC'", s, flags=re.I)
    
    # Problema 3: Usar tabla correcta para consultas de plataformas
    # Si menciona PC/platform pero usa v_game_genres, cambiar a v_game_platforms
    if re.search(r"\b(PC|platform)\b", s, flags=re.I) and "v_game_genres" in s:
        s = s.replace("v_game_genres", "v_game_platforms")
    
    # Problema 4: Columnas inexistentes comunes
    s = re.sub(r"\bplayer\b", "game_name", s, flags=re.I)
    s = re.sub(r"\bcategory\b", "platform_name", s, flags=re.I)
    
    # CORRECCIÓN CRÍTICA: Consultas específicas de juegos
    # Detectar nombres de juegos en la pregunta original y agregar filtro WHERE
    game_name_patterns = [
        r"rating of ([A-Za-z0-9\s:'-]+)\?",
        r"about ([A-Za-z0-9\s:'-]+)(?:\?|$)",
        r"information about ([A-Za-z0-9\s:'-]+)",
        r"tell me about ([A-Za-z0-9\s:'-]+)",
        r"what is.*of ([A-Za-z0-9\s:'-]+)\?",
        r"^([A-Za-z0-9\s:'-]+) rating$",
        r"^([A-Za-z0-9\s:'-]+) information$",
        r"details of ([A-Za-z0-9\s:'-]+)",
        r"info about ([A-Za-z0-9\s:'-]+)"
    ]
    
    # Si la consulta no tiene WHERE pero debería buscar un juego específico
    if not re.search(r"\bWHERE\b", s, flags=re.I):
        for pattern in game_name_patterns:
            match = re.search(pattern, user_question, flags=re.I)
            if match:
                game_name = match.group(1).strip()
                # Limpiar el nombre del juego
                game_name = re.sub(r"[^\w\s'-]", "", game_name).strip()
                if game_name and len(game_name) > 2:
                    # Agregar filtro WHERE con ILIKE para búsqueda flexible
                    if "FROM v_game_details" in s:
                        s = s.replace("FROM v_game_details", f"FROM v_game_details WHERE name ILIKE '%{game_name}%'")
                    elif "FROM v_games" in s:
                        s = s.replace("FROM v_games", f"FROM v_games WHERE name ILIKE '%{game_name}%'")
                    break
    
    # Reparar el patrón problemático WHERE rating = 'N' (genérico)
    if "WHERE rating = 'N'" in s:
        s = s.replace("WHERE rating = 'N'", "")
        if "ORDER BY" not in s.upper():
            s += " ORDER BY rating DESC"
    
    # Corrección de columnas: en v_games la columna es "name" (no game_name)
    # Reescribir select/where/order comunes donde el modelo usa game_name con v_games
    if re.search(r"\bFROM\s+v_games\b", s, flags=re.I):
        s = re.sub(r"\bgame_name\b", "name", s)
    
    # Corrección crítica: rating_count → ratings_count (error común del modelo)
    s = re.sub(r"\brating_count\b", "ratings_count", s, flags=re.I)
    
    # Correcciones de nombres de columnas truncados
    # platform → platform_name en GROUP BY, ORDER BY
    s = re.sub(r'\bGROUP BY platform\b', 'GROUP BY platform_name', s, flags=re.I)
    s = re.sub(r'\bORDER BY platform\b', 'ORDER BY platform_name', s, flags=re.I)
    
    # genre → genre_name en GROUP BY, ORDER BY
    s = re.sub(r'\bGROUP BY genre\b', 'GROUP BY genre_name', s, flags=re.I)
    s = re.sub(r'\bORDER BY genre\b', 'ORDER BY genre_name', s, flags=re.I)
    
    # CORRECCIÓN CRÍTICA: Forzar ORDER BY en consultas de agregación
    # Si hay GROUP BY y COUNT(*) pero no ORDER BY, agregar ORDER BY COUNT(*) DESC
    has_group_by = re.search(r'\bGROUP BY\b', s, flags=re.I)
    has_count = re.search(r'\bCOUNT\(\*\)', s, flags=re.I)
    has_order_by = re.search(r'\bORDER BY\b', s, flags=re.I)
    
    if has_group_by and has_count and not has_order_by:
        # Insertar ORDER BY antes del LIMIT si existe
        if re.search(r'\bLIMIT\b', s, flags=re.I):
            s = re.sub(r'\bLIMIT\b', 'ORDER BY COUNT(*) DESC LIMIT', s, flags=re.I)
        else:
            s += " ORDER BY COUNT(*) DESC"
    
    # Limitar consultas pesadas en v_games
    if re.search(r'\b(best|top|highest)\b', s, flags=re.I) and "FROM v_games" in s:
        # Añadir filtro de rendimiento
        if "WHERE" not in s.upper():
            s = s.replace("FROM v_games", "FROM v_games WHERE rating >= 4.0")
        # Limitar resultados para evitar timeout
        if "LIMIT" not in s.upper():
            s += " LIMIT 20"
    
    # Otras normalizaciones
    s = re.sub(r'\b(released)\s*=\s*(\d{4})\b', r'EXTRACT(YEAR FROM \1) = \2', s, flags=re.I)
    
    # Límites estrictos para consultas
    if "SELECT" in s.upper() and "LIMIT" not in s.upper():
        s += " LIMIT 20"  # Reducir límite por defecto para mejor rendimiento
    
    # Forzar límite máximo de 50 registros para evitar consultas costosas
    limit_match = re.search(r'LIMIT\s+(\d+)', s, re.I)
    if limit_match:
        limit_value = int(limit_match.group(1))
        if limit_value > 50:
            s = re.sub(r'LIMIT\s+\d+', 'LIMIT 50', s, flags=re.I)
    
    if re.search(r'\b(top|best|highest)\b', s, flags=re.I) and "ORDER BY" not in s.upper():
        s += " ORDER BY rating DESC, ratings_count DESC"
    
    # Corrige WHERE NOT id IN (...) → WHERE id IN (...) en patrones comunes
    s = re.sub(r'WHERE\s+NOT\s+(id|id_game)\s+IN', r'WHERE \1 IN', s, flags=re.I)
    return s

def validate_sql_security(sql: str) -> bool:
    up = sql.upper()
    if not up.startswith("SELECT"): return False
    for op in ("DROP","DELETE","INSERT","UPDATE","ALTER","CREATE","TRUNCATE","GRANT","REVOKE","EXEC"):
        if re.search(rf'\b{op}\b', up): return False
    # tablas tolerantes (no invalidar por mayúsculas/quotes)
    for m in re.finditer(r'\b(FROM|JOIN)\s+("?[\w\.]+"?)', sql, re.I):
        t = m.group(2).strip('"').split(".")[-1].lower()
        if t.startswith("("):  # subquery/cte
            continue
        if t not in _VALID_TABLES:
            logger.warning(f"Tabla no reconocida (tolerada): {t}")
    # comillas/paréntesis balanceados
    if sql.count("(") != sql.count(")"): return False
    if (sql.count("'") - sql.count("\\'")) % 2 != 0: return False
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

def is_english_query(user_question: str) -> bool:
    """
    Determina si la pregunta está en inglés usando detección automática de idioma
    
    Args:
        user_question: Pregunta del usuario
        
    Returns:
        bool: True si está en inglés, False si no
    """
    try:
        # Usar langdetect para detectar el idioma automáticamente
        detected_language = detect(user_question)
        return detected_language == 'en'
    except LangDetectError:
        # Si no se puede detectar el idioma (texto muy corto, etc.), asumir inglés
        # para no bloquear consultas válidas
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
def question_to_sql_finetuned(user_question_en: str, max_retries: int = 1) -> Dict[str, Any]:
    t0 = time.time()
    try:
        # Verificar si la pregunta es sobre videojuegos
        if not is_gaming_related_query(user_question_en):
            return {
                "sql": None, 
                "data": [], 
                "error": "Esta API solo responde preguntas sobre videojuegos y datos de RAWG. Por favor, haz una pregunta relacionada con juegos, plataformas, géneros o ratings.",
                "metadata": {
                    "model": MODEL_NAME,
                    "question_original": user_question_en,
                    "rows_returned": 0,
                    "columns": [],
                    "fallback_used": False,
                    "elapsed_s": round(time.time()-t0,3),
                    "error_type": "non_gaming_query"
                }
            }
        
        prompt = build_finetuned_prompt(user_question_en)
        raw_sql = generate_sql_with_finetuned(prompt)
        sql = normalize_postgres(raw_sql, user_question_en)
        
        if not sql or not validate_sql_security(sql):
            return {
                "sql": None,
                "data": [],
                "error": "No puedo responder a esa pregunta, lo siento. Intenta reformular tu consulta o pregunta algo más específico sobre videojuegos.",
                "metadata": {
                    "model": MODEL_NAME,
                    "question_original": user_question_en,
                    "rows_returned": 0,
                    "columns": [],
                    "fallback_used": False,
                    "elapsed_s": round(time.time()-t0,3),
                    "error_type": "cannot_generate_sql"
                }
            }
        
        data = execute_sql_query(sql, timeout=90)
        return {
            "sql": sql, "data": data,
            "metadata": {
                "model": MODEL_NAME,
                "question_original": user_question_en,
                "rows_returned": len(data),
                "columns": list(data[0].keys()) if data else [],
                "fallback_used": False,
                "elapsed_s": round(time.time()-t0,3)
            }
        }
    except Exception as e:
        logger.exception("text2sql error")
        return {
            "sql": None, 
            "data": [], 
            "error": "No puedo responder a esa pregunta, lo siento. Ha ocurrido un error interno al procesar tu consulta.",
            "metadata": {
                "model": MODEL_NAME, 
                "question_original": user_question_en,
                "fallback_used": False,
                "elapsed_s": round(time.time()-t0,3),
                "error_type": "internal_error",
                "error_details": str(e)
            }
        }

def get_model_info() -> Dict[str, Any]:
    return {"model_name": MODEL_NAME, "tokenizer_name": TOKENIZER_NAME,
            "device": _device if _model else "Not loaded",
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "schema_views": sorted(_VALID_TABLES)}

# Utilidad: conexión/prueba rápida del modelo
def test_model_connection() -> bool:
    """
    Realiza una comprobación para verificar que el stack de transformers
    esté disponible y que el modelo/tokenizer pueden inicializarse.
    Llama a load_model() que gestiona la carga y el estado.
    """
    try:
        # load_model() se encarga de la inicialización y la gestión de estado.
        model, tok, _ = load_model()
        # Si la carga es exitosa, los objetos no serán None.
        return model is not None and tok is not None
    except Exception:
        logger.exception("test_model_connection failed")
        return False