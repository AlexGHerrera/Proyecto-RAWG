import os, re, time, logging
from typing import Dict, Any, List, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import hashlib
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

# ---------- Schema ----------
def generate_rawg_schema() -> str:
    return """CREATE TABLE games_complete (
  id BIGINT,
  name TEXT,
  year INT,
  hours INT,
  rating REAL,
  popularity BIGINT,
  genres TEXT,
  platforms TEXT,
  tags TEXT
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
            length_penalty=0.8,          # ← Configuración original funcional
            pad_token_id=tok.eos_token_id
        )
    sql = tok.decode(out[0], skip_special_tokens=True).strip().rstrip(";")
    return sql

# ---------- Normalización / validación ligera ----------
_VALID_TABLES = {"games_complete"}

def fix_common_column_names(sql: str) -> str:
    """
    Corrección mínima de nombres de columnas comunes para el schema games_complete.
    Mapea nombres singulares/antiguos a los nombres correctos de la vista.
    """
    if not sql:
        return sql
    
    # Mapeo de nombres comunes (case-insensitive)
    mappings = {
        r'\bgenre\b': 'genres',
        r'\bplaytime\b': 'hours',
        r'\bplatform\b': 'platforms',
        r'\btag\b': 'tags',
        r'\bgame_name\b': 'name',
        r'\bgame_id\b': 'id',
        r'\brelease_year\b': 'year',
        r'\bratings_count\b': 'total_ratings',
        r'\bmetacritic\b': 'metacritic_score',
        r'\badded\b': 'popularity',
        r'\bbackground_image\b': 'image'
    }
    for pattern, replacement in mappings.items():
        sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
    
    return sql

def get_valid_columns() -> List[str]:
    """Devuelve las columnas válidas del schema games_complete"""
    return ['id', 'name', 'year', 'hours', 'rating', 'popularity', 'genres', 'platforms', 'tags']

def get_column_similarity_map():
    """Mapeo inteligente de columnas comunes a columnas válidas"""
    return {
        # Nombres alternativos comunes
        'game': 'name', 'game_name': 'name', 'title': 'name',
        'game_id': 'id', 'number': 'id',
        'genre': 'genres', 'category': 'genres',
        'platform': 'platforms', 'console': 'platforms',
        'tag': 'tags', 'multiplayer': 'tags', 'singleplayer': 'tags',
        'playtime': 'hours', 'duration': 'hours', 'time': 'hours',
        'release_year': 'year', 'released': 'year',
        'added': 'popularity', 'times_added': 'popularity'
    }

def get_data_type_corrections():
    """Correcciones de tipos de datos basadas en el contexto"""
    return {
        # Géneros que se confunden con rating
        'genres_in_rating': {
            'pattern': r"rating\s*=\s*['\"]([A-Za-z]+)['\"]",
            'replacement': lambda match: f"genres LIKE '%{match.group(1)}%'"
        },
        # Valores cualitativos en campos numéricos
        'qualitative_in_numeric': {
            'high': {'rating': '> 4.0', 'popularity': '> 1000'},
            'low': {'rating': '< 2.5', 'popularity': '< 100'},
            'medium': {'rating': 'BETWEEN 2.5 AND 4.0'}
        }
    }

def fix_sql_syntax_errors(sql: str) -> str:
    """Corrección generalizada de errores de sintaxis SQL"""
    # Palabras clave malformadas (patrón generalizado)
    keyword_patterns = [
        (r'\b(ORTH|ORDR|ODER)\b', 'ORDER BY'),
        (r'\b(ORDER\s+BT|ORDER\s+BI)\b', 'ORDER BY'),
        (r'\b(GROUP\s+BT|GROUP\s+BI)\b', 'GROUP BY'),
        (r'\b(SELCT|SLECT)\b', 'SELECT'),
        (r'\b(FORM|FRM)\b(?=\s+\w)', 'FROM'),
        (r'\b(WHRE|WHR)\b', 'WHERE'),
        (r'\b(LIMT|LMT)\b', 'LIMIT')
    ]
    
    for pattern, replacement in keyword_patterns:
        sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
    
    # Sintaxis incompleta común
    syntax_fixes = [
        (r'\bORDER\s+BY\s+LIMIT\b', 'ORDER BY rating DESC LIMIT'),
        (r'\bORDER\s+LIMIT\b', 'ORDER BY rating DESC LIMIT'),
        (r'\bSELECT\s+FROM\b', 'SELECT * FROM'),
        (r'\bWHERE\s+(AND|OR)\b', 'WHERE'),
        (r'\bGROUP\s+BY\s+BY\b', 'GROUP BY'),
        (r'\bORDER\s+BY\s+BY\b', 'ORDER BY')
    ]
    
    for pattern, replacement in syntax_fixes:
        sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
    
    # Corrección generalizada: convertir = con comillas dobles a LIKE para columnas TEXT
    text_columns = {'name', 'genres', 'platforms', 'tags'}
    
    for column in text_columns:
        pattern = rf'\b{column}\s*=\s*"([^"]+)"'
        def replace_with_like(match):
            value = match.group(1)
            return f"{column} LIKE '%{value}%'"
        
        sql = re.sub(pattern, replace_with_like, sql, flags=re.IGNORECASE)
    
    return sql

def fix_column_references(sql: str) -> str:
    """Corrección inteligente de referencias de columnas"""
    valid_columns = get_valid_columns()
    column_map = get_column_similarity_map()
    
    # Extraer todas las palabras alfanuméricas del SQL
    all_words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', sql)
    
    # Filtrar palabras reservadas SQL
    sql_keywords = {
        'select', 'from', 'where', 'order', 'by', 'group', 'having', 'limit', 
        'and', 'or', 'not', 'like', 'in', 'between', 'is', 'null', 'desc', 'asc',
        'max', 'min', 'avg', 'count', 'sum', 'distinct', 'as', 'join', 'inner',
        'left', 'right', 'outer', 'on', 'union', 'all', 'games_complete'
    }
    
    # Identificar posibles columnas
    potential_columns = set()
    for word in all_words:
        if word.lower() not in sql_keywords and len(word) > 1:
            potential_columns.add(word)
    
    # Corregir cada columna potencial que no existe
    for col_ref in potential_columns:
        if col_ref.lower() not in [c.lower() for c in valid_columns]:
            if col_ref.lower() in column_map:
                correct_col = column_map[col_ref.lower()]
                sql = re.sub(rf'\b{re.escape(col_ref)}\b', correct_col, sql, flags=re.IGNORECASE)
                logger.info(f"Columna corregida: {col_ref} → {correct_col}")
            else:
                best_match = find_best_column_match(col_ref, valid_columns)
                if best_match:
                    sql = re.sub(rf'\b{re.escape(col_ref)}\b', best_match, sql, flags=re.IGNORECASE)
                    logger.info(f"Columna corregida por similitud: {col_ref} → {best_match}")
    
    return sql

def find_best_column_match(invalid_col: str, valid_columns: list) -> str:
    """Encuentra la columna más similar usando distancia de cadena"""
    invalid_lower = invalid_col.lower()
    best_match = None
    best_score = 0
    
    for valid_col in valid_columns:
        valid_lower = valid_col.lower()
        score = 0
        
        # Coincidencia exacta de subcadena
        if invalid_lower in valid_lower or valid_lower in invalid_lower:
            score += 50
        
        # Coincidencia de inicio
        if valid_lower.startswith(invalid_lower[:3]) or invalid_lower.startswith(valid_lower[:3]):
            score += 30
        
        # Coincidencia de caracteres comunes
        common_chars = len(set(invalid_lower) & set(valid_lower))
        score += common_chars * 2
        
        # Penalizar diferencia de longitud
        length_diff = abs(len(invalid_lower) - len(valid_lower))
        score -= length_diff * 2
        
        if score > best_score and score > 20:
            best_score = score
            best_match = valid_col
    
    return best_match

def fix_data_type_mismatches(sql: str, user_question: str) -> str:
    """Corrección inteligente de tipos de datos incorrectos"""
    corrections = get_data_type_corrections()
    
    # Corregir géneros en campo rating
    genre_pattern = corrections['genres_in_rating']['pattern']
    if re.search(genre_pattern, sql, re.IGNORECASE):
        sql = re.sub(genre_pattern, corrections['genres_in_rating']['replacement'], sql, flags=re.IGNORECASE)
        logger.info("Corregido: género en campo rating → genres LIKE")
    
    # Corregir valores cualitativos en campos numéricos
    qualitative = corrections['qualitative_in_numeric']
    for qual_value, field_mappings in qualitative.items():
        for field, numeric_condition in field_mappings.items():
            pattern = rf"{field}\s*=\s*['\"]\s*{qual_value}\s*['\"]"
            if re.search(pattern, sql, re.IGNORECASE):
                replacement = f"{field} {numeric_condition}"
                sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
                logger.info(f"Corregido: {field} = '{qual_value}' → {field} {numeric_condition}")
    
    return sql

def fix_semantic_errors(sql: str, user_question: str) -> str:
    """
    Corrección semántica generalizada para mapear términos clave a columnas/valores correctos.
    Detecta cuando el modelo confunde nombres de columnas con valores en el SQL generado.
    """
    # Mapeo semántico: términos detectados en la pregunta → (columna, valor_correcto)
    semantic_mappings = {
        'multiplayer': ('tags', 'multiplayer'),
        'playstation': ('platforms', 'PlayStation'),
        'xbox': ('platforms', 'Xbox'),
        'nintendo': ('platforms', 'Nintendo'),
        'pc': ('platforms', 'PC'),
        'steam': ('platforms', 'PC'),
        'action': ('genres', 'Action'),
        'rpg': ('genres', 'RPG'),
        'strategy': ('genres', 'Strategy'),
        'adventure': ('genres', 'Adventure'),
        'shooter': ('genres', 'Shooter'),
        'racing': ('genres', 'Racing'),
        'sports': ('genres', 'Sports'),
        'simulation': ('genres', 'Simulation'),
        'puzzle': ('genres', 'Puzzle'),
        'fighting': ('genres', 'Fighting')
    }
    
    question_lower = user_question.lower()
    
    # Aplicar correcciones semánticas basadas en términos detectados
    for term, (correct_column, correct_value) in semantic_mappings.items():
        if term in question_lower:
            # Buscar si el término aparece como nombre de columna incorrectamente
            wrong_patterns = [
                rf'\b{re.escape(term)}\s*=',
                rf'\b{re.escape(term)}\s+LIKE',
                rf'WHERE\s+{re.escape(term)}\s*>',
                rf'WHERE\s+{re.escape(term)}\s*<'
            ]
            
            for pattern in wrong_patterns:
                if re.search(pattern, sql, re.IGNORECASE):
                    # Reemplazar con la columna correcta
                    sql = re.sub(pattern, f'{correct_column} LIKE', sql, flags=re.IGNORECASE)
                    logger.info(f"Corrección semántica: {term} → {correct_column} LIKE '%{correct_value}%'")
                    break
    
    # Corrección generalizada: transformar comparaciones = con comillas dobles en columnas TEXT a LIKE
    # Patrón: columna_text = "valor" → columna_text LIKE '%valor%'
    text_columns = ['name', 'genres', 'platforms', 'tags', 'developers', 'publishers']
    
    for column in text_columns:
        # Buscar patrones como: column = "value" o column = 'value'
        pattern = rf'\b{column}\s*=\s*[\'"]([^\'"]+)[\'"]'
        matches = re.finditer(pattern, sql, re.IGNORECASE)
        
        for match in matches:
            value = match.group(1)
            old_clause = match.group(0)
            new_clause = f"{column} LIKE '%{value}%'"
            sql = sql.replace(old_clause, new_clause)
            logger.info(f"Corrección TEXT: {old_clause} → {new_clause}")
    
    # Corrección inteligente de LIMIT: detectar cuando se pregunta por "games" (plural) pero se aplica LIMIT 1
    question_lower = user_question.lower()
    
    # Indicadores de que el usuario espera múltiples resultados
    plural_indicators = ['games', 'titles', 'best games', 'top games', 'good games', 'popular games', 'high rated games']
    singular_indicators = ['game', 'title', 'the best game', 'top game', 'best game']
    
    # Si la pregunta indica plural pero el SQL tiene LIMIT 1, corregir
    is_plural_question = any(indicator in question_lower for indicator in plural_indicators)
    is_singular_question = any(indicator in question_lower for indicator in singular_indicators)
    
    if is_plural_question and not is_singular_question:
        # Cambiar LIMIT 1 por LIMIT 10 para consultas plurales
        if re.search(r'\bLIMIT\s+1\b', sql, re.IGNORECASE):
            sql = re.sub(r'\bLIMIT\s+1\b', 'LIMIT 10', sql, flags=re.IGNORECASE)
            logger.info("Corrección LIMIT: LIMIT 1 → LIMIT 10 (consulta plural detectada)")
    
    return sql

def clean_sql_artifacts(sql: str) -> str:
    """Limpieza final de artefactos SQL"""
    # Limpiar WHERE vacíos o malformados
    sql = re.sub(r'WHERE\s+(AND|OR)\s+', 'WHERE ', sql, flags=re.IGNORECASE)
    sql = re.sub(r'WHERE\s*$', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'WHERE\s*;', ';', sql, flags=re.IGNORECASE)
    
    # Eliminar duplicados en SELECT
    select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
    if select_match:
        columns = [col.strip() for col in select_match.group(1).split(',')]
        unique_columns = list(dict.fromkeys(columns))  # Preservar orden
        if len(unique_columns) != len(columns):
            new_select = 'SELECT ' + ', '.join(unique_columns) + ' FROM'
            sql = re.sub(r'SELECT\s+.+?\s+FROM', new_select, sql, flags=re.IGNORECASE | re.DOTALL)
            logger.info("Eliminadas columnas duplicadas en SELECT")
    
    return sql.strip()

def fix_intelligent_queries(sql: str, user_question: str) -> str:
    """
    Sistema generalizado de corrección de SQL basado en análisis inteligente.
    Aplica correcciones en capas: sintaxis → columnas → tipos → semántica → limpieza.
    """
    if not sql or not user_question:
        return sql
    
    original_sql = sql
    
    # Aplicar correcciones en capas
    sql = fix_sql_syntax_errors(sql)
    sql = fix_column_references(sql)
    sql = fix_data_type_mismatches(sql, user_question)
    sql = fix_semantic_errors(sql, user_question)
    sql = clean_sql_artifacts(sql)
    
    if sql != original_sql:
        logger.info("Sistema de corrección inteligente aplicado")
    
    return sql

def normalize_postgres(sql: str, user_question: str = "") -> str:
    """
    Normalización mínima del SQL generado - limpieza básica + corrección de nombres de columnas + corrección inteligente.
    """
    if not sql or not sql.strip():
        return ""
    
    sql = sql.strip()
    
    # Remover comentarios SQL
    sql = re.sub(r'--.*?\n', '\n', sql)
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    
    # Limpiar espacios múltiples
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    # Corrección mínima de nombres de columnas comunes
    sql = fix_common_column_names(sql)
    
    # Corrección inteligente de consultas
    sql = fix_intelligent_queries(sql, user_question)
    
    # Asegurar que termine con punto y coma
    if not sql.endswith(';'):
        sql += ';'
    
    # Agregar LIMIT si no existe (para evitar consultas muy grandes)
    if 'limit' not in sql.lower() and 'count(' not in sql.lower():
        sql = sql.rstrip(';') + ' LIMIT 100;'
    
    return sql

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
        
        # Generar SQL directamente sin correcciones
        prompt = build_finetuned_prompt(user_question_en)
        raw_sql = generate_sql_with_finetuned(prompt)
        sql = normalize_postgres(raw_sql, user_question_en)
        
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
    Ejecuta una consulta directa para un nombre de juego específico.
    Evita el modelo T5 completamente y usa el nombre exacto del usuario.
    """
    try:
        # SQL directo con el nombre exacto proporcionado por el usuario
        sql = f"SELECT * FROM games_complete WHERE name ILIKE '%{game_name}%' ORDER BY popularity DESC LIMIT 10;"
        
        logger.info(f"Ejecutando consulta directa: {sql}")
        
        # Ejecutar la consulta
        data = execute_sql_query(sql)
        
        elapsed_time = time.time() - start_time
        
        return {
            "sql": sql,
            "data": data,
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