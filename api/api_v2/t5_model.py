"""
Módulo utilitario para cargar y usar un modelo T5-small para traducir preguntas
en lenguaje natural a consultas SQL adaptadas para la base de datos de juegos RAWG.
El punto de entrada principal es :func:`question_to_sql` que acepta una pregunta del usuario
y retorna una cadena SQL de mejor esfuerzo. Internamente las instancias del modelo y tokenizer
se cargan de forma perezosa en el primer uso y se almacenan en caché para llamadas posteriores.

La plantilla de prompt incluye una descripción concisa de las tablas disponibles
y relaciones. Esto ayuda a guiar al modelo hacia la generación de consultas PostgreSQL
sintácticamente válidas. Después de la generación, heurísticas simples intentan
extraer solo la porción SQL de la respuesta del modelo y aplicar restricciones
básicas de validez.

Nota: Este módulo depende de los paquetes ``transformers`` y ``torch``. Si
estos no están instalados en tu entorno de ejecución la importación fallará.
El código que llama debe manejar cualquier excepción resultante apropiadamente.
"""

from __future__ import annotations

from typing import Tuple, Dict, Optional
import re
import hashlib
import time

try:
    # Las importaciones pesadas se difieren hasta el tiempo de ejecución para evitar penalizaciones
    # de importación cuando el modelo no se usa realmente. ``transformers`` proporciona las
    # clases AutoModelForSeq2SeqLM y AutoTokenizer usadas aquí.
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch
except Exception as e:  # pragma: no cover
    # Si la librería transformers no está disponible en el momento de importación la
    # excepción se captura para que los consumidores aún puedan importar este módulo.
    AutoModelForSeq2SeqLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    torch = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

# Cachés a nivel de módulo para el modelo y tokenizer. Se inicializan en
# el primer uso y posteriormente se reutilizan para evitar descargas repetidas
# y sobrecarga de instanciación.
_t5_model: Optional["AutoModelForSeq2SeqLM"] = None
_t5_tokenizer: Optional["AutoTokenizer"] = None

_query_cache: Dict[str, str] = {}

# Descripción concisa del esquema usada para preparar el modelo. Incluir todo el
# archivo de esquema textualmente haría el prompt difícil de manejar y degradaría
# el rendimiento. Este resumen destaca las tablas más importantes y cómo
# se relacionan entre sí.
RAWG_SCHEMA_SUMMARY = """
RAWG Games Database Schema (summary):

Tables and key fields:
- games(id_game, name, released, rating, playtime, metacritic, esrb_rating_id)
- genres(id_genre, name)
- game_genres(id_game, id_genre)
- platforms(id_platform, name)
- game_platforms(id_game, id_platform)
- tags(id_tag, name)
- game_tags(id_game, id_tag)
- stores(id_store, name)
- game_stores(id_game, id_store)
- ratings(id_rating, title, count, percent)
- game_ratings(id_game, id_rating)
- esrb_ratings(id_esrb_rating, name)
- game_added_by_status(id_game, status, count)

Relationships:
- games ↔ game_genres ↔ genres
- games ↔ game_platforms ↔ platforms
- games ↔ game_tags ↔ tags
- games ↔ game_stores ↔ stores
- games ↔ game_ratings ↔ ratings
- games ↔ game_added_by_status

Common Query Patterns:
- For genres analysis: SELECT g.name, COUNT(*) FROM genres g JOIN game_genres gg ON g.id_genre = gg.id_genre JOIN games gm ON gg.id_game = gm.id_game GROUP BY g.name
- For platforms analysis: SELECT p.name, COUNT(*) FROM platforms p JOIN game_platforms gp ON p.id_platform = gp.id_platform GROUP BY p.name
- For top games: SELECT name, rating FROM games WHERE rating IS NOT NULL ORDER BY rating DESC LIMIT 10
- For averages by category: Use GROUP BY with AVG(), COUNT(), SUM() functions

Only read‑only SELECT queries are permitted.  Avoid destructive SQL such
as INSERT, UPDATE or DROP.  Write a single PostgreSQL query that answers
the user’s question.
"""


def load_t5_model() -> Tuple["AutoModelForSeq2SeqLM", "AutoTokenizer"]:
    """
    Carga de forma perezosa el modelo T5-small y su tokenizer. La primera llamada
    descarga e instancia el modelo; las llamadas posteriores retornan las
    instancias en caché.

    Retorna
    -------
    Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]
        El modelo y tokenizer listos para inferencia.

    Lanza
    -----
    ImportError
        Si las librerías transformers o torch no están instaladas.
    """
    global _t5_model, _t5_tokenizer, _IMPORT_ERROR
    if _IMPORT_ERROR is not None:
        # Surface the import error when attempting to load the model.  This
        # avoids hiding misconfiguration until later use.
        raise ImportError(
            "transformers or torch is not available in the current environment"
        ) from _IMPORT_ERROR

    if _t5_model is None or _t5_tokenizer is None:
        model_name = "t5-small"
        # Log to stderr so that loading progress is visible if run in a
        # console.  Replace with logger as appropriate in your application.
        print(f"Cargando modelo {model_name}…")
        start = time.time()
        _t5_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _t5_model.eval()
        duration = time.time() - start
        print(f"Modelo {model_name} cargado en {duration:.2f} s")
    return _t5_model, _t5_tokenizer  # type: ignore[return-value]


def build_prompt(user_question: str) -> str:
    """
    Construye el prompt presentado al modelo T5. Embebe un resumen del
    esquema RAWG junto con la pregunta en lenguaje natural del usuario. Se
    instruye al modelo para que genere una consulta PostgreSQL.

    Parámetros
    ----------
    user_question : str
        La pregunta planteada por el usuario en lenguaje natural.

    Retorna
    -------
    str
        Una cadena de prompt formateada.
    """
    prompt = (
        "### Task\n"
        "Traducir la pregunta en lenguaje natural a una consulta PostgreSQL SELECT.\n\n"
        "### Database Schema (summary)\n"
        f"{RAWG_SCHEMA_SUMMARY}\n\n"
        "### Question\n"
        f"{user_question}\n\n"
        "### SQL Query\n"
    )
    return prompt


def generate_sql(prompt: str) -> str:
    """
    Generate SQL from the provided prompt using the T5 model.  The model
    produces text which may contain commentary or extraneous output; this
    function extracts and cleans the SQL portion.

    Parameters
    ----------
    prompt : str
        The complete prompt including the schema summary and user question.

    Returns
    -------
    str
        A candidate SQL query trimmed from the model’s response.
    """
    model, tokenizer = load_t5_model()

    # Tokenise the input prompt.  T5 uses an encoder‑decoder architecture,
    # therefore both input and output tokens are handled by the tokenizer.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=True,
    )

    # Generate the output sequence.  Beam search is disabled for speed; the
    # temperature is kept low to encourage deterministic behaviour.  Adjust
    # parameters as needed for your application.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=1,
            do_sample=False,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return _extract_sql_from_response(generated_text, prompt)


def _extract_sql_from_response(generated_text: str, original_prompt: str) -> str:
    """
    Remove the prompt from the model's output and attempt to isolate the
    SQL statement.  If no obvious SELECT or WITH clause is found, the
    remainder of the text is returned as‑is.

    Parameters
    ----------
    generated_text : str
        The raw output text from the model, including the prompt.
    original_prompt : str
        The prompt that was fed to the model.  It is stripped from the
        beginning of the output if present.

    Returns
    -------
    str
        The extracted SQL statement, always terminating with a semicolon.
    """
    # Remove the prompt from the output, if present
    if generated_text.startswith(original_prompt):
        sql_part = generated_text[len(original_prompt):].strip()
    else:
        sql_part = generated_text.strip()

    # Clean any schema contamination patterns
    contamination_patterns = [
        r'\s*-\s*For\s+.*$',
        r'\s*-For\s+.*$', 
        r'\s*Common Query Patterns.*$',
        r'\s*Examples:.*$',
        r'\s*Schema:.*$'
    ]
    
    for pattern in contamination_patterns:
        sql_part = re.sub(pattern, '', sql_part, flags=re.IGNORECASE | re.MULTILINE)

    # T5 often generates just the SQL directly, so let's try a simpler approach first
    # Look for SELECT or WITH at the beginning of the response
    if sql_part.upper().startswith('SELECT') or sql_part.upper().startswith('WITH'):
        # Clean the SQL directly
        lines = sql_part.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Stop at schema examples or comments
            if (line.startswith('-') or 'Common Query Patterns' in line or 
                line.startswith('For ') or line.startswith('Examples:') or
                line.startswith('Schema:')):
                break
            clean_lines.append(line)
            if line.endswith(';'):
                break
        
        if clean_lines:
            result = ' '.join(clean_lines).strip()
            # Additional cleaning for contamination
            result = re.sub(r'\s*-\s*For.*$', '', result)
            result = re.sub(r'\s*-For.*$', '', result)
            result = result if result.endswith(";") else result + ";"
            return result

    # Remove any remaining schema or context text
    # Look for the actual SQL query after "### SQL Query" or similar markers
    sql_markers = ["### SQL Query", "SQL Query:", "Query:", "SELECT", "WITH"]
    for marker in sql_markers:
        if marker in sql_part:
            # Find the position after the marker
            marker_pos = sql_part.find(marker)
            if marker == "SELECT" or marker == "WITH":
                sql_part = sql_part[marker_pos:]
            else:
                sql_part = sql_part[marker_pos + len(marker):].strip()
            break

    # Clean up any remaining non-SQL text before SELECT/WITH
    lines = sql_part.split('\n')
    sql_lines = []
    found_sql = False
    
    for line in lines:
        line = line.strip()
        if not found_sql:
            # Look for the start of SQL
            if line.upper().startswith('SELECT') or line.upper().startswith('WITH'):
                found_sql = True
                sql_lines.append(line)
        else:
            # We're in SQL, keep adding lines until we hit a semicolon, dash, or "For" (schema examples)
            if (line.startswith('-') or line.upper().startswith('FOR ') or 
                'Common Query Patterns' in line or line.startswith('Examples:') or
                line.startswith('Schema:')):
                break  # Stop if we hit schema examples
            sql_lines.append(line)
            if line.endswith(';'):
                break
    
    if sql_lines:
        sql_result = ' '.join(sql_lines).strip()
        # Remove any trailing schema text that might have been included
        sql_result = re.sub(r'\s*-\s*For.*$', '', sql_result)
        sql_result = re.sub(r'\s*-For.*$', '', sql_result)
        result = sql_result if sql_result.endswith(";") else sql_result + ";"
        return result
    
    # Fallback: try regex patterns
    patterns = [
        r"(SELECT\s+.*?);",
        r"(WITH\s+.*?);",
        r"(SELECT\s+.*?)(?=\s*-\s*For|\s*-For|\s*Common Query|\s*Examples:|\s*Schema:|\n\s*$)",
    ]
    for pattern in patterns:
        m = re.search(pattern, sql_part, re.IGNORECASE | re.DOTALL)
        if m:
            candidate = m.group(1).strip()
            candidate = re.sub(r'\s*-\s*For.*$', '', candidate)
            candidate = re.sub(r'\s*-For.*$', '', candidate)
            result = candidate if candidate.endswith(";") else candidate + ";"
            return result

    # Last resort: return a simple default query
    return "SELECT g.name, g.rating FROM games g WHERE g.rating IS NOT NULL ORDER BY g.rating DESC LIMIT 10;"


def clean_sql(sql: str) -> str:
    """
    Normalise whitespace and enforce a terminating semicolon on the SQL
    statement.

    Parameters
    ----------
    sql : str
        The raw SQL string.

    Returns
    -------
    str
        A cleaned SQL string.
    """
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql if sql.endswith(";") else sql + ";"


def validate_sql(sql: str) -> Tuple[bool, str]:
    """
    Perform rudimentary validation of the generated SQL to ensure it is a
    read‑only SELECT query referencing at least one known table.  This
    function does not execute the query and cannot guarantee correctness.

    Parameters
    ----------
    sql : str
        The SQL statement to validate.

    Returns
    -------
    tuple[bool, str]
        A tuple of (is_valid, message).  ``is_valid`` is ``True`` when the
        SQL passes all checks; otherwise ``False``.  ``message`` contains
        feedback on the failure or success condition.
    """
    if not sql or not isinstance(sql, str):
        return False, "Consulta SQL vacía o inválida"

    s = sql.strip().lower()
    if not (s.startswith("select") or s.startswith("with")):
        return False, "Solo se permiten consultas SELECT o CTE"

    forbidden = ["drop", "delete", "update", "insert", "alter", "truncate", "create"]
    if any(op in s for op in forbidden):
        return False, "Operación no permitida detectada"

    # Ensure at least one known table name is referenced
    valid_tables = [
        "games", "genres", "game_genres", "platforms", "game_platforms",
        "tags", "game_tags", "stores", "game_stores", "ratings", "game_ratings",
        "esrb_ratings", "game_added_by_status", "parent_platforms",
        "game_parent_platforms"
    ]
    if not any(tbl in s for tbl in valid_tables):
        return False, "No se detectaron tablas válidas"
    # Ensure statement terminates with semicolon
    if not sql.strip().endswith(";"):
        return False, "SQL debe terminar con punto y coma"

    return True, "Consulta válida"


def detect_query_intent_and_fix_sql(user_question: str, generated_sql: str) -> str:
    """
    Detect the user's intent and fix the SQL if it doesn't match the expected pattern.
    
    Parameters
    ----------
    user_question : str
        The user's original question
    generated_sql : str
        The SQL generated by the model
        
    Returns
    -------
    str
        Fixed SQL that better matches the user's intent
    """
    question_lower = user_question.lower()
    sql_lower = generated_sql.lower()
    
    # Intent: Count/number by genre
    if any(word in question_lower for word in ['número', 'count', 'cantidad', 'cuántos']) and 'género' in question_lower:
        if 'group by' not in sql_lower or 'genres' not in sql_lower:
            return """SELECT g.name as genre, COUNT(*) as game_count 
                     FROM genres g 
                     JOIN game_genres gg ON g.id_genre = gg.id_genre 
                     JOIN games gm ON gg.id_game = gm.id_game 
                     GROUP BY g.name 
                     ORDER BY game_count DESC;"""
    
    # Intent: Count/number by platform
    if any(word in question_lower for word in ['número', 'count', 'cantidad', 'cuántos']) and 'plataforma' in question_lower:
        if 'group by' not in sql_lower or 'platforms' not in sql_lower:
            return """SELECT p.name as platform, COUNT(*) as game_count 
                     FROM platforms p 
                     JOIN game_platforms gp ON p.id_platform = gp.id_platform 
                     GROUP BY p.name 
                     ORDER BY game_count DESC;"""
    
    # Intent: Average by genre
    if any(word in question_lower for word in ['promedio', 'media', 'average']) and 'género' in question_lower:
        if 'group by' not in sql_lower or 'avg(' not in sql_lower:
            return """SELECT g.name as genre, AVG(gm.rating) as avg_rating, COUNT(*) as game_count
                     FROM genres g 
                     JOIN game_genres gg ON g.id_genre = gg.id_genre 
                     JOIN games gm ON gg.id_game = gm.id_game 
                     WHERE gm.rating IS NOT NULL
                     GROUP BY g.name 
                     ORDER BY avg_rating DESC;"""
    
    # Intent: Distribution/frequency
    if any(word in question_lower for word in ['distribución', 'distribution', 'frecuencia']):
        if 'género' in question_lower and 'group by' not in sql_lower:
            return """SELECT g.name as genre, COUNT(*) as frequency 
                     FROM genres g 
                     JOIN game_genres gg ON g.id_genre = gg.id_genre 
                     GROUP BY g.name 
                     ORDER BY frequency DESC;"""
    
    # Intent: Top genres (most popular)
    if any(word in question_lower for word in ['top', 'mejor', 'mejores', 'populares', 'más']) and 'género' in question_lower:
        if 'group by' not in sql_lower:
            return """SELECT g.name as genre, COUNT(*) as popularity 
                     FROM genres g 
                     JOIN game_genres gg ON g.id_genre = gg.id_genre 
                     GROUP BY g.name 
                     ORDER BY popularity DESC 
                     LIMIT 15;"""
    
    # Intent: Best games of specific genre (RPG, Action, etc.)
    specific_genres = ['rpg', 'action', 'adventure', 'strategy', 'simulation', 'sports', 'racing', 'shooter', 'puzzle', 'platformer']
    genre_found = None
    for genre in specific_genres:
        if genre in question_lower:
            genre_found = genre
            break
    
    if genre_found and any(word in question_lower for word in ['mejor', 'mejores', 'best', 'top', 'bueno', 'buenos']):
        # Check if the current SQL doesn't filter by genre
        if 'join' not in sql_lower or 'genres' not in sql_lower or genre_found not in sql_lower:
            return f"""SELECT gm.name, gm.rating, g.name as genre
                      FROM games gm
                      JOIN game_genres gg ON gm.id_game = gg.id_game
                      JOIN genres g ON gg.id_genre = g.id_genre
                      WHERE LOWER(g.name) LIKE '%{genre_found}%' 
                      AND gm.rating IS NOT NULL
                      ORDER BY gm.rating DESC
                      LIMIT 20;"""
    
    # Intent: Games by specific platform
    platforms = ['pc', 'playstation', 'xbox', 'nintendo', 'switch', 'ps4', 'ps5', 'xbox one', 'steam']
    platform_found = None
    for platform in platforms:
        if platform in question_lower:
            platform_found = platform
            break
    
    if platform_found and any(word in question_lower for word in ['juegos', 'games', 'mejor', 'mejores', 'top']):
        if 'join' not in sql_lower or 'platforms' not in sql_lower:
            return f"""SELECT gm.name, gm.rating, p.name as platform
                      FROM games gm
                      JOIN game_platforms gp ON gm.id_game = gp.id_game
                      JOIN platforms p ON gp.id_platform = p.id_platform
                      WHERE LOWER(p.name) LIKE '%{platform_found}%'
                      AND gm.rating IS NOT NULL
                      ORDER BY gm.rating DESC
                      LIMIT 20;"""
    
    # Intent: Games by year
    if any(word in question_lower for word in ['año', 'year', '2023', '2022', '2021', '2020']) and any(word in question_lower for word in ['juegos', 'games', 'lanzados', 'released']):
        if 'released' not in sql_lower and 'extract' not in sql_lower:
            return """SELECT name, rating, released
                     FROM games
                     WHERE released IS NOT NULL
                     AND rating IS NOT NULL
                     ORDER BY released DESC, rating DESC
                     LIMIT 20;"""
    
    # Return original SQL if no specific intent detected
    return generated_sql


def question_to_sql(user_question: str) -> str:
    """
    Convierte una pregunta en lenguaje natural en una consulta SQL usando el modelo
    de lenguaje T5. Los resultados se almacenan en caché por pregunta para evitar
    inferencia repetida para la misma entrada.

    Parámetros
    ----------
    user_question : str
        La pregunta del usuario en lenguaje natural.

    Retorna
    -------
    str
        La consulta SQL generada.
    """
    if not user_question:
        return ""

    # Normalise and hash the question for caching
    key = hashlib.md5(user_question.strip().lower().encode()).hexdigest()
    if key in _query_cache:
        return _query_cache[key]

    prompt = build_prompt(user_question)
    raw_sql = generate_sql(prompt)
    cleaned_sql = clean_sql(raw_sql)
    
    # Apply intent detection and SQL fixing
    fixed_sql = detect_query_intent_and_fix_sql(user_question, cleaned_sql)
    
    _query_cache[key] = fixed_sql
    return fixed_sql