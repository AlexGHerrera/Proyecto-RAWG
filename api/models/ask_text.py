# Carga manual del modelo y configuración del esquema (sin pipeline)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# Nombre del modelo

model_id = "PipableAI/pip-sql-1.3b"

# Tokenizador y modelo
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Uso de CPU (compatible con EC2 t3.medium o Mac M2)
device = torch.device("cpu")
model.to(device)

# Prompt del esquema RAWG
schema_prompt = """
### Base de datos PostgreSQL: RAWG Video Games

Tablas principales:
- games(id_game, name, released, rating, rating_top, ratings_count, metacritic, tba, playtime, updated, esrb_rating_id)
- genres(id_genre, name)
- game_genres(id_game, id_genre)
- platforms(id_platform, name)
- game_platforms(id_game, id_platform, released_at)
- parent_platforms(id_parent_platform, name)
- game_parent_platforms(id_game, id_parent_platform)
- tags(id_tag, name)
- game_tags(id_game, id_tag)
- stores(id_store, name)
- game_stores(id_game, id_store)
- ratings(id_rating, title, count, percent)
- game_ratings(id_game, id_rating)
- esrb_ratings(id_esrb_rating, name)
- game_added_by_status(id_game, status, count)

### Relaciones clave:
- Un juego puede tener múltiples géneros, plataformas, tags, tiendas y ratings.
- La tabla `game_platforms` enlaza juegos con plataformas y fechas de lanzamiento específicas.
- La tabla `ratings` define tipos de calificación como “exceptional”, “meh”, etc.
- `game_added_by_status` representa cuántos usuarios tienen un juego en estados como “playing” o “completed”.

### Restricciones semánticas:
- Las preguntas deben referirse a datos relacionados con videojuegos, plataformas, géneros, puntuaciones, fechas de lanzamiento, etiquetas u otras métricas del ecosistema RAWG.
- Si la pregunta no está relacionada con videojuegos o con esta base de datos, no debe generarse ninguna consulta SQL.
"""

# Función generadora
def question_to_sql(user_question: str) -> str:
    prompt = f"{schema_prompt}\n\n### Pregunta:\n{user_question}\n\n### SQL:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=256)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Usamos regex para extraer desde "SELECT ..." hasta antes del siguiente "###"
    match = re.search(r"### SQL:\s*(select.+?)(?:\n###|\Z)", decoded, re.IGNORECASE | re.DOTALL)
    if match:
        sql_code = match.group(1).strip()
        return sql_code
    else:
        return ""  # Devolver cadena vacía ante fallo de extracción

def validar_sql_generada(sql_code: str) -> (bool, str):
    if not sql_code or not isinstance(sql_code, str):
        return False, "La consulta SQL está vacía o no es válida."

    sql_lower = sql_code.strip().lower()
    if not sql_lower.startswith("select"):
        return False, "Solo se permiten consultas que empiecen con SELECT."

    # Bloquear comandos peligrosos
    if re.search(r"\b(drop|delete|update|insert|alter|truncate)\b", sql_lower):
        return False, "La consulta contiene operaciones no permitidas."

    tablas_validas = [
        "games", "genres", "game_genres", "platforms", "game_platforms", "tags",
        "game_tags", "stores", "game_stores", "ratings", "game_ratings",
        "esrb_ratings", "game_added_by_status", "parent_platforms", "game_parent_platforms"
    ]
    if not any(tabla in sql_lower for tabla in tablas_validas):
        return False, "No se detectaron tablas válidas en la consulta."

    return True, "Consulta válida."

# Código de prueba - solo se ejecuta si se llama directamente
#if __name__ == "__main__":
    # Prueba aquí tus preguntas
    #question = "what are the best 10 rated games?"
    #generated_sql = question_to_sql(question)
    #print("Pregunta:", question)
    #print("\nSQL generada:\n", generated_sql)
    #print(validar_sql_generada(generated_sql))