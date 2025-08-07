# Singleton para carga única del modelo transformer

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import os
from typing import Tuple, Optional

class SQLModelSingleton:
    """Singleton para manejar la carga única del modelo transformer SQL"""
    _instance = None
    _model = None
    _tokenizer = None
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SQLModelSingleton, cls).__new__(cls)
        return cls._instance
    
    def initialize(self, model_id: str = "PipableAI/pip-sql-1.3b"):
        """Inicializa el modelo una sola vez"""
        if self._model is None:
            print(f"Cargando modelo SQL: {model_id}...")
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._model = AutoModelForCausalLM.from_pretrained(model_id)
            
            # Uso de CPU (compatible con EC2 t3.medium o Mac M2)
            self._device = torch.device("cpu")
            self._model.to(self._device)
            print("Modelo SQL cargado exitosamente.")
    
    @property
    def model(self):
        if self._model is None:
            self.initialize()
        return self._model
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self.initialize()
        return self._tokenizer
    
    @property
    def device(self):
        if self._device is None:
            self.initialize()
        return self._device

# Instancia global del singleton
sql_model = SQLModelSingleton()


def get_schema_prompt(visual_mode: bool = False) -> str:
    """Genera el prompt del esquema RAWG con instrucciones específicas según el modo"""
    base_schema = """
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

Relaciones clave:
- Un juego puede tener múltiples géneros, plataformas, tags, tiendas y ratings.
- La tabla `game_platforms` enlaza juegos con plataformas y fechas de lanzamiento específicas.
- La tabla `ratings` define tipos de calificación como "exceptional", "meh", etc.
- `game_added_by_status` representa cuántos usuarios tienen un juego en estados como "playing" o "completed".

- Debes usar la función de agregación COUNT(), cuando quieras usar cualquier conteo de columnas.

Restricciones semánticas:
- Las preguntas deben referirse a datos relacionados con videojuegos, plataformas, géneros, puntuaciones, fechas de lanzamiento, etiquetas u otras métricas del ecosistema RAWG.
- Si la pregunta no está relacionada con videojuegos o con esta base de datos, no debe generarse ninguna consulta SQL."""
    
    if visual_mode:
        visual_instructions = """

Instrucciones para visualización:
- Si la pregunta implica contar, agrupar o comparar, usa agregaciones como COUNT, AVG, MAX o GROUP BY.
- Devuelve columnas que puedan representarse gráficamente: fechas, categorías, valores numéricos.
- Evita SELECT * y enfócate en columnas relevantes para gráficos."""
        return base_schema + visual_instructions
    
    return base_schema

def extract_sql_from_response(decoded_response: str) -> str:
    """Extrae la consulta SQL de la respuesta del modelo con múltiples patrones"""
    # Patrones de extracción más robustos
    patterns = [
        r"### SQL:\s*(select.+?)(?:\n###|\Z)",  # Patrón original
        r"```sql\s*(select.+?)```",  # Formato markdown
        r"SQL:\s*(select.+?)(?:\n|\Z)",  # Sin ###
        r"(select.+?)(?:\n\n|\Z)",  # Solo SELECT hasta doble salto
    ]
    
    for pattern in patterns:
        match = re.search(pattern, decoded_response, re.IGNORECASE | re.DOTALL)
        if match:
            sql_code = match.group(1).strip()
            # Limpiar caracteres no deseados
            sql_code = re.sub(r'[\n\r]+', ' ', sql_code)
            sql_code = re.sub(r'\s+', ' ', sql_code)
            return sql_code
    
    return "[ERROR] No se pudo extraer una consulta SQL válida."

def question_to_sql(user_question: str, visual_mode: bool = False) -> str:
    """Genera consulta SQL a partir de pregunta en lenguaje natural"""
    try:
        # Usar el singleton para acceder al modelo
        schema_prompt = get_schema_prompt(visual_mode)
        prompt = f"{schema_prompt}\n\n### Pregunta:\n{user_question}\n\n### SQL:"
        
        # Generar con el modelo singleton
        inputs = sql_model.tokenizer(prompt, return_tensors="pt").to(sql_model.device)
        outputs = sql_model.model.generate(
            **inputs, 
            max_new_tokens=256,
            temperature=0.1,  # Más determinístico
            do_sample=True,
            pad_token_id=sql_model.tokenizer.eos_token_id
        )
        decoded = sql_model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return extract_sql_from_response(decoded)
        
    except Exception as e:
        return f"[ERROR] Error generando SQL: {str(e)}"

def validar_sql_generada(sql_code: str) -> Tuple[bool, str]:
    """
    Verifica si la consulta SQL generada es válida para su ejecución.

    Args:
        sql_code: Consulta SQL a validar
    
    Returns:
        Tuple[bool, str]: (es_valida, mensaje_validacion)
    
    Criterios:
    - Debe comenzar con SELECT
    - No debe contener operaciones peligrosas (DROP, DELETE, UPDATE, INSERT, etc.)
    - Debe contener al menos una tabla válida del esquema
    - No debe estar vacía
    """
    if not sql_code or not isinstance(sql_code, str):
        return False, "La consulta SQL está vacía o no es una cadena de texto."
    
    # Manejar errores del modelo
    if sql_code.startswith("[ERROR]"):
        return False, f"Error del modelo: {sql_code}"

    sql_lower = sql_code.strip().lower()
    
    # Limpiar comentarios y espacios
    sql_trimmed = re.sub(r"^\s+|(--.*\n)+", "", sql_lower, flags=re.MULTILINE)
    if not sql_trimmed.startswith("select"):
        return False, "Solo se permiten consultas SELECT."

    # Evitar consultas peligrosas
    dangerous_operations = ["drop", "delete", "update", "insert", "alter", "truncate", "create", "grant", "revoke"]
    if any(re.search(rf"\b{op}\b", sql_lower) for op in dangerous_operations):
        return False, "La consulta contiene operaciones peligrosas no permitidas."

    # Comprobar si hace referencia al menos a una tabla esperada
    tablas_validas = [
        "games", "genres", "game_genres", "platforms", "game_platforms", "tags",
        "game_tags", "stores", "game_stores", "ratings", "game_ratings", 
        "esrb_ratings", "game_added_by_status", "parent_platforms", "game_parent_platforms"
    ]

    if not any(tabla in sql_lower for tabla in tablas_validas):
        return False, "La consulta no hace referencia a ninguna tabla conocida del esquema RAWG."

    return True, "Consulta válida"

# Código de prueba - solo se ejecuta si se llama directamente
if __name__ == "__main__":
    # Prueba aquí tus preguntas
    question = "what are the best 10 rated games?"
    generated_sql = question_to_sql(question)
    print("Pregunta:", question)
    print("\nSQL generada:\n", generated_sql)
    print(validar_sql_generada(generated_sql))