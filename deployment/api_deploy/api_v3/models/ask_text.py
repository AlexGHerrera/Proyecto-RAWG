#!/usr/bin/env python3
"""
ask_text.py - API v3
====================

Endpoint de consultas de texto usando modelo T5 fine-tuned.
Arquitectura unificada con sql_model_finetuned.py

Características v3:
- Modelo fine-tuned: cssupport/t5-small-awesome-text-to-sql
- Schema simplificado de RAWG
- Prompt nativo del modelo
- Validación y seguridad SQL
- Respuestas estructuradas

Autor: API v3 Team
Fecha: 2025-01-09
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from typing import Dict, Any, List
import time

# Importar el módulo unificado
from .sql_model_finetuned import question_to_sql_finetuned, get_model_info

# Configurar logging
logger = logging.getLogger(__name__)

# Crear router
router = APIRouter()

# ============================================================================
# MODELOS DE DATOS
# ============================================================================

class TextQuery(BaseModel):
    """Modelo para consultas de texto"""
    question: str

class TextResponse(BaseModel):
    """Modelo para respuestas de texto"""
    question: str
    sql: str
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    execution_time: float
    success: bool
    error: str = None

# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/ask-text", response_model=TextResponse)
async def ask_text_endpoint(query: TextQuery):
    """
    Endpoint principal para consultas de texto.
    
    Convierte pregunta en lenguaje natural a SQL usando modelo fine-tuned
    y retorna los resultados estructurados.
    
    Args:
        query: Objeto con la pregunta del usuario
        
    Returns:
        TextResponse: Respuesta estructurada con SQL, datos y metadatos
    """
    start_time = time.time()
    
    try:
        logger.info(f"Nueva consulta de texto: {query.question}")
        
        # Validar entrada
        if not query.question or not query.question.strip():
            raise HTTPException(
                status_code=400, 
                detail="La pregunta no puede estar vacía"
            )
        
        # Procesar con modelo unificado
        result = question_to_sql_finetuned(query.question)
        
        # Calcular tiempo de ejecución
        execution_time = time.time() - start_time
        
        # Verificar si hubo error
        if "error" in result:
            logger.error(f"Error procesando consulta: {result['error']}")
            return TextResponse(
                question=query.question,
                sql="",
                data=[],
                metadata=result.get("metadata", {}),
                execution_time=execution_time,
                success=False,
                error=result["error"]
            )
        
        # Respuesta exitosa
        logger.info(f"Consulta procesada exitosamente en {execution_time:.2f}s")
        
        return TextResponse(
            question=query.question,
            sql=result["sql"],
            data=result["data"],
            metadata=result["metadata"],
            execution_time=execution_time,
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Error inesperado en ask_text: {e}")
        
        return TextResponse(
            question=query.question,
            sql="",
            data=[],
            metadata={"error_type": "unexpected_error"},
            execution_time=execution_time,
            success=False,
            error=str(e)
        )


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def get_router():
    """
    Obtiene el router configurado para ask_text.
    
    Returns:
        APIRouter: Router con todos los endpoints
    """
    return router

# ============================================================================
# EJEMPLOS DE USO
# ============================================================================

"""
Ejemplos de consultas que maneja este endpoint:

1. Consulta básica:
   POST /ask-text
   {"question": "¿Cuáles son los mejores juegos de acción?"}

2. Consulta por plataforma:
   POST /ask-text  
   {"question": "¿Cuántos juegos hay en PC?"}

3. Consulta de agregación:
   POST /ask-text
   {"question": "¿Qué géneros tienen más juegos?"}

4. Consulta específica:
   POST /ask-text
   {"question": "Dame los datos de The Witcher 3"}

5. Consulta temporal:
   POST /ask-text
   {"question": "¿Cuáles son los mejores juegos desde 2020?"}

Respuesta típica:
{
  "question": "¿Cuáles son los mejores juegos de acción?",
  "sql": "SELECT g.name, g.rating FROM games g JOIN game_genres gg ON g.id_game = gg.id_game JOIN genres gen ON gen.id_genre = gg.id_genre WHERE gen.name ILIKE '%Action%' ORDER BY g.rating DESC LIMIT 10;",
  "data": [
    {"name": "Grand Theft Auto V", "rating": 4.47},
    {"name": "The Witcher 3: Wild Hunt", "rating": 4.66}
  ],
  "metadata": {
    "model": "cssupport/t5-small-awesome-text-to-sql",
    "rows_returned": 10,
    "columns": ["name", "rating"]
  },
  "execution_time": 2.34,
  "success": true
}
"""
