from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from typing import List, Dict, Optional
from models import predict
from models import ask_text_lite
from models import ask_visual_enhanced
import pandas as pd
import time

# Configurar el logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = FastAPI(title="API RAWG HAB - Simple")

# ESQUEMAS
class GameInput(BaseModel):
    genres: List[str]
    platforms: List[str]
    tags: List[str]
    estimated_hours: float
    release_year: int = 2024

class PredictRequest(BaseModel):
    game: GameInput

class TextQueryRequest(BaseModel):
    question: str

class VisualQueryRequest(BaseModel):
    question: str
    sql_query: Optional[str] = None

# ENDPOINTS
@app.get("/")
def root():
    return {"message": "API RAWG HAB - Modelo v3 funcionando"}

@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    logger.info("Endpoint /predict llamado")
    
    try:
        # Preparar datos para el modelo v3
        input_data = {
            'genres': request.game.genres,
            'platforms': request.game.platforms,
            'tags': request.game.tags,
            'estimated_hours': request.game.estimated_hours,
            'release_year': request.game.release_year
        }
        
        # Realizar predicción
        result = predict.predict(input_data)
        
        return {
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "input_summary": {
                "genres": request.game.genres,
                "platforms": request.game.platforms,
                "tags": request.game.tags,
                "estimated_hours": request.game.estimated_hours,
                "release_year": request.game.release_year
            }
        }
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return {"error": f"Error en la predicción: {str(e)}"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "v3_random_forest"}

# NUEVOS ENDPOINTS OPTIMIZADOS

@app.post("/ask-text")
def ask_text_endpoint(request: TextQueryRequest):
    """
    Convierte preguntas en lenguaje natural a consultas SQL
    """
    logger.info(f"Endpoint /ask-text-lite llamado: {request.question}")
    
    start_time = time.time()
    
    try:
        # Generar SQL con modelo ligero
        sql_query = ask_text_lite.question_to_sql_lite(request.question)
        
        # Validar SQL
        is_valid, validation_message = ask_text_lite.validate_sql_lite(sql_query)
        
        latency = time.time() - start_time
        
        # Detectar tablas relevantes para ayudar al usuario
        relevant_tables = ask_visual_enhanced.detect_relevant_tables(request.question)
        suggested_joins = ask_visual_enhanced.suggest_table_joins(relevant_tables)
        
        return {
            "sql_query": sql_query,
            "is_valid": is_valid,
            "validation_message": validation_message,
            "detected_tables": relevant_tables,
            "suggested_joins": suggested_joins
        }
        
    except Exception as e:
        logger.error(f"Error en ask-text-lite: {e}")
        return {
            "error": f"Error generando SQL: {str(e)}",
            "sql_query": "",
            "is_valid": False,
            "latency_ms": round((time.time() - start_time) * 1000, 2)
        }

@app.post("/ask-visual")
def ask_visual_endpoint(request: VisualQueryRequest):
    """
    Genera visualizaciones automáticas basadas en consultas de datos
    """
    logger.info(f"Endpoint /ask-visual-enhanced llamado: {request.question}")
    
    start_time = time.time()
    
    try:
        # Si no se proporciona SQL, generar con modelo lite
        if not request.sql_query:
            sql_query = ask_text_lite.question_to_sql_lite(request.question)
            is_valid, _ = ask_text_lite.validate_sql_lite(sql_query)
            
            if not is_valid:
                return {
                    "error": "No se pudo generar SQL válida para la visualización",
                    "suggestions": [
                        "Reformular la pregunta",
                        "Especificar tablas más claramente",
                        "Proporcionar SQL manualmente"
                    ]
                }
        else:
            sql_query = request.sql_query
        
        # Simular ejecución de SQL (en producción conectar a BD)
        # Por ahora, crear datos de ejemplo basados en la consulta
        sample_data = create_sample_data_from_query(request.question, sql_query)
        
        # Generar análisis visual mejorado
        analysis = ask_visual_enhanced.enhanced_auto_viz(sample_data, request.question)
        
        latency = time.time() - start_time
        
        return {
            "sql_query": sql_query,
            "viz_type": analysis["viz_type"],
            "intent_description": analysis["intent_description"],
            "detected_tables": analysis["detected_tables"],
            "suggested_joins": analysis["suggested_joins"],
            "recommendations": analysis["recommendations"],
            "figure_available": analysis["figure"] is not None
        }
        
    except Exception as e:
        logger.error(f"Error en ask-visual-enhanced: {e}")
        return {
            "error": f"Error en visualización: {str(e)}",
            "latency_ms": round((time.time() - start_time) * 1000, 2)
        }


def create_sample_data_from_query(question: str, sql_query: str) -> pd.DataFrame:
    """
    Crea datos de ejemplo basados en la consulta para demostración
    En producción, esto ejecutaría la SQL real contra la base de datos
    """
    question_lower = question.lower()
    
    # Datos de ejemplo basados en patrones comunes
    if "género" in question_lower or "genre" in question_lower:
        return pd.DataFrame({
            'genre': ['Action', 'RPG', 'Strategy', 'Adventure', 'Sports'],
            'avg_playtime': [25.5, 45.2, 30.1, 20.8, 15.3],
            'game_count': [150, 89, 67, 123, 45],
            'avg_rating': [4.2, 4.5, 4.0, 4.1, 3.8]
        })
    
    elif "plataforma" in question_lower or "platform" in question_lower:
        return pd.DataFrame({
            'platform': ['PC', 'PlayStation 4', 'Xbox One', 'Nintendo Switch'],
            'game_count': [1250, 890, 750, 650],
            'avg_rating': [4.3, 4.2, 4.1, 4.4]
        })
    
    elif "top" in question_lower or "mejor" in question_lower:
        return pd.DataFrame({
            'game_name': ['Game A', 'Game B', 'Game C', 'Game D', 'Game E'],
            'rating': [4.8, 4.7, 4.6, 4.5, 4.4],
            'playtime': [35, 42, 28, 31, 25]
        })
    
    else:
        # Datos genéricos
        return pd.DataFrame({
            'category': ['Cat A', 'Cat B', 'Cat C'],
            'value': [100, 150, 120],
            'count': [25, 30, 28]
        })
