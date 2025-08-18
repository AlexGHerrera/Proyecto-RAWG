#!/usr/bin/env python3
"""
main_v5.py - RAWG API v5
========================

FastAPI principal para el sistema NL→SQL con SQLCoder-7B-2 y visualización automática.
Arquitectura optimizada con esquema PostgreSQL normalizado.

Endpoints:
- /ask-text: Consultas de texto → SQL + resultados JSON
- /ask-visual: Consultas de texto → SQL + resultados + visualización JSON
- /predict: Predicción de éxito de videojuegos
- /health: Health check del sistema
- /model/info: Información del modelo SQLCoder

Características v5:
- Modelo SQLCoder-7B-2 via HuggingFace Inference API
- Esquema PostgreSQL normalizado con JOINs complejos
- Soporte bilingüe (inglés y español)
- Corrección inteligente de SQL generado
- Visualización automática con Plotly
- Sistema de fallbacks robusto

Autor: RAWG API v5 Team
Fecha: 2025-08-08
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import plotly.graph_objects as go
import time
import json
import numpy as np

# Importar módulos internos v5
# Importaciones con fallback para compatibilidad local y EC2
try:
    # Intentar importaciones relativas (funciona cuando se ejecuta como paquete)
    from .models.sql_model_finetuned import question_to_sql_finetuned, get_model_info, test_model_connection, execute_sql_query
    from .models.ask_visual import auto_viz
    from .models.predict import predict
    from .models import ask_text
    from .models import utility_endpoints
except ImportError:
    # Fallback a importaciones absolutas (funciona cuando se ejecuta directamente)
    from models.sql_model_finetuned import question_to_sql_finetuned, get_model_info, test_model_connection, execute_sql_query
    from models.ask_visual import auto_viz
    from models.predict import predict
    from models import ask_text
    from models import utility_endpoints

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN DE LA APP
# ============================================================================

app = FastAPI(
    title="RAWG API v5",
    description="Sistema NL→SQL con SQLCoder-7B-2, esquema PostgreSQL normalizado y visualización automática",
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir los routers modulares con organización mejorada
app.include_router(ask_text.router, tags=["Natural Language Queries"])
app.include_router(utility_endpoints.router, tags=["System Utilities"])


# ============================================================================
# MODELOS DE DATOS (Solo para endpoints definidos aquí)
# ============================================================================

class VisualQuery(BaseModel):
    """Modelo para consultas de visualización"""
    question: str

class VisualResponse(BaseModel):
    """Modelo para respuestas de visualización"""
    question: str
    sql: str
    data: List[Dict[str, Any]]
    visualization: Dict[str, Any]  # Plotly figure as dict
    metadata: Dict[str, Any]
    execution_time: float
    success: bool
    error: Optional[str] = None

class PredictionRequest(BaseModel):
    """Modelo para solicitudes de predicción"""
    genres: List[str]
    platforms: List[str]
    tags: List[str]
    estimated_hours: float
    release_year: int

class PredictionResponse(BaseModel):
    """Modelo para respuestas de predicción"""
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    execution_time: float
    success: bool
    error: Optional[str] = None

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def convert_numpy_types(obj):
    """
    Convierte tipos numpy a tipos nativos de Python para serialización JSON.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# ============================================================================
# EVENTOS DE STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Evento de inicio de la aplicación.
    Verifica que todos los componentes estén disponibles.
    """
    logger.info("Iniciando API v5...")
    
    try:
        # Verificar información del modelo
        model_info = get_model_info()
        logger.info(f"Modelo: {model_info['model_name']}")
        logger.info(f"Transformers disponible: {model_info['transformers_available']}")
        
        # El modelo se carga automáticamente en la primera consulta
        logger.info("API v5 iniciada exitosamente")
        
    except Exception as e:
        logger.error(f"Error en startup: {e}")
        # No fallar el startup, solo advertir

# ============================================================================
# ENDPOINTS PRINCIPALES (Visual y otros)
# ============================================================================

@app.post("/ask-visual", response_model=VisualResponse, tags=["Data Visualization"])
async def ask_visual_endpoint(query: VisualQuery):
    """
    Endpoint para consultas con visualización.
    
    Convierte pregunta en lenguaje natural a SQL, ejecuta la consulta
    y genera una visualización automática de los resultados.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Nueva consulta visual: {query.question}")
        
        # Validar entrada
        if not query.question or not query.question.strip():
            raise HTTPException(
                status_code=400, 
                detail="La pregunta no puede estar vacía"
            )
        
        # Procesar con modelo unificado (mismo que ask_text)
        result = question_to_sql_finetuned(query.question)
        
        # Verificar si hubo error en la consulta SQL
        if "error" in result:
            execution_time = time.time() - start_time
            logger.error(f"Error procesando consulta: {result['error']}")
            return VisualResponse(
                question=query.question,
                sql="",
                data=[],
                visualization={},
                metadata=result.get("metadata", {}),
                execution_time=execution_time,
                success=False,
                error=result["error"]
            )
        
        # Generar visualización
        try:
            df = pd.DataFrame(result["data"])
            
            if df.empty:
                # Sin datos para visualizar
                fig_dict = {"data": [], "layout": {"title": "Sin datos para mostrar"}}
            else:
                # Generar visualización automática
                fig = auto_viz(df, query.question)
                if fig:
                    fig_dict = fig.to_dict()
                    # Convertir tipos numpy para serialización JSON
                    fig_dict = convert_numpy_types(fig_dict)
                else:
                    fig_dict = {"data": [], "layout": {"title": "Error generando visualización"}}
            
        except Exception as viz_error:
            logger.warning(f"Error generando visualización: {viz_error}")
            fig_dict = {"data": [], "layout": {"title": "Error en visualización", "annotations": [{"text": str(viz_error)}]}}
        
        # Calcular tiempo de ejecución
        execution_time = time.time() - start_time
        
        # Respuesta exitosa
        logger.info(f"Consulta visual procesada exitosamente en {execution_time:.2f}s")
        
        return VisualResponse(
            question=query.question,
            sql=result["sql"],
            data=convert_numpy_types(result["data"]),
            visualization=fig_dict,
            metadata=convert_numpy_types(result["metadata"]),
            execution_time=execution_time,
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Error inesperado en ask_visual: {e}")
        
        return VisualResponse(
            question=query.question,
            sql="",
            data=[],
            visualization={},
            metadata={"error_type": "unexpected_error"},
            execution_time=execution_time,
            success=False,
            error=str(e)
        )



@app.post("/predict", response_model=PredictionResponse, tags=["Machine Learning"])
async def predict_endpoint(request: PredictionRequest):
    """
    Endpoint para predicción de éxito de videojuegos.
    
    Utiliza el modelo v3 entrenado para predecir el éxito de un videojuego
    basándose en sus características de diseño.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Nueva predicción: {len(request.genres)} géneros, {len(request.platforms)} plataformas")
        
        # Preparar datos de entrada
        input_data = {
            "genres": request.genres,
            "platforms": request.platforms,
            "tags": request.tags,
            "estimated_hours": request.estimated_hours,
            "release_year": request.release_year
        }
        
        # Realizar predicción
        result = predict(input_data)
        
        # Calcular tiempo de ejecución
        execution_time = time.time() - start_time
        
        logger.info(f"Predicción completada: {result['predicted_class']} (confianza: {result['confidence']:.2f})")
        
        return PredictionResponse(
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            execution_time=execution_time,
            success=True
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Error en predicción: {e}")
        
        return PredictionResponse(
            predicted_class="",
            confidence=0.0,
            probabilities={},
            execution_time=execution_time,
            success=False,
            error=str(e)
        )

# ============================================================================
# ENDPOINTS DE UTILIDAD
# ============================================================================




@app.get("/", tags=["API Information"])
async def root():
    """
    Endpoint raíz con información básica de la API.
    """
    return {
        "message": "RAWG API v5 - Natural Language to SQL with Unified Schema",
        "version": "5.0.0",
        "status": "Production Ready",
        "model": "defog/sqlcoder-7b-2",
        "description": "Convert natural language questions about video games into SQL queries and visualizations",
        "endpoints": {
            "ask_text": "/ask-text - Natural language text queries",
            "ask_visual": "/ask-visual - Queries with automatic visualizations (JSON)",
            "predict": "/predict - Video game success predictions",
            "health": "/health - API health check",
            "model_info": "/model/info - Model information",
            "docs": "/docs - Interactive API documentation"
        },
        "features": [
            "SQLCoder-7B-2 model via HuggingFace Inference API",
            "PostgreSQL normalized schema with complex JOINs",
            "Bilingual support (English and Spanish)",
            "Intelligent SQL correction and validation",
            "Automatic data visualizations with Plotly",
            "Language detection and validation",
            "Gaming-focused query filtering",
            "SQL security validation and sanitization",
            "RAWG database integration with 500k+ games"
        ],
        "supported_queries": [
            "Show me the top 5 highest rated games",
            "RPG games with rating above 4.0",
            "Games available on PlayStation",
            "Count of games by genre",
            "Average rating by platform",
            "Games released between 2020 and 2023",
            "Muéstrame juegos de RPG con valoración superior a 4.0",
            "Juegos disponibles en PlayStation"
        ],
        "usage": {
            "documentation": "/docs",
            "example_request": {
                "endpoint": "/ask-text",
                "method": "POST",
                "body": {"question": "games by platform"}
            }
        }
    }

# ============================================================================
# MAIN - PARA DESARROLLO
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Iniciando servidor de desarrollo...")
    uvicorn.run(
        "api.api_v5.main_v5:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
