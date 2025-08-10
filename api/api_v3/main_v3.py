#!/usr/bin/env python3
"""
main_v3.py - API v3
==================

FastAPI principal para el sistema NL→SQL con T5 fine-tuned.
Arquitectura unificada: NL→SQL directo usando modelo fine-tuned.

Endpoints:
- /ask-text: Consultas de texto → SQL + resultados
- /ask-visual: Consultas de texto → SQL + resultados + visualización

Características v3:
- Modelo fine-tuned: cssupport/t5-small-awesome-text-to-sql
- Schema simplificado de RAWG
- Código compartido entre endpoints
- Validación y seguridad mejoradas

Autor: API v3 Team
Fecha: 2025-01-09
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import plotly.graph_objects as go
import time
import json
import numpy as np

# Importar módulos internos v3
from .models.sql_model_finetuned import question_to_sql_finetuned, get_model_info, test_model_connection, execute_sql_query
from .models.ask_visual import auto_viz
from .models import ask_text
from .models import utility_endpoints  # Importar el módulo de utilidades

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN DE LA APP
# ============================================================================

app = FastAPI(
    title="API RAWG HAB v3",
    description="Sistema NL→SQL con modelo T5 fine-tuned para base de datos RAWG",
    version="3.0.0",
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

# Incluir los routers modulares
app.include_router(ask_text.router, tags=["Text Queries"])
app.include_router(utility_endpoints.router, tags=["Utility"])


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
    logger.info("Iniciando API v3...")
    
    try:
        # Verificar información del modelo
        model_info = get_model_info()
        logger.info(f"Modelo: {model_info['model_name']}")
        logger.info(f"Transformers disponible: {model_info['transformers_available']}")
        
        # Nota: El modelo se carga de forma lazy en la primera consulta
        logger.info("API v3 iniciada exitosamente")
        
    except Exception as e:
        logger.error(f"Error en startup: {e}")
        # No fallar el startup, solo advertir

# ============================================================================
# ENDPOINTS PRINCIPALES (Visual y otros)
# ============================================================================

@app.post("/ask-visual", response_model=VisualResponse)
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

# ============================================================================
# ENDPOINTS DE UTILIDAD
# ============================================================================



@app.get("/test-model")
async def test_model_endpoint():
    """
    Endpoint para probar la conexión con el modelo fine-tuned.
    """
    try:
        logger.info("Probando conexión con modelo...")
        
        # Probar conexión
        success = test_model_connection()
        
        if success:
            return {
                "success": True,
                "message": "Modelo fine-tuned funcionando correctamente",
                "api_version": "v3"
            }
        else:
            return {
                "success": False,
                "message": "Error conectando con el modelo",
                "api_version": "v3"
            }
            
    except Exception as e:
        logger.error(f"Error probando modelo: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error probando modelo: {str(e)}"
        )

@app.post("/test-fast-query")
async def test_fast_query():
    """
    Endpoint de prueba rápida con consulta SQL optimizada para DB lenta.
    """
    try:
        logger.info("Probando consulta SQL optimizada...")
        
        # Consulta SQL simple y rápida
        simple_sql = "SELECT name, rating FROM v_games ORDER BY rating DESC LIMIT 5"
        
        start_time = time.time()
        data = execute_sql_query(simple_sql, timeout=60)
        execution_time = time.time() - start_time
        
        return {
            "success": True,
            "message": "Consulta rápida ejecutada exitosamente",
            "sql": simple_sql,
            "execution_time": execution_time,
            "rows_returned": len(data),
            "data": data[:3],  # Solo mostrar primeros 3 resultados
            "api_version": "v3"
        }
        
    except Exception as e:
        logger.error(f"Error en consulta rápida: {e}")
        return {
            "success": False,
            "message": f"Error en consulta rápida: {str(e)}",
            "api_version": "v3"
        }

@app.get("/")
async def root():
    """
    Endpoint raíz con información básica de la API.
    """
    return {
        "message": "API RAWG HAB v3 - Sistema NL→SQL con T5 fine-tuned",
        "version": "3.0.0",
        "model": "cssupport/t5-small-awesome-text-to-sql",
        "endpoints": {
            "ask_text": "/ask-text",
            "ask_visual": "/ask-visual",
            "health": "/health",
            "test_model": "/test-model",
            "docs": "/docs"
        },
        "architecture": "unified_finetuned",
        "features": [
            "Modelo T5 fine-tuned para text-to-SQL",
            "Schema simplificado de RAWG",
            "Validación y seguridad SQL",
            "Visualizaciones automáticas",
            "Código compartido entre endpoints"
        ]
    }

# ============================================================================
# MAIN - PARA DESARROLLO
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Iniciando servidor de desarrollo...")
    uvicorn.run(
        "api.api_v3.main_v3:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
