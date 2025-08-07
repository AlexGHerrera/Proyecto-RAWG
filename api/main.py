from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
from psycopg2 import pool
import pickle
import logging
import os
import numpy as np
from dotenv import load_dotenv
from contextlib import contextmanager
from typing import Optional
from models import predict
from models import ask_visual
from models import ask_text


#Cargar variables de entorno
load_dotenv()

#Configurar el logger para que los mensajes aparezcan en CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = FastAPI(title="API RAWG HAB")

# SINGLETON PARA MODELO ML
class MLModelSingleton:
    """Singleton para manejar la carga única del modelo ML"""
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLModelSingleton, cls).__new__(cls)
        return cls._instance
    
    def load_model(self) -> Optional[object]:
        """Carga el modelo una sola vez"""
        if self._model is None:
            try:
                model_path = os.getenv("MODEL_PATH", "model.pkl")
                with open(model_path, "br") as file:
                    self._model = pickle.load(file)
                    logger.info("Modelo ML cargado correctamente.")
            except FileNotFoundError:
                logger.warning(f"Modelo no encontrado en {model_path}. Funcionalidad de predicción deshabilitada.")
                self._model = None
            except Exception as e:
                logger.error(f"Error al cargar el modelo: {e}")
                self._model = None
        return self._model
    
    @property
    def model(self):
        return self.load_model()

# Instancia global del singleton
ml_model = MLModelSingleton()

# POOL DE CONEXIONES A LA BASE DE DATOS
class DatabasePool:
    """Singleton para manejar pool de conexiones a PostgreSQL"""
    _instance = None
    _pool = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabasePool, cls).__new__(cls)
        return cls._instance
    
    def initialize_pool(self):
        """Inicializa el pool de conexiones"""
        if self._pool is None:
            try:
                self._pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,
                    host=os.getenv("DB_HOST"),
                    port=os.getenv("DB_PORT"),
                    dbname=os.getenv("DB_NAME"),
                    user=os.getenv("DB_USER"),
                    password=os.getenv("DB_PASS")
                )
                logger.info("Pool de conexiones DB inicializado correctamente.")
            except Exception as e:
                logger.error(f"Error inicializando pool de conexiones: {e}")
                raise
    
    @contextmanager
    def get_connection(self):
        """Context manager para obtener conexión del pool"""
        if self._pool is None:
            self.initialize_pool()
        
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error en conexión DB: {e}")
            raise
        finally:
            if conn:
                self._pool.putconn(conn)
    
    def close_pool(self):
        """Cierra el pool de conexiones"""
        if self._pool:
            self._pool.closeall()
            logger.info("Pool de conexiones cerrado.")

# Instancia global del pool
db_pool = DatabasePool()

# EJECUCIÓN DE CONSULTAS SQL
def execute_sql(sql: str):
    """Ejecuta consulta SQL usando el pool de conexiones"""
    try:
        with db_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            # Obtener nombres de columnas
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            cursor.close()
            logger.info(f"Consulta ejecutada exitosamente: {sql[:100]}...")
            return result, columns
    except Exception as e:
        logger.error(f"Error ejecutando SQL: {sql[:100]}... | Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error en consulta SQL: {str(e)}")

def execute_sql_to_dataframe(sql: str) -> pd.DataFrame:
    """Ejecuta consulta SQL y retorna DataFrame"""
    try:
        with db_pool.get_connection() as conn:
            df = pd.read_sql_query(sql, conn)
            logger.info(f"DataFrame creado con {len(df)} filas: {sql[:100]}...")
            return df
    except Exception as e:
        logger.error(f"Error creando DataFrame: {sql[:100]}... | Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error en consulta SQL: {str(e)}")

#ESQUEMAS

class Data(BaseModel):
    # Features de diseño según el proyecto RAWG
    n_genres: int
    n_platforms: int
    n_tags: int
    esrb_rating_id: int
    estimated_hours: float
    planned_year: int

class PredictRequest(BaseModel):
    features: Data

class AskTextRequest(BaseModel):
    question: str

class AskVisualRequest(BaseModel):
    question: str

#ENDPOINTS

@app.get("/")
def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "API RAWG HAB - Predicción y análisis de videojuegos",
        "version": "2.0",
        "endpoints": {
            "/predict": "POST - Predicción de éxito de videojuegos",
            "/ask-text": "POST - Consultas en lenguaje natural",
            "/ask-visual": "POST - Consultas con visualización",
            "/health": "GET - Estado de la API",
            "/docs": "GET - Documentación automática"
        },
        "status": "active"
    }

@app.get("/health")
def health_check():
    """Health check endpoint para monitoreo"""
    try:
        # Verificar conexión a DB
        with db_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            db_status = "healthy"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    # Verificar modelo SQL
    try:
        sql_model.tokenizer  # Acceder para inicializar si es necesario
        sql_model_status = "loaded"
    except Exception as e:
        sql_model_status = f"error: {str(e)}"
    
    # Verificar modelo ML
    ml_model_status = "loaded" if ml_model.model is not None else "not_available"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "timestamp": pd.Timestamp.now().isoformat(),
        "services": {
            "database": db_status,
            "sql_model": sql_model_status,
            "ml_model": ml_model_status
        }
    }

@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    logger.info("Endpoint /predict llamado")
    model = ml_model.model
    if model is None:
        return {"error": "El modelo de predicción no está disponible. Funcionalidad en desarrollo."}
    
    try:
        # Convertir las features de diseño a array numpy
        features = request.features
        input_data = np.array([[
            features.n_genres,
            features.n_platforms, 
            features.n_tags,
            features.esrb_rating_id,
            features.estimated_hours,
            features.planned_year
        ]])
        
        prediction = model.predict(input_data)[0]
        return {
            "input_data": request.dict(),
            "prediction": float(prediction),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        return {"error": f"Error en la predicción: {str(e)}", "status": "error"}

@app.post("/ask-text")
def ask_text_endpoint(request: AskTextRequest):
    logger.info("Endpoint /ask-text llamado")
    user_question = request.question
    
    try:
        # Generar SQL usando el modelo transformer
        generated_sql = ask_text.question_to_sql(user_question, visual_mode=False)
        is_valid, validation_message = ask_text.validar_sql_generada(generated_sql)
        
        logger.info(f"Pregunta: {user_question}")
        logger.info(f"SQL generada: {generated_sql}")
        logger.info(f"Validación: {validation_message}")
        
        if not is_valid:
            return {
                "question": user_question,
                "sql": generated_sql,
                "error": f"SQL inválida: {validation_message}",
                "status": "validation_error"
            }
        
        # Ejecutar consulta SQL
        result, columns = execute_sql(generated_sql)
        return {
            "question": user_question,
            "sql": generated_sql,
            "result": result,
            "columns": columns,
            "row_count": len(result),
            "status": "success"
        }
    except HTTPException:
        raise  # Re-lanzar HTTPException
    except Exception as e:
        logger.error(f"Error en ask-text: {str(e)}")
        return {
            "question": user_question,
            "error": f"Error procesando la pregunta: {str(e)}",
            "status": "error"
        }

@app.post("/ask-visual")
def ask_visual_endpoint(request: AskVisualRequest):
    logger.info("Endpoint /ask-visual llamado")
    user_question = request.question
    
    try:
        # Generar SQL usando el modelo transformer en modo visual
        generated_sql = ask_text.question_to_sql(user_question, visual_mode=True)
        is_valid, validation_message = ask_text.validar_sql_generada(generated_sql)
        
        logger.info(f"Pregunta: {user_question}")
        logger.info(f"SQL generada: {generated_sql}")
        logger.info(f"Validación: {validation_message}")
        
        if not is_valid:
            return {
                "question": user_question,
                "sql": generated_sql,
                "error": f"SQL inválida: {validation_message}",
                "status": "validation_error"
            }
        
        # Ejecutar consulta y crear DataFrame
        df = execute_sql_to_dataframe(generated_sql)
        
        if df.empty:
            return {
                "question": user_question,
                "sql": generated_sql,
                "error": "La consulta no retornó datos para visualizar",
                "status": "no_data"
            }
        
        # Generar visualización
        fig = ask_visual.auto_viz(df)
        if fig is None:
            return {
                "question": user_question,
                "sql": generated_sql,
                "data_preview": df.head().to_dict('records'),
                "data_shape": df.shape,
                "error": "No se pudo generar visualización para este tipo de datos",
                "status": "visualization_error"
            }
        
        # Convertir figura a imagen base64
        import plotly.io as pio
        import base64
        from io import BytesIO
        
        # Generar imagen PNG
        img_bytes = pio.to_image(fig, format='png', width=800, height=600)
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return {
            "question": user_question,
            "sql": generated_sql,
            "data_shape": df.shape,
            "visualization": {
                "image_base64": img_base64,
                "format": "png",
                "width": 800,
                "height": 600
            },
            "data_preview": df.head(5).to_dict('records'),
            "status": "success"
        }
        
    except HTTPException:
        raise  # Re-lanzar HTTPException
    except Exception as e:
        logger.error(f"Error en ask-visual: {str(e)}")
        return {
            "question": user_question,
            "error": f"Error procesando la pregunta visual: {str(e)}",
            "status": "error"
        }




