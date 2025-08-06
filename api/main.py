from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import pickle
import logging
import os
import numpy as np
from dotenv import load_dotenv
from models import predict
from models import ask_visual
from models import ask_text

#Cargar variables de entorno
load_dotenv()

#Configurar el logger para que los mensajes aparezcan en CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = FastAPI(title="API RAWG HAB")

#MODELO ML
def load_model():
    try:
        with open("model.pkl", "br") as file:
            logger.info("Modelo cargado correctamente.")
            return pickle.load(file)
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        raise

#CONEXIÓN A LA BASE DE DATOS
def get_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS")
        )
        logger.info("Conexión a la base de datos exitosa.")
        return conn
    except Exception as e:
        logger.error(f"Error de conexión a la base de datos: {e}")
        raise

#CONSULTA A SQL
def execute_sql(sql: str):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.close()
        logger.info(f"Consulta ejecutada: {sql}")
        return result
    except Exception as e:
        logger.error(f"Error ejecutando SQL: {sql} | Error: {e}")
        raise

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

@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    logger.info("Endpoint /predict llamado")
    model = load_model()
    if model is None:
        return {"error": "El modelo no está disponible."}
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
    except Exception as e:
        return {"error": f"Error procesando las features: {str(e)}"}

    try:
        prediction = model.predict(input_data)[0]
        return {
            "input_data": request.dict(),
            "prediction": float(prediction)
        }
    except Exception as e:
        return {"error": f"Error en la predicción: {str(e)}"}

@app.post("/ask-text")
def ask_text_endpoint(request: AskTextRequest):
    logger.info("Endpoint /ask-text llamado")
    user_question = request.question
    
    try:
        generated_sql = ask_text.question_to_sql(user_question)
        validation_message = ask_text.validar_sql_generada(generated_sql)
        
        logger.info(f"Pregunta: {user_question}")
        logger.info(f"SQL generada: {generated_sql}")
        
        result = execute_sql(generated_sql)
        return {
            "question": user_question,
            "sql": generated_sql,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error en ask-text: {str(e)}")
        return {"error": f"Error procesando la pregunta: {str(e)}"}

@app.post("/ask-visual")
def ask_visual_endpoint(request: AskVisualRequest):
    logger.info("Endpoint /ask-visual llamado")
    user_question = request.question
    
    try:
        generated_sql = ask_text.question_to_sql(user_question)
        is_valid, validation_message = ask_text.validar_sql_generada(generated_sql)
        
        if not is_valid:
            return {"error": f"SQL inválida: {validation_message}"}
        
        logger.info(f"Pregunta: {user_question}")
        logger.info(f"SQL generada: {generated_sql}")
        
        df = pd.read_sql_query(generated_sql, get_connection())
        
        """
        Convierte la pregunta en visualización y retorna un gráfico como imagen.
        Aquí se insertará la lógica de visualización.
        """
        # === Aquí insertas el código que genera el gráfico ===
        # Ejemplo futuro:
        # image_path = ask_visual.generate_visual_from_question(request.question, df)
        # return FileResponse(image_path, media_type="image/png")
        
        return {
            "message": "Pregunta visual recibida. Falta implementar lógica.",
            "question": user_question,
            "sql": generated_sql,
            "data_shape": df.shape
        }
    except Exception as e:
        logger.error(f"Error en ask-visual: {str(e)}")
        return {"error": f"Error procesando la pregunta visual: {str(e)}"}

