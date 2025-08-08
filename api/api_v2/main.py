from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
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
from io import BytesIO
import plotly.io as pio
import kaleido
from typing import List


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

class GameInput(BaseModel):
    genres: List[str]
    platforms: List[str]
    tags: List[str]
    estimated_hours: float
    release_year: int = 2024

class PredictRequest(BaseModel):
    game: GameInput

class AskTextRequest(BaseModel):
    question: str

class AskVisualRequest(BaseModel):
    question: str

#ENDPOINTS

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
            "probabilities": result["probabilities"]
        }
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
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
async def ask_visual_endpoint(request: AskVisualRequest):
    logger.info("Endpoint /ask-visual llamado")
    user_question = request.question

    generated_sql = ask_text.question_to_sql(user_question)
    is_valid, validation_message = ask_text.validar_sql_generada(generated_sql)
    if not is_valid:
        return JSONResponse(content={"error": f"SQL inválida: {validation_message}"}, status_code=400)

    df = pd.read_sql_query(generated_sql, get_connection())
    fig = ask_visual.auto_viz(df, user_question)

    if fig is None:
        return JSONResponse(
            content={"message": "No se pudo generar gráfico", "data_preview": df.head().to_dict(orient="records")},
            status_code=200
        )

    # Generar la imagen en memoria
    buf = BytesIO()
    pio.write_image(fig, buf, format="png")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")