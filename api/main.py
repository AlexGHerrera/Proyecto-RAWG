from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import pickle
import logging
import os
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

model = load_model()

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
    rating: float
    playtime: str
    # Agregar más campos si es necesario

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
    """
    Recibe características de un videojuego y devuelve la predicción del modelo.
    Aquí se insertará la lógica del modelo ML.
    """
    # === Aquí insertas el código del modelo ML ===
    # Ejemplo futuro:
    # result = predict_game(request.features)
    return {"message": "Predicción recibida. Falta implementar lógica."}

@app.post("/ask-text")
def ask_text_endpoint(request: AskTextRequest):
    logger.info("Endpoint /ask-text llamado")
    user_question = request.question
    generated_sql = question_to_sql(user_question)
    validar_sql_generada(generated_sql)
    print("Pregunta:", user_question)
    print("\nSQL generada:\n", generated_sql)

    get_connection()
    return execute_sql(generated_sql: str)

@app.post("/ask-visual")
def ask_visual_endpoint(request: AskVisualRequest):
    logger.info("Endpoint /ask-visual llamado")
    generated_sql = question_to_sql(user_question)
    validar_sql_generada(generated_sql)
    print("Pregunta:", user_question)
    print("\nSQL generada:\n", generated_sql)

    df = pd.read_sql_query(generated_sql, get_connection())
    
    """
    Convierte la pregunta en visualización y retorna un gráfico como imagen.
    Aquí se insertará la lógica de visualización.
    """
    # === Aquí insertas el código que genera el gráfico ===
    # Ejemplo futuro:
    # image_path = generate_visual_from_question(request.question)
    # return FileResponse(image_path, media_type="image/png")
    return {"message": "Pregunta visual recibida. Falta implementar lógica."}

