import os
import psycopg2
import logging
import boto3
import json
from botocore.exceptions import ClientError
import requests
from rawg.rawg_api import fetch_new_games
from dotenv import load_dotenv # Quitar en Lambda
from io import BytesIO
from dateutil import parser

load_dotenv() # Quitar en Lambda
logger = logging.getLogger(__name__)
RAWG_API_KEY = os.getenv('RAWG_API_KEY')
logger.setLevel(logging.INFO)
s3_client = boto3.client('s3')
S3_BUCKET = os.getenv('S3_BUCKET')
S3_PREFIX = os.getenv('S3_PREFIX')

# Configure logger handler and formatter
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Validate RAWG API key
if not RAWG_API_KEY:
    logger.error("Environment variable RAWG_API_KEY is not set")
    raise RuntimeError("Missing RAWG_API_KEY")


def get_RDS_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASS'),
            port=os.getenv('DB_PORT', 5432)
        )
        conn.autocommit = False
        return conn
    except psycopg2.OperationalError:
        logger.exception("DB connection failed")
        raise

def get_last_updated(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COALESCE(MAX(updated), '1970-01-01T00:00:00Z'::timestamptz) FROM games;"
        )
        return cur.fetchone()[0] or 0


def fetch_new_games(last_updated, page_size=40):
    """
    Obtiene de RAWG todos los juegos cuya fecha 'updated' sea posterior a last_updated.
    
    :param last_updated: datetime.datetime, límite inferior (estricto) de actualización.
    :param page_size: int, número de items por página.
    :return: list de dicts con los juegos nuevos.
    """
    page = 1
    new_games = []
    
    while True:
        try:
            resp = requests.get(
                'https://api.rawg.io/api/games',
                params={
                    'key': RAWG_API_KEY,
                    'ordering': '-updated',
                    'page_size': page_size,
                    'page': page
                },
                timeout=10
            )
            resp.raise_for_status()
            logger.info(f"RAWG API: página {page} obtenida correctamente")
        except requests.RequestException as e:
            logger.exception("Error al conectar con RAWG API")
            break

        data = resp.json()
        results = data.get('results', [])
        
        # Si no hay resultados, salimos
        if not results:
            logger.info("RAWG API: sin resultados en esta página, terminando.")
            break

        # Recorremos y solo añadimos los que realmente son más recientes
        for g in results:
            updated_dt = parser.isoparse(g['updated'])
            if updated_dt > last_updated:
                new_games.append(g)
            else:                
                # En cuanto encontramos uno viejo, no hay más nuevos en esta
                # ni en siguientes páginas ya que vienen ordenados de más reciente a menos.
                logger.info("Encontrado juego ≤ last_updated; terminando búsqueda.")
                return new_games

        # Si no hay siguiente página, terminamos
        if not data.get('next'):
            logger.info("RAWG API: no hay más páginas, terminando.")
            break

        page += 1

    return new_games


def upload_json_to_s3(new_games):
    """Genera directamente el JSON y lo sube a S3."""
    buf = BytesIO()
    for g in new_games:
        buf.write(json.dumps(g, ensure_ascii=False).encode('utf-8'))
        buf.write(b'\n')
    buf.seek(0)
    key = os.path.join(S3_PREFIX, f"updated {len(new_games)} games.json")
    try:
        s3_client.upload_fileobj(buf, S3_BUCKET, key)
        logger.info(f"Upload successful: s3://{S3_BUCKET}/{key}")
    except ClientError as e:
        logger.error(f"Error uploading to S3: {e}")
        raise 


def lambda_handler(event, context):
    logger.info("Lambda handler invoked with event: %s", event)
    conn = None
    try:
        # Conexión a RDS
        try:
            conn = get_RDS_connection()
            logger.info("Conexión a RDS establecida")
        except Exception:
            logger.error("Error al conectar a RDS", exc_info=True)
            raise

        # Obtener última fecha de actualización
        try:
            last = get_last_updated(conn)
            logger.info("Última actualización en DB: %s", last)
        except Exception:
            logger.error("Error al obtener last_updated de la DB", exc_info=True)
            raise

        # Obtener juegos nuevos
        try:
            new_games = fetch_new_games(last)
            logger.info("RAWG API devolvió %d juegos nuevos", len(new_games))
        except Exception:
            logger.error("Error al llamar a fetch_new_games", exc_info=True)
            raise

        if not new_games:
            logger.info("No hay juegos nuevos para procesar")
            return {'status': 'ok', 'new_count': 0}

        # Subir JSON a S3
        try:
            upload_json_to_s3(new_games=new_games)
            logger.info("Se han subido %d juegos nuevos a S3", len(new_games))
        except Exception:
            logger.error("Error al subir JSON a S3", exc_info=True)
            raise

        return {'status': 'ok', 'new_count': len(new_games)}

    except Exception:
        # Cualquier fallo cae aquí
        logger.error("La función Lambda terminó con error", exc_info=True)
        # Opcional: reenviar excepción para que Lambda marque fallo
        raise

    finally:
        if conn:
            try:
                conn.close()
                logger.info("Conexión a RDS cerrada")
            except Exception:
                logger.warning("Error cerrando la conexión a RDS", exc_info=True)