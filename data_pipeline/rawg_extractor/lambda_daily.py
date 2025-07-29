
import logging
from processing import get_conn, get_last_updated, fetch_new_games, upload_json_batches_to_s3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    logger.info("Lambda invocada")

    try:
        conn = get_conn()
        last_updated = get_last_updated(conn)
        logger.info(f"Última actualización en DB: {last_updated}")
        conn.close()
        conn = None

        new_games = fetch_new_games(last_updated)
        if not new_games:
            logger.info("No hay juegos nuevos o actualizados.")
            return {'status': 'ok', 'new_games': 0}

        upload_json_batches_to_s3(all_new_games=new_games)
        return {'status': 'ok', 'new_games': len(new_games)}

    except Exception:
        logger.exception("Error en la ejecución de la Lambda")
        return {'status': 'error'}
