import os
import json
import time
import requests
import logging
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv() # Cargar variables de entorno desde .env

# ========= CONFIGURACIÓN =========
RAWG_API_KEY = os.getenv("RAWG_API_KEY")
SAVE_DIR = "data/raw"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "extraccion_rawg.log")
PAGE_SIZE = 40
WAIT_BETWEEN_CALLS = 1

# ========= PREPARACIÓN =========
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ========= LOGGING =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)

# ========= FUNCIONES =========
def get_games_page(page: int):
    url = "https://api.rawg.io/api/games"
    params = {
        "key": RAWG_API_KEY,
        "page": page,
        "page_size": PAGE_SIZE
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Fallo al solicitar la página {page}: {e}")
        return None

def save_page_to_file(data, page):
    filename = os.path.join(SAVE_DIR, f"games_page_{page}.json")
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Guardada página {page} en {filename}")
    except Exception as e:
        logging.error(f"No se pudo guardar la página {page}: {e}")

def extract_all_games_resume_safe():
    logging.info("Iniciando extracción RAWG ...")

    page = 1
    first_page = get_games_page(page)
    if not first_page:
        logging.error("No se pudo obtener la primera página.")
        return

    total_count = first_page.get("count", 0)
    total_pages = (total_count // PAGE_SIZE) + (1 if total_count % PAGE_SIZE else 0)
    logging.info(f"Total de juegos estimado: {total_count} | Páginas: {total_pages}")

    filename = os.path.join(SAVE_DIR, f"games_page_{page}.json")
    if not os.path.exists(filename):
        save_page_to_file(first_page, page)
    else:
        logging.info(f"[SKIP] Página {page} ya existe.")

    for page in tqdm(range(2, total_pages + 1), desc="Extrayendo juegos"):
        filename = os.path.join(SAVE_DIR, f"games_page_{page}.json")
        if os.path.exists(filename):
            logging.debug(f"[SKIP] Página {page} ya existe.")
            continue

        data = get_games_page(page)
        if not data:
            logging.warning(f"No se pudo obtener la página {page}. Se omitirá.")
            continue

        save_page_to_file(data, page)
        time.sleep(WAIT_BETWEEN_CALLS)

    logging.info("Extracción finalizada con éxito.")

# ========= EJECUCIÓN =========
if __name__ == "__main__":
    extract_all_games_resume_safe()