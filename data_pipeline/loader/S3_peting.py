from dotenv import load_dotenv
import os
import boto3
import time
from tqdm import tqdm
from datetime import datetime

## Conexiones maximas de RDS 79

load_dotenv()

# Configuración
S3_BUCKET = os.getenv("S3_BUCKET")
if not S3_BUCKET:
    raise ValueError("❌ La variable de entorno S3_BUCKET no está definida. Verifica tu archivo .env.")
LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'games_pages'))
BATCH_SIZE = 60
SLEEP_SECONDS = 10
STATUS_FILE = os.path.join(os.path.dirname(__file__), 'uploaded_files.txt')

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# Inicializar cliente de S3
s3_client = boto3.client('s3')

# Leer lista de archivos ya subidos
if os.path.exists(STATUS_FILE):
    with open(STATUS_FILE, 'r') as f:
        uploaded = set(line.strip() for line in f if line.strip())
else:
    uploaded = set()

# Obtener lista de archivos pendientes
all_files = sorted([f for f in os.listdir(LOCAL_DIR) if f.endswith('.json') and f not in uploaded])

# Subida por lotes
for i in range(0, len(all_files), BATCH_SIZE):
    batch = all_files[i:i+BATCH_SIZE]
    log(f"Subiendo lote {i+1} a {i+len(batch)}...")

    for filename in tqdm(batch, desc=f"Lote {i+1} a {i+len(batch)}"):
        file_path = os.path.join(LOCAL_DIR, filename)
        s3_key = f"games_pages/{filename}"

        try:
            s3_client.upload_file(file_path, S3_BUCKET, s3_key)
            with open(STATUS_FILE, 'a') as f:
                f.write(filename + '\n')
        except Exception as e:
            log(f"❌ Error subiendo {filename}: {e}")

    log(f"✅ Lote {i+1} a {i+len(batch)} subido. Esperando {SLEEP_SECONDS}s...")
    time.sleep(SLEEP_SECONDS)

log("🎉 Subida completada.shiuuuuuuu!!")
