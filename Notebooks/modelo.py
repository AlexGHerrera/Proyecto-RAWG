import psycopg2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
# Establecer conexión
conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASS'),
        port=os.getenv('DB_PORT', 5432)
)

# Consultar datos
query = """
SELECT
    id_games,
    name,
    genero,
    playtime,
    added,
    plataforma,
    released
FROM
    games
"""
df = pd.read_sql(query, conn)

# Cerrar conexión
conn.close()